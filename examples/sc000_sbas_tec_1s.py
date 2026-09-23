#!/usr/bin/env python3
"""Build one 1 Hz SBAS TEC Parquet from SC000 Septentrio files.

Run on a compute node with RXTools and ScintKit installed, for example:

    module load rxtools
    conda activate scintkit
    python examples/sc000_sbas_tec_1s.py \
      --input /mfs/io/groups/uars/scintpi/sc000 \
      --start 20260505 \
      --scratch /home/$USER/scratch/sc000_sbas \
      --output /home/$USER/scratch/sc000_sbas_1s_20260505_onward.pq \
      --workers 8

Each worker owns one day and processes its 15-minute files in order. RXTools
text stays on scratch and is removed after parsing. The final Parquet is
published only when all selected raw files have processed successfully.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scintkit.services.compute import add_tec_columns


RAW_NAME = re.compile(r"\.[0-9]{2}_$")
SIGNAL_SLOT = {"GEO_L1": 1, "GEO_L5": 2, "SBAS_L5": 2}
OUTPUT_COLUMNS = ["datetime", "prn", "tec_cph12", "tec_rng12", "sig_1", "sig_2"]


def parse_measurements(txt: Path) -> list[pd.DataFrame]:
    """Read only SBAS phase/range, reducing each text chunk to 1 Hz."""
    chunks = []
    reader = pd.read_csv(
        txt, header=None, skiprows=2, usecols=[0, 1, 2, 3, 5, 6],
        names=["tow", "week", "svid", "signal", "range_m", "phase_cycles"],
        chunksize=500_000, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk.loc[
            chunk["signal"].isin(SIGNAL_SLOT)
            & chunk["svid"].astype(str).str.startswith("S")
        ].copy()
        if chunk.empty:
            continue
        chunk["tow"] = pd.to_numeric(chunk["tow"], errors="coerce")
        chunk["week"] = pd.to_numeric(chunk["week"], errors="coerce")
        chunk["range_m"] = pd.to_numeric(chunk["range_m"], errors="coerce")
        chunk["phase_cycles"] = pd.to_numeric(chunk["phase_cycles"], errors="coerce")
        chunk = chunk.dropna(subset=["tow", "week"])
        if chunk.empty:
            continue
        chunk["datetime"] = (
            pd.Timestamp("1980-01-06")
            + pd.to_timedelta(chunk["week"] * 7, unit="D")
            + pd.to_timedelta(chunk["tow"], unit="s")
        ).dt.floor("s")
        chunk["slot"] = chunk["signal"].map(SIGNAL_SLOT)
        chunk = chunk.drop_duplicates(["datetime", "svid", "slot"])
        chunks.append(chunk[["datetime", "svid", "slot", "signal", "range_m", "phase_cycles"]])
    return chunks


def process_day(day_dir: str, scratch: str) -> tuple[str, str | None, int, int]:
    day = Path(day_dir)
    raw_files = sorted(p for p in day.iterdir() if p.is_file() and RAW_NAME.search(p.name))
    if not raw_files:
        return day.name, None, 0, 0

    scratch_root = Path(scratch)
    with tempfile.TemporaryDirectory(prefix=f"sc000_{day.name}_", dir=scratch_root) as tmp:
        work = Path(tmp)
        frames = []
        for raw in raw_files:
            local = work / raw.name
            shutil.copy2(raw, local)
            try:
                result = subprocess.run(
                    ["rxtools_exec.sh", "bin2asc", "-f", local.name, "--extractGenMeas"],
                    cwd=work, capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(f"RXTools failed for {raw}: {result.stderr[-2000:]}")
                texts = list(work.glob(f"{raw.name}*measurements.txt"))
                if len(texts) != 1:
                    raise RuntimeError(f"Expected one measurements file for {raw}; found {len(texts)}")
                frames.extend(parse_measurements(texts[0]))
            finally:
                local.unlink(missing_ok=True)
                for txt in work.glob(f"{raw.name}*measurements.txt"):
                    txt.unlink()

        if not frames:
            return day.name, None, len(raw_files), 0
        observations = pd.concat(frames, ignore_index=True)
        observations = observations.drop_duplicates(["datetime", "svid", "slot"])
        wide = observations.pivot(index=["datetime", "svid"], columns="slot",
                                  values=["signal", "range_m", "phase_cycles"])
        wide.columns = [f"{field}_{slot}" for field, slot in wide.columns]
        wide = wide.reset_index()
        required = ["signal_1", "signal_2", "range_m_1", "range_m_2",
                    "phase_cycles_1", "phase_cycles_2"]
        for column in required:
            if column not in wide:
                wide[column] = np.nan
        wide = wide.dropna(subset=["signal_1", "signal_2"])
        if wide.empty:
            return day.name, None, len(raw_files), 0
        wide = wide.rename(columns={
            "svid": "prn", "signal_1": "sig_1", "signal_2": "sig_2",
            "range_m_1": "rng1", "range_m_2": "rng2",
            "phase_cycles_1": "cph1", "phase_cycles_2": "cph2",
        })
        wide["freq_1"] = 1575.42
        wide["freq_2"] = 1176.45
        wide = wide.sort_values(["prn", "datetime"]).reset_index(drop=True)
        # ScintKit repairs cycle slips and levels carrier TEC to pseudorange TEC.
        wide = add_tec_columns(wide, pair="12", fs=1)
        wide = wide[OUTPUT_COLUMNS].sort_values(["datetime", "prn"])
        wide = wide.loc[wide["tec_cph12"].notna() | wide["tec_rng12"].notna()]
        if wide.empty:
            return day.name, None, len(raw_files), 0
        part = scratch_root / f"sc000_sbas_{day.name}.pq"
        wide.to_parquet(part, index=False, compression="zstd")
        return day.name, str(part), len(raw_files), len(wide)


def parse_date(value: str) -> date:
    if re.fullmatch(r"\d{8}", value):
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start", type=parse_date, required=True,
                        help="YYYY-MM-DD (YYYYMMDD is also accepted)")
    parser.add_argument("--end", type=parse_date, default=date.today(),
                        help="inclusive; default is today")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.start > args.end:
        parser.error("workers must be positive and start must be on or before end")
    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    if shutil.which("rxtools_exec.sh") is None:
        parser.error("rxtools_exec.sh is unavailable; load the RXTools module")
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    days = [p for p in sorted(args.input.iterdir()) if p.is_dir()
            and re.fullmatch(r"\d{8}", p.name)
            and args.start <= parse_date(p.name) <= args.end]
    if not days:
        parser.error("no dated input directories in the selected interval")
    selected_dates = {parse_date(p.name) for p in days}
    missing_days = []
    current = args.start
    while current <= args.end:
        if current not in selected_dates:
            missing_days.append(current.isoformat())
        current += timedelta(days=1)
    if missing_days:
        print(f"Missing {len(missing_days)} calendar days: {', '.join(missing_days)}", flush=True)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    with tempfile.TemporaryDirectory(prefix="sc000_sbas_run_", dir=args.scratch) as run_dir:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_day, str(day), run_dir): day for day in days}
            for future in as_completed(futures):
                day_name, part, n_files, n_rows = future.result()
                print(f"{day_name}: {n_files} files, {n_rows} SBAS rows", flush=True)
                results.append((day_name, part, n_files, n_rows))

        if not any(r[1] for r in results):
            raise RuntimeError("No SBAS L1/L5 TEC rows found")
        temporary_output = args.output.with_name(args.output.name + ".incomplete")
        writer = None
        try:
            for _, part, _, _ in sorted(results):
                if part is None:
                    continue
                table = pq.read_table(part)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_output, table.schema, compression="zstd")
                writer.write_table(table)
            writer.close()
            writer = None
            temporary_output.replace(args.output)
        finally:
            if writer is not None:
                writer.close()
            temporary_output.unlink(missing_ok=True)
        print(f"Wrote {sum(r[3] for r in results)} rows to {args.output}")


if __name__ == "__main__":
    main()
