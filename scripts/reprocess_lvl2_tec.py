#!/usr/bin/env python3
"""Atomically recompute TEC columns in existing level-2 Parquet files."""

import argparse
import os
from pathlib import Path

import pandas as pd

from scintkit.tec import add_tec_columns


def reprocess_file(path: Path, compression: str = "brotli") -> dict[str, int]:
    frame = pd.read_parquet(path)
    counts = {}

    for pair in ("12", "13"):
        required = {
            f"cph{pair[0]}",
            f"cph{pair[1]}",
            f"rng{pair[0]}",
            f"rng{pair[1]}",
            f"freq_{pair[0]}",
            f"freq_{pair[1]}",
        }
        if required.issubset(frame.columns):
            # Level-2 files are one-second products.
            frame = add_tec_columns(frame, pair=pair, fs=1)
            counts[f"tec_cph{pair}"] = int(frame[f"tec_cph{pair}"].notna().sum())
            counts[f"tec_rng{pair}"] = int(frame[f"tec_rng{pair}"].notna().sum())

    temporary = path.with_name(f".{path.name}.tec-reprocess.tmp")
    try:
        frame.to_parquet(
            temporary,
            compression=compression,
            compression_level=6 if compression == "brotli" else None,
            index=False,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="*_lvl2.pq")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="include derived files whose names start with 'total_'",
    )
    args = parser.parse_args()

    paths = sorted(args.folder.glob(args.pattern))
    if not args.include_aggregates:
        paths = [path for path in paths if not path.name.startswith("total_")]
    if args.limit is not None:
        paths = paths[: args.limit]

    print(f"Reprocessing {len(paths)} level-2 files in {args.folder}", flush=True)
    for number, path in enumerate(paths, start=1):
        counts = reprocess_file(path)
        summary = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"[{number}/{len(paths)}] {path.name}: {summary}", flush=True)


if __name__ == "__main__":
    main()
