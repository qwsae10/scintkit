#!/usr/bin/env python3


# %%

import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scintkit.reading.binaryreaders import readv324, readv325, readv326

READERS = {
    "v324": readv324,
    "v325": readv325,
    "v326": readv326,
}


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    # to reduce file size, drop columns that are not needed for scintillation analysis. These can be re-added later if needed.
    cols = ["sats", "pst1", "pst2", "rst1", "rst2", "lck1", "lck2"]
    df = df.drop(columns=cols, errors="ignore")
    df = df[df.svid != 255]
    df["elev"] = df["elev"].astype("uint8")  # 0–90
    df["azim"] = df["azim"].astype("uint16")  # 0–359 or 0–360
    return df


def get_version(path: str | os.PathLike) -> str | None:
    path = os.fspath(path)
    m = re.search(r"_v(\d+)", path)
    if m:
        return f"v{m.group(1)}"
    return None


def build_output_path(
    input_file: str,
    input_root: str,
    output_root: str,
    output_suffix: str = ".pq",
) -> str:
    rel = os.path.relpath(input_file, input_root)

    if rel.endswith(".bin.zip"):
        rel = rel[:-8]
    elif rel.endswith(".bin"):
        rel = rel[:-4]
    else:
        rel = os.path.splitext(rel)[0]

    return os.path.join(output_root, rel + output_suffix)


def gpsweek_tow_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "week" in df.columns and "towe" in df.columns:
        gps_epoch = datetime(1980, 1, 6)
        df = df.copy()
        df["datetime"] = (
            gps_epoch
            + pd.to_timedelta(df["week"].astype(int), unit="W")
            + pd.to_timedelta(df["towe"].astype(float), unit="s")
        )
        df = df.drop(columns=["week", "towe"], errors="ignore")
    return df


def read_binary_file(bin_file: str, version: str) -> pd.DataFrame:
    reader = READERS.get(version)
    if reader is None:
        raise ValueError(f"unsupported version: {version}")
    return reader(bin_file)


def process_one(
    input_file: str | os.PathLike,
    input_root: str | os.PathLike | None = None,
    output_root: str | os.PathLike | None = None,
    temp_root: str = "/tmp",
    compression: str = "brotli",
    compression_level: int = 6,
    overwrite: bool = False,
    verbose: bool = True,
) -> str:
    local_tmpdir = None

    try:
        input_file = os.fspath(input_file)
        if input_root is not None:
            input_root = os.fspath(input_root)
        if output_root is not None:
            output_root = os.fspath(output_root)

        input_dir = os.path.dirname(os.path.abspath(input_file))
        if input_root is None:
            input_root = input_dir
        if output_root is None:
            output_root = input_dir
        output_file = build_output_path(
            input_file=input_file,
            input_root=input_root,
            output_root=output_root,
            output_suffix="_lvl0.pq",
        )

        if os.path.exists(output_file) and not overwrite:
            if verbose:
                print(f"skip exists: {output_file}")
            return output_file

        version = get_version(input_file)

        if version is None:
            raise ValueError(f"could not determine version from filename: {input_file}")

        local_tmpdir = tempfile.mkdtemp(prefix="scintpi_", dir=temp_root)
        local_input = os.path.join(local_tmpdir, os.path.basename(input_file))

        if verbose:
            print(f"copy: {input_file} -> {local_input}")
        shutil.copy2(input_file, local_input)

        if local_input.endswith(".bin.zip"):
            if verbose:
                print(f"extract: {local_input}")

            with zipfile.ZipFile(local_input, "r") as zf:
                zf.extractall(local_tmpdir)

            local_bin = local_input[:-4]

            if not os.path.exists(local_bin):
                raise FileNotFoundError(f"extracted .bin not found: {local_bin}")

        elif local_input.endswith(".bin"):
            local_bin = local_input

        else:
            raise ValueError(f"expected .bin or .bin.zip file: {input_file}")

        local_pq = str(Path(local_bin).with_suffix(".pq"))

        if verbose:
            print(f"read: {local_bin} ({version})")
        df = read_binary_file(local_bin, version)
        df = gpsweek_tow_to_datetime(df)

        df = drop_unnecessary_columns(df)

        if verbose:
            print(f"writing to local parquet: {local_pq}")

        df.to_parquet(
            local_pq,
            compression=compression,
            compression_level=compression_level,
            index=False,
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if verbose:
            print(f"move: {local_pq} -> {output_file}")
        shutil.move(local_pq, output_file)

        if verbose:
            print(f"done: {output_file}")

        return output_file

    except Exception as e:
        print(f"error processing {input_file}: {e}")
        return ""

    finally:
        if local_tmpdir and os.path.exists(local_tmpdir):
            try:
                shutil.rmtree(local_tmpdir)
            except Exception as e:
                print(f"cleanup failed for {local_tmpdir}: {e}")


def _process_one_star(args):
    return process_one(*args)


def process_files(
    flist: list[str],
    input_root: str,
    output_root: str,
    n_workers: int = 1,
    temp_root: str = "/tmp",
    compression: str = "brotli",
    compression_level: int = 6,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """
    Convert a list of binary files to Parquet format, with optional parallel processing (set n_workers).

    """
    args = [
        (
            f,
            input_root,
            output_root,
            temp_root,
            compression,
            compression_level,
            overwrite,
            verbose,
        )
        for f in flist
    ]

    if n_workers == 1:
        return [_process_one_star(a) for a in args]

    with Pool(n_workers) as pool:
        return pool.map(_process_one_star, args)


def process_files_slurm(
    flist: list[str],
    input_root: str,
    output_root: str,
    n_tasks: int,
    task_id: int | None = None,
    n_workers: int = 1,
    temp_root: str = "/tmp",
    compression: str = "brotli",
    compression_level: int = 6,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    chunks = np.array_split(flist, n_tasks)
    chunk = list(chunks[task_id])

    if verbose:
        print(f"task_id={task_id}, files_in_chunk={len(chunk)}")

    return process_files(
        flist=chunk,
        input_root=input_root,
        output_root=output_root,
        n_workers=n_workers,
        temp_root=temp_root,
        compression=compression,
        compression_level=compression_level,
        overwrite=overwrite,
        verbose=verbose,
    )


# %%
