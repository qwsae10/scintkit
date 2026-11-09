"""ScintPi binary readers and raw -> lvl0 Parquet conversion."""

import logging
import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scintkit.constants import GPS_EPOCH, INVALID_SVID, LVL0_COMPRESSION, LVL0_DROPPED_COLUMNS
from scintkit.filenames import firmware_version

logger = logging.getLogger(__name__)


def readv324(path):
    """Read a v324 whitespace-separated text log."""
    cols2read = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]
    colnames = ['week', 'towe', 'cons', 'svid', 'elev', 'azim',
                'snr1', 'snr2', 'cph1', 'cph2', 'rng1', 'rng2']
    df = pd.read_csv(path, sep=r'\s+', header=None, usecols=cols2read)
    df.columns = colnames
    return df


def readv325(path):
    """Read a v325 binary file (three signals, receiver position)."""
    dt = np.dtype([
        ('week', np.int32), ('towe', np.float32),
        ('leap', np.uint8), ('cons', np.uint8),
        ('sats', np.uint8), ('svid', np.uint8),
        ('elev', np.int8), ('azim', np.int32),
        ('snr1', np.uint8), ('snr2', np.uint8),
        ('snr3', np.uint8), ('pst1', np.uint8),
        ('pst2', np.uint8), ('pst3', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8),
        ('rst3', np.uint8), ('cph1', np.float64),
        ('cph2', np.float64), ('cph3', np.float64),
        ('rng1', np.float64), ('rng2', np.float64),
        ('rng3', np.float64), ('lon', np.float32),
        ('lat', np.float32), ('hei', np.float32),
    ], align=True)
    arr = np.fromfile(path, dtype=dt)
    return pd.DataFrame.from_records(arr)


def readv326(path):
    """Read a v326 binary file; the GPS week is stored once in the header."""
    rec_dt = np.dtype([
        ('towe', np.float32), ('cons', np.uint8),
        ('sats', np.uint8), ('svid', np.uint8),
        ('elev', np.int8), ('azim', np.int32),
        ('snr1', np.uint8), ('snr2', np.uint8),
        ('pst1', np.uint8), ('pst2', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8),
        ('cph1', np.float64), ('cph2', np.float64),
        ('rng1', np.float64), ('rng2', np.float64),
        ('lck1', np.int32), ('lck2', np.int32),
    ], align=True)
    header_size = 60
    records_offset = 64 * 2  # the header occupies the first two records
    with open(path, 'rb') as f:
        header = struct.unpack('@fBbiBBBBBBddddi', f.read(header_size))
    week = header[-1]
    df = pd.DataFrame.from_records(np.fromfile(path, dtype=rec_dt, offset=records_offset))
    df['week'] = week
    return df


READERS = {
    "v324": readv324,
    "v325": readv325,
    "v326": readv326,
}


def read_binary_file(bin_file, version):
    """Raw receiver records for a firmware version such as ``"v326"``."""
    reader = READERS.get(version)
    if reader is None:
        raise ValueError(f"unsupported version: {version}")
    return reader(bin_file)


def gpsweek_tow_to_datetime(df):
    """Replace ``week``/``towe`` with a ``datetime`` column (GPS time scale)."""
    df["datetime"] = (
        GPS_EPOCH
        + pd.to_timedelta(df["week"].astype(int), unit="W")
        + pd.to_timedelta(df["towe"].astype(float), unit="s")
    )
    return df.drop(columns=["week", "towe"])


def drop_unnecessary_columns(df):
    """Drop receiver status columns and untracked channels; shrink angle dtypes."""
    df = df.drop(columns=list(LVL0_DROPPED_COLUMNS), errors='ignore')
    df = df[df.svid != INVALID_SVID]
    df["elev"] = df["elev"].astype("uint8")
    df["azim"] = df["azim"].astype("uint16")
    return df


def build_output_path(input_file, input_root, output_root, output_suffix=".pq"):
    """Mirror ``input_file``'s path below ``input_root`` into ``output_root``."""
    rel = os.path.relpath(input_file, input_root)
    if rel.endswith(".bin.zip"):
        rel = rel[:-8]
    elif rel.endswith(".bin"):
        rel = rel[:-4]
    else:
        rel = os.path.splitext(rel)[0]
    return os.path.join(output_root, rel + output_suffix)


def convert_to_lvl0(
    input_file,
    input_root=None,
    output_root=None,
    temp_root=None,
    compression_level=6,
    overwrite=False,
):
    """Convert a raw ``.bin``/``.bin.zip`` file to ``*_lvl0.pq`` and return its path.

    Outputs go next to the input unless ``input_root``/``output_root`` are given.
    Work happens in a scratch directory under ``temp_root`` (system default if None).
    """
    input_file = os.fspath(input_file)
    input_dir = os.path.dirname(os.path.abspath(input_file))
    output_file = build_output_path(
        input_file,
        os.fspath(input_root) if input_root is not None else input_dir,
        os.fspath(output_root) if output_root is not None else input_dir,
        output_suffix="_lvl0.pq",
    )
    if os.path.exists(output_file) and not overwrite:
        logger.info("skip exists: %s", output_file)
        return output_file

    version = firmware_version(input_file)
    if version is None:
        raise ValueError(f"could not determine version from filename: {input_file}")

    with tempfile.TemporaryDirectory(prefix="scintpi_", dir=temp_root) as local_tmpdir:
        local_input = os.path.join(local_tmpdir, os.path.basename(input_file))
        logger.debug("copy: %s -> %s", input_file, local_input)
        shutil.copy2(input_file, local_input)

        if local_input.endswith(".bin.zip"):
            with zipfile.ZipFile(local_input, "r") as zf:
                zf.extractall(local_tmpdir)
            local_bin = local_input[:-4]
            if not os.path.exists(local_bin):
                raise FileNotFoundError(f"extracted .bin not found: {local_bin}")
        elif local_input.endswith(".bin"):
            local_bin = local_input
        else:
            raise ValueError(f"expected .bin or .bin.zip file: {input_file}")

        logger.info("read: %s (%s)", input_file, version)
        df = read_binary_file(local_bin, version)
        df = gpsweek_tow_to_datetime(df)
        df = drop_unnecessary_columns(df)

        local_pq = str(Path(local_bin).with_suffix(".pq"))
        df.to_parquet(
            local_pq,
            compression=LVL0_COMPRESSION,
            compression_level=compression_level,
            index=False,
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.move(local_pq, output_file)

    logger.info("wrote: %s", output_file)
    return output_file
