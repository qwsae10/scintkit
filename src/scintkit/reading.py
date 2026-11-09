"""ScintPi binary readers, raw -> lvl0 Parquet conversion and lvl0 reading."""

import logging
import os
import struct
import zipfile

import numpy as np
import pandas as pd

from scintkit import schema
from scintkit.constants import (
    DEFAULT_LEAP_SECONDS,
    GPS_EPOCH,
    INVALID_SVID,
    LVL0_DROPPED_COLUMNS,
    SUPPORTED_FIRMWARE,
    V326_HEADER_FORMAT,
)
from scintkit.filenames import firmware_version, output_path
from scintkit.timing import detect_sampling_rate

logger = logging.getLogger(__name__)

V325_RECORD = np.dtype([
    ('week', np.int32), ('towe', np.float32),
    ('leap', np.uint8), ('cons', np.uint8), ('sats', np.uint8), ('svid', np.uint8),
    ('elev', np.int8), ('azim', np.int32),
    ('snr1', np.uint8), ('snr2', np.uint8), ('snr3', np.uint8),
    ('pst1', np.uint8), ('pst2', np.uint8), ('pst3', np.uint8),
    ('rst1', np.uint8), ('rst2', np.uint8), ('rst3', np.uint8),
    ('cph1', np.float64), ('cph2', np.float64), ('cph3', np.float64),
    ('rng1', np.float64), ('rng2', np.float64), ('rng3', np.float64),
    ('lon', np.float32), ('lat', np.float32), ('hei', np.float32),
], align=True)

V326_RECORD = np.dtype([
    ('towe', np.float32), ('cons', np.uint8), ('sats', np.uint8), ('svid', np.uint8),
    ('elev', np.int8), ('azim', np.int32),
    ('snr1', np.uint8), ('snr2', np.uint8),
    ('pst1', np.uint8), ('pst2', np.uint8), ('rst1', np.uint8), ('rst2', np.uint8),
    ('cph1', np.float64), ('cph2', np.float64), ('rng1', np.float64), ('rng2', np.float64),
    ('lck1', np.int32), ('lck2', np.int32),
], align=True)


class UnsupportedFirmware(ValueError):
    """The file's firmware version has no reader."""


def _records(buf, dtype, offset=0):
    return pd.DataFrame(np.frombuffer(buf, dtype, count=(len(buf) - offset) // dtype.itemsize,
                                      offset=offset))


def readv325(buf):
    """v325 records (96 bytes each, no header) -> ``(frame, leap seconds)``."""
    df = _records(buf, V325_RECORD)
    leap = int(np.bincount(df["leap"]).argmax()) if len(df) else 0
    return df, leap


def readv326(buf):
    """v326c-f: one 64-byte header record (towe, leap, ..., GPS week), then 64-byte records."""
    header = struct.unpack(V326_HEADER_FORMAT, buf[:struct.calcsize(V326_HEADER_FORMAT)])
    df = _records(buf, V326_RECORD, offset=V326_RECORD.itemsize)
    df["week"] = header[-1]
    return df, header[1]


READERS = {"v325": readv325, **{v: readv326 for v in SUPPORTED_FIRMWARE if v.startswith("v326")}}


def _leap_or_default(leap, name):
    if leap:
        return int(leap)
    logger.warning("%s: no leap-second count, assuming %d s", name, DEFAULT_LEAP_SECONDS)
    return DEFAULT_LEAP_SECONDS


def normalise_geometry(df):
    """Elevation/azimuth as float32 degrees; unknown geometry (elev outside ±90, azim outside 0-360) -> NaN."""
    elev = df["elev"].astype(np.float32)
    azim = df["azim"].astype(np.float32)
    unknown = ~elev.between(-90, 90) | ~azim.between(0, 360)
    df["elev"] = elev.mask(unknown)
    df["azim"] = azim.mask(unknown)
    return df


def read_raw(path):
    """Raw ``.bin``/``.bin.zip`` file -> ``(frame, leap seconds)``; ``datetime`` is UTC.

    Raises UnsupportedFirmware unless the name carries one of SUPPORTED_FIRMWARE.
    """
    path = os.fspath(path)
    firmware = firmware_version(path)
    if firmware not in READERS:
        raise UnsupportedFirmware(
            f"firmware {firmware} is not supported (supported: {', '.join(SUPPORTED_FIRMWARE)}): "
            f"{os.path.basename(path)}"
        )
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            buf = zf.read(zf.namelist()[0])
    else:
        with open(path, "rb") as f:
            buf = f.read()

    df, leap = READERS[firmware](buf)
    leap = _leap_or_default(leap, os.path.basename(path))
    df = df[~((df["cons"] == 6) & (df["svid"] == INVALID_SVID))]
    df = df.assign(datetime=(
        GPS_EPOCH
        + pd.to_timedelta(df["week"].astype(np.int64), unit="W")
        + pd.to_timedelta(df["towe"].astype(np.float64), unit="s")
        - pd.Timedelta(seconds=leap)
    ))
    df = df.drop(columns=["week", "towe", *LVL0_DROPPED_COLUMNS], errors="ignore")
    return normalise_geometry(df).reset_index(drop=True), leap


def to_lvl0(path, out_dir, overwrite=False, compression_level=6):
    """Convert a raw file to ``out_dir/<stem>_lvl0.pq`` (UTC, with file metadata); return its path.

    The reader is picked from the firmware tag in the file name.
    """
    out = output_path(path, out_dir, "lvl0")
    if os.path.exists(out) and not overwrite:
        logger.info("lvl0 exists: %s", out)
        return out
    logger.info("read: %s", path)
    df, leap = read_raw(path)
    meta = schema.file_metadata(path, firmware_version(path), detect_sampling_rate(df), leap)
    schema.write(df, out, meta, compression_level=compression_level)
    logger.info("wrote: %s", out)
    return out


def read_lvl0(path):
    """lvl0 Parquet -> ``(frame, metadata)`` with UTC ``datetime`` and signed, NaN-for-unknown geometry.

    Files without a ``time_scale`` entry predate it: they are in GPS time and
    their elevation was cast to uint8 (165 = -91, 216-255 = -40..-1).
    """
    meta = schema.read_metadata(path)
    df = pd.read_parquet(path)
    if "time_scale" in meta:
        return df, meta

    name = os.path.basename(os.fspath(path))
    leap = _leap_or_default(int(df["leap"].mode()[0]) if "leap" in df else 0, name)
    df["datetime"] -= pd.Timedelta(seconds=leap)
    if df["elev"].dtype == np.uint8:
        df["elev"] = df["elev"].astype(np.int16).where(df["elev"] < 128, df["elev"].astype(np.int16) - 256)
    df = normalise_geometry(df.drop(columns=list(LVL0_DROPPED_COLUMNS), errors="ignore"))
    meta.update(firmware=firmware_version(path), time_scale="UTC", leap_seconds=str(leap),
                source_file=name)
    return df, meta
