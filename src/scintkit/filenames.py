"""Metadata encoded in ScintPi file names.

Examples: ``scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq`` (ScintPi 2/3) and
``cr01312m00.26_`` (ScintPi 4 SBF: station prefix, day of year, hour a-x, minute, year).
"""

import os
import re
from datetime import datetime, timedelta

_VERSION = re.compile(r"_(v\d+[a-z]*)(?=[_.]|$)", re.IGNORECASE)
_START_TIME = re.compile(r"_(\d{8})_(\d{4})_")
_COORDINATES = re.compile(
    r"_([0-9]+(?:\.[0-9]+)?)([EW])_([0-9]+(?:\.[0-9]+)?)([NS])",
    re.IGNORECASE,
)
# SBF name: <4-char station prefix><day of year><hour a-x><minute>.<yy>_
_SBF_NAME = re.compile(r"^([a-z]{2}[0-9]{2})([0-9]{3})([a-x])([0-9]{2})\.([0-9]{2})_$", re.IGNORECASE)
_SUFFIXES = ("_lvl0.parquet", "_lvl0.pq", ".bin.zip", ".dat.zip", ".bin", ".dat", ".parquet", ".pq")
PRODUCT_LEVELS = ("lvl2", "lvl3", "ml_features")  # Parquet outputs that are not lvl0 inputs

# Older receivers wrote decimal degrees multiplied by this factor.
LEGACY_COORDINATE_SCALE = 10_000


def firmware_version(filename: str | os.PathLike) -> str | None:
    """Full firmware tag such as ``"v326f"``, or None if the name has none."""
    match = _VERSION.search(os.path.basename(os.fspath(filename)))
    return match.group(1).lower() if match else None


def sbf_name(filename: str | os.PathLike) -> tuple[str, datetime] | None:
    """``(station prefix, start time)`` of a ScintPi 4 SBF name, or None. Time is GPS time."""
    match = _SBF_NAME.match(os.path.basename(os.fspath(filename)))
    if match is None:
        return None
    prefix, doy, hour, minute, yy = match.groups()
    start = datetime(2000 + int(yy), 1, 1) + timedelta(
        days=int(doy) - 1, hours=ord(hour.lower()) - ord("a"), minutes=int(minute))
    return prefix.lower(), start


def start_time(filename: str | os.PathLike) -> datetime | None:
    """Recording start time (as written by the receiver), or None."""
    sbf = sbf_name(filename)
    if sbf is not None:
        return sbf[1]
    match = _START_TIME.search(os.path.basename(os.fspath(filename)))
    if match is None:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


def is_lvl0_parquet(filename: str | os.PathLike) -> bool:
    """True for a ``.pq``/``.parquet`` file that is not an lvl2/lvl3/ml_features product."""
    name = os.path.basename(os.fspath(filename)).lower()
    stem, ext = os.path.splitext(name)
    return ext in (".pq", ".parquet") and not stem.endswith(tuple(f"_{level}" for level in PRODUCT_LEVELS))


def coordinates(filename: str | os.PathLike) -> tuple[float, float] | None:
    """Receiver ``(latitude, longitude)`` in degrees, or None if not encoded."""
    match = _COORDINATES.search(os.fspath(filename))
    if match is None:
        return None

    longitude = float(match.group(1))
    latitude = float(match.group(3))
    if longitude > 180:
        longitude /= LEGACY_COORDINATE_SCALE
    if latitude > 90:
        latitude /= LEGACY_COORDINATE_SCALE

    if match.group(2).upper() == "W":
        longitude = -longitude
    if match.group(4).upper() == "S":
        latitude = -latitude
    return latitude, longitude


def output_path(filename: str | os.PathLike, out_dir: str | os.PathLike, level: str) -> str:
    """``out_dir/<input stem>_<level>.pq``, e.g. level ``"lvl3"``.

    An SBF name ``cr01312m00.26_`` gives the stem ``cr01_20261108_1200``.
    """
    name = os.path.basename(os.fspath(filename))
    sbf = sbf_name(name)
    if sbf is not None:
        return os.path.join(os.fspath(out_dir), f"{sbf[0]}_{sbf[1]:%Y%m%d_%H%M}_{level}.pq")
    for suffix in _SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return os.path.join(os.fspath(out_dir), f"{name}_{level}.pq")
