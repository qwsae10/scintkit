"""Metadata encoded in ScintPi file names.

Example: ``scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq``.
"""

import os
import re
from datetime import datetime

_VERSION = re.compile(r"_v(\d+)")
_START_TIME = re.compile(r"_(\d{8})_(\d{4})_")
_COORDINATES = re.compile(
    r"_([0-9]+(?:\.[0-9]+)?)([EW])_([0-9]+(?:\.[0-9]+)?)([NS])",
    re.IGNORECASE,
)

# Older receivers wrote decimal degrees multiplied by this factor.
LEGACY_COORDINATE_SCALE = 10_000


def firmware_version(filename: str | os.PathLike) -> str | None:
    """Firmware version such as ``"v326"``, or None if the name has none."""
    match = _VERSION.search(os.fspath(filename))
    return f"v{match.group(1)}" if match else None


def start_time(filename: str | os.PathLike) -> datetime | None:
    """Recording start time (as written by the receiver), or None."""
    match = _START_TIME.search(os.path.basename(os.fspath(filename)))
    if match is None:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")


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
