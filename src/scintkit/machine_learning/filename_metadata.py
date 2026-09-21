"""Decode metadata embedded in ScintPi filenames."""

from __future__ import annotations

import math
import re
from pathlib import Path

DEFAULT_FILENAME_COORDINATE_SCALE = 10_000.0

_COORDINATE_TOKEN = re.compile(
    r"_(?P<value>\d+(?:\.\d+)?)(?P<hemisphere>[NSEW])(?=_|\.|$)",
    flags=re.IGNORECASE,
)


def parse_filename_coordinates(
    filename: str | Path,
    *,
    scale: float = DEFAULT_FILENAME_COORDINATE_SCALE,
) -> tuple[float, float]:
    """Return ``(latitude, longitude)`` from packed or decimal-degree tokens.

    Older ScintPi names multiply coordinate magnitudes by ``scale``. Newer
    names store ordinary decimal degrees. A token already within the valid
    range of its latitude or longitude axis is treated as decimal degrees;
    otherwise it is decoded as a packed value.
    """

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("filename coordinate scale must be finite and positive")

    decoded: dict[str, float] = {}
    for match in _COORDINATE_TOKEN.finditer(Path(filename).name):
        hemisphere = match.group("hemisphere").upper()
        axis = "latitude" if hemisphere in {"N", "S"} else "longitude"
        if axis in decoded:
            raise ValueError(
                f"filename contains multiple {axis} tokens: {Path(filename).name}"
            )

        raw_magnitude = float(match.group("value"))
        axis_limit = 90.0 if axis == "latitude" else 180.0
        magnitude = (
            raw_magnitude if raw_magnitude <= axis_limit else raw_magnitude / scale
        )
        if not math.isfinite(magnitude) or magnitude > axis_limit:
            raise ValueError(f"decoded {axis} is outside valid bounds: {raw_magnitude}")

        sign = -1.0 if hemisphere in {"S", "W"} else 1.0
        decoded[axis] = sign * magnitude

    if set(decoded) != {"latitude", "longitude"}:
        raise ValueError(
            "filename must contain one N/S token and one E/W token: "
            f"{Path(filename).name}"
        )
    return decoded["latitude"], decoded["longitude"]
