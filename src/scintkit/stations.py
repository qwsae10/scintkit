"""Lookup in the bundled ScintPi station registry (data/station_scintpi_codes.csv)."""

import os
import re
from importlib import resources

import numpy as np
import pandas as pd

from scintkit import filenames

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius
SCINTPI_TYPES = {"SC2", "SC3", "SC4"}


def load_station_codes() -> pd.DataFrame:
    """The station registry as a DataFrame."""
    csv = resources.files("scintkit").joinpath("data/station_scintpi_codes.csv")
    with csv.open("r", encoding="latin1", newline="") as f:
        return pd.read_csv(f)


def _station_dict(row: pd.Series) -> dict:
    return {
        column: None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
        for column, value in row.items()
    }


def _nearest_station(stations, latitude, longitude, max_distance_km) -> dict | None:
    """Closest registry entry within ``max_distance_km`` (haversine distance)."""
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180 degrees")

    lat1 = np.radians(latitude)
    lat2 = np.radians(pd.to_numeric(stations["Latitude"], errors="coerce"))
    delta_lon = np.radians(pd.to_numeric(stations["Longitude"], errors="coerce") - longitude)
    haversine = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))

    if distances.isna().all() or distances.min() > max_distance_km:
        return None

    nearest = stations[np.isclose(distances, distances.min())]
    if len(nearest) > 1:
        # A colocated non-ScintPi receiver (e.g. UTD's Septentrio) should not
        # make a unique ScintPi coordinate match ambiguous.
        scintpi = nearest[nearest["Type"].astype(str).str.upper().isin(SCINTPI_TYPES)]
        if len(scintpi) != 1:
            codes = ", ".join(nearest["Code"].astype(str))
            raise ValueError("coordinates match multiple station entries equally: " + codes)
        nearest = scintpi

    return _station_dict(nearest.iloc[0])


def identify_station(
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    filename: str | os.PathLike | None = None,
    max_distance_km: float = 3.0,
) -> dict | None:
    """Registry row (as a dict) for a station given coordinates or a file name.

    SC4 names match on their station prefix; older names on their encoded
    coordinates and ScintPi generation. None if nothing is within range.
    """
    if filename is not None and (latitude is not None or longitude is not None):
        raise ValueError("provide either filename or latitude/longitude, not both")
    if filename is None and (latitude is None or longitude is None):
        raise ValueError("provide filename or both latitude and longitude")

    stations = load_station_codes()

    if filename is not None:
        basename = os.path.basename(os.fspath(filename)).lower()
        prefixes = stations["SC4 Prefix"].astype("string").str.lower()
        prefix_matches = stations[
            prefixes.notna()
            & prefixes.map(basename.startswith, na_action="ignore")
        ]
        if len(prefix_matches) == 1:
            return _station_dict(prefix_matches.iloc[0])
        if len(prefix_matches) > 1:
            codes = ", ".join(prefix_matches["Code"].astype(str))
            raise ValueError("filename prefix matches multiple stations: " + codes)

        coords = filenames.coordinates(filename)
        if coords is None:
            raise ValueError("filename has no recognized station prefix or coordinates")
        latitude, longitude = coords

        generation = re.search(r"scintpi[_-]?([234])", basename)
        if generation:
            stations = stations[
                stations["Type"].astype(str).str.upper() == f"SC{generation.group(1)}"
            ]

    return _nearest_station(stations, latitude, longitude, max_distance_km)
