"""Shared reference data used across ScintKit."""

import os
import re
from importlib import resources

import numpy as np
import pandas as pd


_FILENAME_COORDINATES = re.compile(
    r"_([0-9]+(?:\.[0-9]+)?)([EW])_([0-9]+(?:\.[0-9]+)?)([NS])",
    re.IGNORECASE,
)


def load_station_codes() -> pd.DataFrame:
    """Load the maintained ScintPi station registry bundled with ScintKit."""
    csv_resource = resources.files(__package__).joinpath(
        "station_scintpi_codes.csv"
    )
    with csv_resource.open("r", encoding="latin1", newline="") as csv_file:
        return pd.read_csv(csv_file)


def _coordinates_from_filename(filename: str | os.PathLike) -> tuple[float, float] | None:
    match = _FILENAME_COORDINATES.search(os.fspath(filename))
    if match is None:
        return None

    longitude = float(match.group(1))
    latitude = float(match.group(3))

    # Older receivers stored decimal degrees multiplied by 10,000.
    if longitude > 180:
        longitude /= 10_000
    if latitude > 90:
        latitude /= 10_000

    if match.group(2).upper() == "W":
        longitude = -longitude
    if match.group(4).upper() == "S":
        latitude = -latitude

    return latitude, longitude


def _station_dict(row: pd.Series) -> dict:
    station = {}
    for column, value in row.items():
        if pd.isna(value):
            station[column] = None
        elif isinstance(value, np.generic):
            station[column] = value.item()
        else:
            station[column] = value
    return station


def _nearest_station(
    stations: pd.DataFrame,
    latitude: float,
    longitude: float,
    max_distance_km: float,
) -> dict | None:
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180 degrees")
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be greater than zero")

    station_latitudes = pd.to_numeric(stations["Latitude"], errors="coerce")
    station_longitudes = pd.to_numeric(stations["Longitude"], errors="coerce")

    lat1 = np.radians(latitude)
    lat2 = np.radians(station_latitudes)
    delta_lat = lat2 - lat1
    delta_lon = np.radians(station_longitudes - longitude)
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    distances = 2 * 6371.0088 * np.arcsin(np.sqrt(haversine))

    if distances.isna().all():
        return None

    minimum_distance = distances.min()
    if minimum_distance > max_distance_km:
        return None

    nearest = stations[np.isclose(distances, minimum_distance)]
    if len(nearest) > 1:
        # Coordinate-only lookup is used for coordinate-encoded ScintPi data.
        # A colocated standalone receiver (such as UTD's Septentrio) should
        # not make an otherwise unique ScintPi coordinate match ambiguous.
        scintpi_types = nearest["Type"].astype(str).str.upper().isin(
            {"SC2", "SC3", "SC4"}
        )
        scintpi_nearest = nearest[scintpi_types]
        if len(scintpi_nearest) == 1:
            nearest = scintpi_nearest
        else:
            codes = ", ".join(nearest["Code"].astype(str))
            raise ValueError(
                "coordinates match multiple station entries equally: " + codes
            )

    return _station_dict(nearest.iloc[0])


def identify_station(
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    filename: str | os.PathLike | None = None,
    max_distance_km: float = 3.0,
) -> dict | None:
    """Identify a station from coordinates or a ScintPi filename.

    Returns the matching row from the station registry as a dictionary, or
    ``None`` when no station is within ``max_distance_km``. SC4 filenames are
    matched by their station prefix; legacy filenames are matched using their
    encoded coordinates and receiver version.
    """
    has_coordinates = latitude is not None or longitude is not None
    if filename is not None and has_coordinates:
        raise ValueError("provide either filename or latitude/longitude, not both")
    if filename is None and (latitude is None or longitude is None):
        raise ValueError("provide filename or both latitude and longitude")

    stations = load_station_codes()

    if filename is not None:
        basename = os.path.basename(os.fspath(filename)).lower()
        prefixes = stations["SC4 Prefix"].astype("string").str.lower()
        prefix_matches = stations[
            prefixes.notna()
            & prefixes.map(
                lambda prefix: basename.startswith(prefix),
                na_action="ignore",
            )
        ]
        if len(prefix_matches) == 1:
            return _station_dict(prefix_matches.iloc[0])
        if len(prefix_matches) > 1:
            codes = ", ".join(prefix_matches["Code"].astype(str))
            raise ValueError("filename prefix matches multiple stations: " + codes)

        coordinates = _coordinates_from_filename(filename)
        if coordinates is None:
            raise ValueError("filename has no recognized station prefix or coordinates")
        latitude, longitude = coordinates

        version_match = re.search(r"scintpi[_-]?([234])", basename)
        if version_match:
            version = f"SC{version_match.group(1)}"
            stations = stations[
                stations["Type"].astype(str).str.upper() == version
            ]

    return _nearest_station(
        stations,
        latitude,
        longitude,
        max_distance_km,
    )


__all__ = ["identify_station", "load_station_codes"]
