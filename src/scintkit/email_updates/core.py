"""Scan the raw-data archive for days on which each station delivered files."""

import glob
import os
import re

import numpy as np
import pandas as pd

from scintkit import filenames
from scintkit.stations import load_station_codes

MAX_MATCH_DEGREES = 0.02  # file coordinates must be this close to a station


def load_targets(csv_path=None):
    """Station dicts from the registry (or a CSV), sorted by descending latitude."""
    if csv_path is None:
        stations = load_station_codes()
    else:
        stations = pd.read_csv(csv_path, encoding="latin1")

    required = ["Station Location", "Latitude", "Longitude", "Code", "Type"]
    missing = [c for c in required if c not in stations.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if "SC4 Prefix" not in stations.columns:
        stations["SC4 Prefix"] = pd.NA

    stations["Latitude"] = pd.to_numeric(stations["Latitude"], errors="coerce")
    stations["Longitude"] = pd.to_numeric(stations["Longitude"], errors="coerce")
    stations = stations.dropna(subset=["Latitude", "Longitude"])

    targets = []
    for idx, row in stations.iterrows():
        lat, lon = float(row["Latitude"]), float(row["Longitude"])
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            continue
        code = str(row["Code"]).strip()
        station_type = str(row["Type"]).strip().upper() if pd.notnull(row["Type"]) else ""
        prefix = str(row["SC4 Prefix"]).strip().lower() if pd.notna(row["SC4 Prefix"]) else ""
        targets.append({
            "csv_index": idx,
            "name": str(row["Station Location"]).strip(),
            "lat": lat,
            "lon": lon,
            "code": "" if code.lower() == "nan" else code,
            "version": station_type or "UNK",
            "sc4_prefix": prefix or None,
            "valid_times": set(),
        })

    targets.sort(key=lambda t: t["lat"], reverse=True)
    print(f"Loaded {len(targets)} target rows.")
    return targets


def scan_legacy_files(plot_targets, cutoff, base_dir):
    """Mark days with ScintPi 2/3 files, matched to stations by coordinates and version."""
    print("Scanning legacy ScintPi 2/3 files...")
    pattern = os.path.join(base_dir, "*", "*")
    paths = glob.glob(f"{pattern}.bin.zip") + glob.glob(f"{pattern}.dat.zip")
    cutoff_day = cutoff.normalize()

    for path in paths:
        name = os.path.basename(path)
        try:
            start = filenames.start_time(name)
        except ValueError:  # impossible date in the file name
            continue
        coords = filenames.coordinates(name)
        if start is None or coords is None:
            continue
        day = pd.Timestamp(start).normalize()
        if day < cutoff_day:
            continue
        lat, lon = coords
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            continue

        version = "SC2" if "scintpi2" in path.lower() else "SC3"
        best, best_dist = None, MAX_MATCH_DEGREES
        for target in plot_targets:
            dist = np.hypot(lat - target["lat"], lon - target["lon"])
            if dist <= best_dist and target["version"] == version:
                best, best_dist = target, dist
        if best:
            best["valid_times"].add(day)


def scan_sc4_files(plot_targets, cutoff, base_dir, sc4_dict=None):
    """Mark days with ScintPi 4 files, matched by the SC4 prefixes in the targets.

    ``sc4_dict`` ({prefix: station code}) overrides the prefixes from the targets.
    """
    print("Scanning ScintPi 4 files...")
    if sc4_dict is None:
        sc4_dict = {}
        for target in plot_targets:
            prefix = target.get("sc4_prefix")
            if not prefix:
                continue
            if sc4_dict.setdefault(prefix, target["code"]) != target["code"]:
                raise ValueError(f"SC4 prefix {prefix!r} maps to multiple station codes")
    else:
        sc4_dict = {str(p).strip().lower(): code for p, code in sc4_dict.items()}

    by_code = {}
    for target in plot_targets:
        by_code.setdefault(target["code"], target)
    cutoff_day = cutoff.normalize()

    for prefix, code in sc4_dict.items():
        target = by_code.get(code)
        if target is None:
            continue
        for path in glob.glob(os.path.join(base_dir, "*", "*", f"{prefix}*_")):
            if len(os.path.basename(path)) < 10:
                continue
            day = pd.to_datetime(os.path.basename(os.path.dirname(path)),
                                 format="%Y%m%d", errors="coerce")
            if pd.notnull(day) and day >= cutoff_day:
                target["valid_times"].add(day.normalize())


def scan_septentrio_files(plot_targets, cutoff, base_dir):
    """Mark days with Septentrio files named ``CSS_DDDN.YY_`` (and ``.ismr``).

    ``DDD`` is the day of year, ``N`` the session number, ``YY`` the two-digit
    year. Applies to the single target whose version is ``SEPT``.
    """
    print("Scanning Septentrio files...")
    sept_targets = [t for t in plot_targets if t["version"] == "SEPT"]
    if len(sept_targets) != 1:
        raise ValueError(f"expected exactly one SEPT target, found {len(sept_targets)}")
    target = sept_targets[0]

    pattern = re.compile(r"^CSS_(?P<doy>\d{3})\d\.(?P<year>\d{2})_(?:\.ismr)?$", re.IGNORECASE)
    cutoff_day = cutoff.normalize()

    for path in glob.glob(os.path.join(base_dir, "CSS_*")):
        match = pattern.match(os.path.basename(path))
        if match is None:
            continue
        year = 2000 + int(match.group("year"))
        doy = int(match.group("doy"))
        day = pd.to_datetime(f"{year:04d}{doy:03d}", format="%Y%j", errors="coerce")
        if pd.isna(day) or day.year != year or day.dayofyear != doy or day < cutoff_day:
            continue
        target["valid_times"].add(day.normalize())
