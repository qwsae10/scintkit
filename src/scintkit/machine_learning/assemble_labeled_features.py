#!/usr/bin/env python3
"""Assemble selected ML feature rows into one labeled Parquet file.

Edit FEATURE_DIR and OUTPUT_FILE in the settings block, then run:

    python assemble_labeled_features.py

Coordinate tokens may use the older ScintPi 10,000x packing or ordinary
decimal degrees. Only rows matching one of the three label definitions are
written. The source feature files are never modified.
"""

# %% Settings
from pathlib import Path

FEATURE_DIR = Path("/titan/frodrigues/scintpi_storage/ml_features")
OUTPUT_FILE = FEATURE_DIR / "labeled_ml_features.pq"
FEATURE_PATTERN = "*_ml_features.pq"
OVERWRITE = False


# %% Imports and label definitions
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:  # Support both ``python file.py`` and ``python -m scintkit...``.
    from .filename_metadata import parse_filename_coordinates
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from filename_metadata import parse_filename_coordinates


COORDINATE_TOLERANCE_DEG = 0.0005
S4_MIN = 0.25

# Coordinates are (latitude, longitude) in signed decimal degrees.
SCINTILLATION_SITE = (-7.213, -35.907)
RFI_SITE = (43.271, -120.358)
MULTIPATH_SITE = (46.907, -96.793)

SCINTILLATION_TIMEZONE = "America/Fortaleza"

REQUIRED_COLUMNS = {
    "svid",
    "s4_1",
    "elevation_deg",
    "minute_timestamp_utc",
}


def coordinates_from_filename(path: Path) -> tuple[float, float]:
    """Return (latitude, longitude) from one ScintPi filename."""

    return parse_filename_coordinates(path)


def is_site(
    actual: tuple[float, float],
    target: tuple[float, float],
) -> bool:
    """Match rounded requested coordinates to the precise filename values."""

    return (
        abs(actual[0] - target[0]) <= COORDINATE_TOLERANCE_DEG
        and abs(actual[1] - target[1]) <= COORDINATE_TOLERANCE_DEG
    )


def select_and_label(frame: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Apply the quality filters and the label rule for one feature file."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"{source_path.name} is missing columns: {sorted(missing)}")

    svid = pd.to_numeric(frame["svid"], errors="coerce")
    elevation = pd.to_numeric(frame["elevation_deg"], errors="coerce")
    s4 = pd.to_numeric(frame["s4_1"], errors="coerce")

    # Defensive cleanup. The feature generator already applies these filters,
    # but doing it again guarantees they are absent from the labeled dataset.
    clean = frame.loc[svid.ne(255) & elevation.le(90)].copy()
    if clean.empty:
        return clean.assign(label=pd.Series(dtype="string"))

    elevation = elevation.loc[clean.index]
    s4 = s4.loc[clean.index]
    coordinate = coordinates_from_filename(source_path)

    if is_site(coordinate, SCINTILLATION_SITE):
        utc = pd.to_datetime(clean["minute_timestamp_utc"], errors="raise", utc=True)
        local_hour = utc.dt.tz_convert(SCINTILLATION_TIMEZONE).dt.hour
        selected = (
            elevation.gt(40) & s4.gt(S4_MIN) & (local_hour.gt(18) | local_hour.lt(4))
        )
        label = "Scintillation"
    elif is_site(coordinate, RFI_SITE):
        selected = elevation.gt(40) & s4.gt(S4_MIN)
        label = "RFI"
    elif is_site(coordinate, MULTIPATH_SITE):
        selected = elevation.gt(20) & s4.gt(S4_MIN)
        label = "Multipath"
    else:
        return clean.iloc[0:0].assign(label=pd.Series(dtype="string"))

    labeled = clean.loc[selected].copy()
    labeled["label"] = label
    return labeled


def assemble_labeled_features() -> None:
    """Stream labeled rows from all feature files into one Parquet."""

    feature_dir = FEATURE_DIR.expanduser().resolve()
    output_file = OUTPUT_FILE.expanduser().resolve()
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"feature directory does not exist: {feature_dir}")
    if output_file.exists() and not OVERWRITE:
        raise FileExistsError(
            f"output already exists: {output_file}; set OVERWRITE = True to replace it"
        )

    files = sorted(
        path
        for path in feature_dir.glob(FEATURE_PATTERN)
        if path.is_file() and path.resolve() != output_file
    )
    if not files:
        raise FileNotFoundError(
            f"no files matching {FEATURE_PATTERN!r} in {feature_dir}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(
        f".{output_file.name}.in_progress_{os.getpid()}"
    )

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    label_counts: dict[str, int] = {}
    labeled_rows = 0

    try:
        for number, feature_file in enumerate(files, start=1):
            frame = pd.read_parquet(feature_file)
            labeled = select_and_label(frame, feature_file)
            if labeled.empty:
                continue

            counts = labeled["label"].value_counts()
            for label, count in counts.items():
                label_counts[label] = label_counts.get(label, 0) + int(count)
            labeled_rows += len(labeled)

            table = pa.Table.from_pandas(labeled, preserve_index=False)
            table = table.replace_schema_metadata(None)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    temporary_file,
                    output_schema,
                    compression="zstd",
                )
            elif table.schema != output_schema:
                table = table.cast(output_schema)
            writer.write_table(table)

            if number % 100 == 0:
                print(f"Read {number:,}/{len(files):,} files")

        if writer is None:
            raise RuntimeError("no rows matched any label definition")
        writer.close()
        writer = None
        temporary_file.replace(output_file)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_file.unlink(missing_ok=True)
        raise

    print(f"Wrote {labeled_rows:,} labeled rows to {output_file}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count:,}")


# %% Run
if __name__ == "__main__":
    assemble_labeled_features()
