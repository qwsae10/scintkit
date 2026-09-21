#!/usr/bin/env python3
"""Discover and process independent ScintPi files with a worker pool.

ScintPi filenames store longitude and latitude as signed hemisphere tokens.
The decoder accepts both 10,000x packed coordinates and ordinary decimal
degrees. This command filters files by decoded coordinates, assigns one whole
source file to each process, and writes one restart-safe output per source
file.

Example
-------
python batch_ml_features.py \
    --input-root /titan/frodrigues/scintpi_storage \
    --output-dir /titan/frodrigues/scintpi_storage/ml_features \
    --year 2024,2025,2026 --workers 4 \
    --coordinate -7.213 -35.907
"""

from __future__ import annotations

import argparse
import math
import os
import re
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

try:  # Support both ``python file.py`` and ``python -m scintkit...``.
    from .filename_metadata import (
        DEFAULT_FILENAME_COORDINATE_SCALE,
        parse_filename_coordinates,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from filename_metadata import (
        DEFAULT_FILENAME_COORDINATE_SCALE,
        parse_filename_coordinates,
    )

try:  # Support both ``python file.py`` and ``python -m scintkit...``.
    from .compute_ml_features import (
        DEFAULT_N_THRESHOLD,
        DEFAULT_S4_THRESHOLD,
        OUTPUT_COLUMNS,
        compute_features,
        write_features,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from compute_ml_features import (
        DEFAULT_N_THRESHOLD,
        DEFAULT_S4_THRESHOLD,
        OUTPUT_COLUMNS,
        compute_features,
        write_features,
    )


DEFAULT_INPUT_ROOT = Path("/titan/frodrigues/scintpi_storage")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "ml_features"
DEFAULT_YEAR = 2024
DEFAULT_WORKERS = 4
DEFAULT_SHARD_INDEX = 0
DEFAULT_SHARD_COUNT = 1
DEFAULT_PATTERN = "*/*{year}*.pq"
DEFAULT_COORDINATE_TOLERANCE_DEG = 0.0005
OUTPUT_SUFFIX = "_ml_features.pq"

_DATE_TOKEN = re.compile(r"_(?P<date>(?:19|20)\d{6})(?=_)")


@dataclass(frozen=True)
class BatchResult:
    """Serializable status returned by one file worker."""

    source: Path
    output: Path
    row_count: int
    status: str
    elapsed_seconds: float
    sample_order_grid_fallback_used: bool = False
    error: str | None = None


def filename_year(filename: str | Path) -> int:
    """Return the four-digit year in a ScintPi filename."""

    match = _DATE_TOKEN.search(Path(filename).name)
    if match is None:
        raise ValueError(f"filename does not contain a YYYYMMDD token: {filename}")
    return int(match.group("date")[:4])


def normalize_years(year: int | Iterable[int]) -> tuple[int, ...]:
    """Return unique requested years in their original order."""

    values = (year,) if isinstance(year, int) else tuple(year)
    if not values:
        raise ValueError("at least one year is required")

    years: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"year must be an integer: {value!r}")
        if not 1900 <= value <= 2099:
            raise ValueError(f"year must be between 1900 and 2099: {value}")
        if value not in years:
            years.append(value)
    return tuple(years)


def parse_year_argument(value: str) -> tuple[int, ...]:
    """Parse a comma-separated ``--year`` argument for argparse."""

    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "year must be one year or a comma-separated list, such as 2024,2025,2026"
        )
    try:
        years = tuple(int(token) for token in tokens)
        return normalize_years(years)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def coordinate_matches(
    coordinate: tuple[float, float],
    targets: list[tuple[float, float]],
    *,
    tolerance_deg: float = DEFAULT_COORDINATE_TOLERANCE_DEG,
) -> bool:
    """Return whether latitude/longitude is close to any requested target."""

    if not math.isfinite(tolerance_deg) or tolerance_deg < 0:
        raise ValueError("coordinate tolerance must be finite and nonnegative")
    latitude, longitude = coordinate
    return any(
        abs(latitude - target_latitude) <= tolerance_deg
        and abs(longitude - target_longitude) <= tolerance_deg
        for target_latitude, target_longitude in targets
    )


def discover_input_files(
    input_root: Path,
    *,
    output_dir: Path,
    year: int | Iterable[int],
    coordinates: list[tuple[float, float]],
    pattern: str = DEFAULT_PATTERN,
    coordinate_scale: float = DEFAULT_FILENAME_COORDINATE_SCALE,
    coordinate_tolerance_deg: float = DEFAULT_COORDINATE_TOLERANCE_DEG,
) -> list[Path]:
    """Discover source Parquets for one or more years and receiver sites."""

    input_root = input_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    years = normalize_years(year)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if not coordinates:
        raise ValueError("at least one target coordinate is required")
    for latitude, longitude in coordinates:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"invalid target coordinate: ({latitude}, {longitude})")

    sources: set[Path] = set()
    rendered_patterns: list[str] = []
    for requested_year in years:
        try:
            rendered_pattern = pattern.format(year=requested_year)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                "pattern may contain only the optional {year} placeholder"
            ) from error
        rendered_patterns.append(rendered_pattern)

        for path in input_root.glob(rendered_pattern):
            if not path.is_file():
                continue
            if path.parent == output_dir or path.name.endswith(OUTPUT_SUFFIX):
                continue
            try:
                if filename_year(path) != requested_year:
                    continue
                coordinate = parse_filename_coordinates(
                    path,
                    scale=coordinate_scale,
                )
            except ValueError:
                continue
            if coordinate_matches(
                coordinate,
                coordinates,
                tolerance_deg=coordinate_tolerance_deg,
            ):
                sources.add(path)

    ordered_sources = sorted(sources, key=lambda path: str(path))
    if not ordered_sources:
        year_text = ",".join(str(value) for value in years)
        raise FileNotFoundError(
            f"no source files for year(s) {year_text} under {input_root} "
            f"matched pattern(s) {rendered_patterns!r} and the requested "
            "coordinates"
        )
    return ordered_sources


def output_path_for(source: Path, output_dir: Path) -> Path:
    """Return the flat output path for one source file."""

    return output_dir / f"{source.stem}{OUTPUT_SUFFIX}"


def select_file_shard(
    sources: list[Path],
    *,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    """Select one deterministic round-robin shard of whole source files."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be between 0 and {shard_count - 1}; "
            f"received {shard_index}"
        )
    return sources[shard_index::shard_count]


def error_path_for(output: Path) -> Path:
    """Return the per-file diagnostic path for one feature output."""

    return output.with_name(f"{output.stem}_err.txt")


def _validate_unique_outputs(outputs: dict[Path, Path]) -> None:
    by_name: dict[str, list[Path]] = {}
    for source, output in outputs.items():
        by_name.setdefault(output.name, []).append(source)
    collisions = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if collisions:
        details = "; ".join(
            f"{name}: {', '.join(str(path) for path in paths)}"
            for name, paths in sorted(collisions.items())
        )
        raise ValueError(
            "source basenames must be unique when writing to one flat output "
            f"directory; collisions: {details}"
        )


def existing_output_is_current(source: Path, output: Path) -> bool:
    """Return whether an output is newer than its source and has the schema."""

    if not output.is_file() or output.stat().st_mtime < source.stat().st_mtime:
        return False
    try:
        columns = set(pq.read_schema(output).names)
    except Exception:
        return False
    return set(OUTPUT_COLUMNS).issubset(columns)


def _remove_stale_error(output: Path) -> None:
    error_path_for(output).unlink(missing_ok=True)


def _write_error_diagnostic(
    source: Path,
    output: Path,
    *,
    error: BaseException,
    traceback_text: str,
) -> None:
    diagnostic_path = error_path_for(output)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = diagnostic_path.with_name(f".{diagnostic_path.name}.{os.getpid()}.tmp")
    diagnostic = (
        f"Source: {source}\n"
        f"Output: {output}\n"
        f"Exception: {type(error).__name__}: {error}\n\n"
        f"Traceback:\n{traceback_text}"
    )
    try:
        temporary.write_text(diagnostic, encoding="utf-8")
        os.replace(temporary, diagnostic_path)
    finally:
        temporary.unlink(missing_ok=True)


def process_file_task(
    source: Path,
    output: Path,
    n_threshold: int,
    s4_threshold: float,
    overwrite: bool,
) -> BatchResult:
    """Compute and atomically write one source file inside a worker."""

    started = perf_counter()
    try:
        if not overwrite and existing_output_is_current(source, output):
            row_count = pq.ParquetFile(output).metadata.num_rows
            _remove_stale_error(output)
            return BatchResult(
                source=source,
                output=output,
                row_count=row_count,
                status="skipped",
                elapsed_seconds=perf_counter() - started,
            )

        features, clock_report = compute_features(
            source,
            n_threshold=n_threshold,
            s4_threshold=s4_threshold,
            verbose=False,
        )
        write_features(
            features,
            output,
            input_path=source,
            overwrite=overwrite or output.exists(),
        )
        _remove_stale_error(output)
        return BatchResult(
            source=source,
            output=output,
            row_count=len(features),
            status="written",
            elapsed_seconds=perf_counter() - started,
            sample_order_grid_fallback_used=bool(
                clock_report["sample_order_grid_fallback_used"]
            ),
        )
    except Exception as error:
        _write_error_diagnostic(
            source,
            output,
            error=error,
            traceback_text=traceback.format_exc(),
        )
        return BatchResult(
            source=source,
            output=output,
            row_count=0,
            status="failed",
            elapsed_seconds=perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )


def _print_result(result: BatchResult) -> None:
    clock_note = (
        ", sample-order clock fallback"
        if result.sample_order_grid_fallback_used
        else ""
    )
    print(
        f"[{result.status.upper()}] {result.source.name} -> "
        f"{result.output.name} ({result.row_count:,} rows, "
        f"{result.elapsed_seconds:.1f} s{clock_note})",
        flush=True,
    )


def run_batch(
    input_root: Path,
    output_dir: Path,
    *,
    year: int | Iterable[int],
    coordinates: list[tuple[float, float]],
    workers: int = DEFAULT_WORKERS,
    shard_index: int = DEFAULT_SHARD_INDEX,
    shard_count: int = DEFAULT_SHARD_COUNT,
    pattern: str = DEFAULT_PATTERN,
    coordinate_scale: float = DEFAULT_FILENAME_COORDINATE_SCALE,
    coordinate_tolerance_deg: float = DEFAULT_COORDINATE_TOLERANCE_DEG,
    n_threshold: int = DEFAULT_N_THRESHOLD,
    s4_threshold: float = DEFAULT_S4_THRESHOLD,
    overwrite: bool = False,
) -> list[BatchResult]:
    """Process matching whole files independently with up to ``workers``."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = output_dir.expanduser().resolve()
    years = normalize_years(year)
    all_sources = discover_input_files(
        input_root,
        output_dir=output_dir,
        year=years,
        coordinates=coordinates,
        pattern=pattern,
        coordinate_scale=coordinate_scale,
        coordinate_tolerance_deg=coordinate_tolerance_deg,
    )
    all_outputs = {
        source: output_path_for(source, output_dir) for source in all_sources
    }
    _validate_unique_outputs(all_outputs)
    sources = select_file_shard(
        all_sources,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    outputs = {source: all_outputs[source] for source in sources}
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Matching source files: {len(all_sources):,}")
    print(
        f"File shard: {shard_index + 1}/{shard_count} ({len(sources):,} files assigned)"
    )
    print(f"Workers: {workers}")
    print(f"Years: {','.join(str(value) for value in years)}")
    print(f"Target coordinates (lat, lon): {coordinates}")
    print(f"Output directory: {output_dir}", flush=True)

    if not sources:
        print("This shard has no files to process.", flush=True)
        return []

    results: list[BatchResult] = []
    if workers == 1:
        for source in sources:
            result = process_file_task(
                source,
                outputs[source],
                n_threshold,
                s4_threshold,
                overwrite,
            )
            results.append(result)
            _print_result(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_file_task,
                    source,
                    outputs[source],
                    n_threshold,
                    s4_threshold,
                    overwrite,
                ): source
                for source in sources
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    output = outputs[source]
                    _write_error_diagnostic(
                        source,
                        output,
                        error=error,
                        traceback_text=traceback.format_exc(),
                    )
                    result = BatchResult(
                        source=source,
                        output=output,
                        row_count=0,
                        status="failed",
                        elapsed_seconds=0.0,
                        error=f"{type(error).__name__}: {error}",
                    )
                results.append(result)
                _print_result(result)

    results.sort(key=lambda result: str(result.source))
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("written", "skipped", "failed")
    }
    print(
        "Complete: "
        f"{counts['written']:,} written, {counts['skipped']:,} skipped, "
        f"{counts['failed']:,} failed",
        flush=True,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the reusable batch command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--year",
        type=parse_year_argument,
        default=(DEFAULT_YEAR,),
        help="one year or a comma-separated list (for example: 2024,2025,2026)",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=DEFAULT_SHARD_INDEX,
        help="zero-based index of this machine's whole-file shard",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_SHARD_COUNT,
        help="total number of disjoint whole-file shards",
    )
    parser.add_argument(
        "--coordinate",
        dest="coordinates",
        action="append",
        nargs=2,
        type=float,
        required=True,
        metavar=("LAT", "LON"),
        help="target latitude and longitude in decimal degrees; repeat as needed",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="glob below INPUT_ROOT; may contain {year}",
    )
    parser.add_argument(
        "--filename-coordinate-scale",
        type=float,
        default=DEFAULT_FILENAME_COORDINATE_SCALE,
        help="divisor for packed filename coordinates (default: 10000)",
    )
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=DEFAULT_COORDINATE_TOLERANCE_DEG,
        help="absolute per-axis match tolerance in degrees (default: 0.0005)",
    )
    parser.add_argument("--n-thresh", type=int, default=DEFAULT_N_THRESHOLD)
    parser.add_argument("--s4-thresh", type=float, default=DEFAULT_S4_THRESHOLD)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute outputs that already satisfy the current schema",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print matching inputs without computing features",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run discovery and return nonzero when any individual file fails."""

    arguments = build_parser().parse_args(argv)
    coordinates = [tuple(values) for values in arguments.coordinates]
    if arguments.list_only:
        sources = discover_input_files(
            arguments.input_root,
            output_dir=arguments.output_dir,
            year=arguments.year,
            coordinates=coordinates,
            pattern=arguments.pattern,
            coordinate_scale=arguments.filename_coordinate_scale,
            coordinate_tolerance_deg=arguments.coordinate_tolerance,
        )
        sources = select_file_shard(
            sources,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
        for source in sources:
            print(source)
        print(
            f"Shard {arguments.shard_index + 1}/{arguments.shard_count} "
            f"contains {len(sources):,} source files"
        )
        return 0

    results = run_batch(
        arguments.input_root,
        arguments.output_dir,
        year=arguments.year,
        coordinates=coordinates,
        workers=arguments.workers,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
        pattern=arguments.pattern,
        coordinate_scale=arguments.filename_coordinate_scale,
        coordinate_tolerance_deg=arguments.coordinate_tolerance,
        n_threshold=arguments.n_thresh,
        s4_threshold=arguments.s4_thresh,
        overwrite=arguments.overwrite,
    )
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
