#!/usr/bin/env python3
"""Compute one-minute machine-learning features from raw 20 Hz ScintPi data.

The pipeline keeps the source Parquet untouched.  It reconstructs an internal
20 Hz receiver clock from row order, uses ScintKit formatting and TEC/S4
functions, computes receiver-wide common-mode delta-SNR before applying the
satellite-minute gate, and evaluates FFT features only for eligible rows.

Example
-------
python compute_ml_features.py raw_file.pq --output raw_file_ml_features.pq
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import signal

from scintkit.preprocessing.format import temp_formating
from scintkit.services.compute import add_tec_columns

try:  # Support both ``python file.py`` and ``python -m scintkit...``.
    from .receiver_clock import reconstruct_receiver_clock
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from receiver_clock import reconstruct_receiver_clock


FS_HZ: Final[float] = 20.0
EXPECTED_MINUTE_SAMPLES: Final[int] = int(FS_HZ * 60)
DEFAULT_N_THRESHOLD: Final[int] = 1190
DEFAULT_S4_THRESHOLD: Final[float] = 0.15
COMMON_BASELINE: Final[str] = "5min"
COMMON_BASELINE_MIN_FRACTION: Final[float] = 0.50
COMMON_PASS_GAP: Final[str] = "30min"
COMMON_ELEVATION_MIN_DEG: Final[float] = 10.0
COMMON_MIN_SATELLITES: Final[int] = 4

# The first positive bin of a 60 s FFT is 1/60 Hz.  It is intentionally
# included in the band described as starting at 0.017 Hz.
SPECTRAL_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0p017_0p05_hz", 1.0 / 60.0, 0.05),
    ("0p05_0p1_hz", 0.05, 0.1),
    ("0p1_0p3_hz", 0.1, 0.3),
    ("0p3_1_hz", 0.3, 1.0),
    ("1_10_hz", 1.0, 10.0),
)

REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "datetime",
    "cons",
    "svid",
    "elev",
    "azim",
    "snr1",
    "snr2",
    "cph1",
    "cph2",
    "rng1",
    "rng2",
)


def spectral_columns(prefix: str) -> list[str]:
    return [f"{prefix}_power_db_{label}" for label, _, _ in SPECTRAL_BANDS]


SNR1_SPECTRAL_COLUMNS = spectral_columns("snr1")
SNR2_SPECTRAL_COLUMNS = spectral_columns("snr2")
TEC_SPECTRAL_COLUMNS = spectral_columns("tec12")
COMMON_COLUMNS = [
    "common_delta_snr1_std_dbhz",
    "common_delta_snr1_median_dbhz",
    "common_delta_snr1_p95_dbhz",
    "common_delta_snr1_p99_dbhz",
    "common_delta_snr2_std_dbhz",
    "common_delta_snr2_median_dbhz",
    "common_delta_snr2_p95_dbhz",
    "common_delta_snr2_p99_dbhz",
]
OUTPUT_COLUMNS = [
    *SNR1_SPECTRAL_COLUMNS,
    *SNR2_SPECTRAL_COLUMNS,
    *TEC_SPECTRAL_COLUMNS,
    *COMMON_COLUMNS,
    "s4_1",
    "s4_2",
    "elevation_deg",
    "azimuth_deg",
    "n_samples",
    "n_snr2_samples",
    "n_tec12_samples",
    "n_common_delta_snr1",
    "n_common_delta_snr2",
    "prn",
    "svid",
    "constellation",
    "signal_code_1",
    "signal_code_2",
    "frequency_1_mhz",
    "frequency_2_mhz",
    "minute_timestamp_utc",
    "source_filename",
]


def integrate_periodogram_bands_db(
    values: np.ndarray,
    *,
    preprocessing: str,
    fs_hz: float = FS_HZ,
) -> dict[str, float]:
    """Return integrated periodogram powers in dB for the configured bands."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("values must be a one-dimensional signal")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite before spectral processing")

    if preprocessing == "median":
        prepared = values - np.median(values)
    elif preprocessing == "linear":
        prepared = signal.detrend(values, type="linear")
    else:
        raise ValueError("preprocessing must be 'median' or 'linear'")

    frequencies, psd = signal.periodogram(
        prepared,
        fs=fs_hz,
        window="boxcar",
        detrend=False,
        scaling="density",
    )
    result: dict[str, float] = {}
    tolerance = np.finfo(float).eps * max(1.0, fs_hz) * 8
    for label, lower, upper in SPECTRAL_BANDS:
        # Shared band-edge samples are valid trapezoid endpoints for both
        # adjacent integrals; they do not represent a finite duplicated area.
        selected = (frequencies >= lower - tolerance) & (
            frequencies <= upper + tolerance
        )
        if np.count_nonzero(selected) < 2:
            result[label] = np.nan
            continue
        power = float(np.trapezoid(psd[selected], frequencies[selected]))
        result[label] = 10.0 * np.log10(power) if power > 0 else np.nan
    return result


def _linearly_fill_small_gaps(values: np.ndarray) -> np.ndarray:
    """Interpolate finite-bracket gaps and linearly extrapolate short edges."""

    values = np.asarray(values, dtype=float)
    finite_index = np.flatnonzero(np.isfinite(values))
    if len(finite_index) < 2:
        raise ValueError("at least two finite samples are required")
    target = np.arange(len(values))
    filled = np.interp(target, finite_index, values[finite_index])

    first, second = finite_index[:2]
    if first:
        slope = (values[second] - values[first]) / (second - first)
        filled[:first] = values[first] + slope * (target[:first] - first)
    penultimate, last = finite_index[-2:]
    if last < len(values) - 1:
        slope = (values[last] - values[penultimate]) / (last - penultimate)
        filled[last + 1 :] = values[last] + slope * (target[last + 1 :] - last)
    return filled


def regular_minute_signal(
    satellite_minute: pd.DataFrame,
    value_column: str,
    *,
    minute: pd.Timestamp,
    n_threshold: int,
    fs_hz: float = FS_HZ,
) -> tuple[np.ndarray | None, int]:
    """Place values on the internal 20 Hz minute grid and fill small gaps."""

    time_ns = (
        satellite_minute["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    )
    source_values = pd.to_numeric(
        satellite_minute[value_column], errors="coerce"
    ).to_numpy(dtype=float)
    return regular_minute_values(
        time_ns,
        source_values,
        minute=minute,
        n_threshold=n_threshold,
        fs_hz=fs_hz,
    )


def regular_minute_values(
    time_ns: np.ndarray,
    source_values: np.ndarray,
    *,
    minute: pd.Timestamp,
    n_threshold: int,
    fs_hz: float = FS_HZ,
) -> tuple[np.ndarray | None, int]:
    """Regularize one value array without constructing a temporary frame."""

    time_ns = np.asarray(time_ns, dtype="int64")
    source_values = np.asarray(source_values, dtype=float)
    if len(time_ns) != len(source_values):
        raise ValueError("time and value arrays must have equal length")
    expected = int(round(fs_hz * 60))
    period_ns = int(round(1_000_000_000 / fs_hz))
    minute_ns = pd.Timestamp(minute).value
    slots = np.rint((time_ns - minute_ns) / period_ns).astype(int)
    if np.any((slots < 0) | (slots >= expected)):
        raise ValueError("satellite-minute contains samples outside its minute")
    if len(np.unique(slots)) != len(slots):
        raise ValueError("satellite-minute contains duplicate receiver epochs")

    regular = np.full(expected, np.nan, dtype=float)
    regular[slots] = source_values
    valid_count = int(np.isfinite(regular).sum())
    if valid_count <= n_threshold:
        return None, valid_count
    return _linearly_fill_small_gaps(regular), valid_count


def compute_minute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute vectorized S4/count/identifier fields before FFT work."""

    grouped = frame.groupby(["prn", "minbin"], sort=False, observed=True)
    summary = grouped.agg(
        n_samples=("snr1", "count"),
        n_snr2_samples=("snr2", "count"),
        svid=("svid", "first"),
        constellation=("cons", "first"),
        elevation_deg=("elev", "first"),
        azimuth_deg=("azim", "first"),
        signal_code_1=("sig_1", "first"),
        signal_code_2=("sig_2", "first"),
        frequency_1_mhz=("freq_1", "first"),
        frequency_2_mhz=("freq_2", "first"),
    )
    linear_snr = pd.DataFrame(
        np.power(10.0, frame[["snr1", "snr2"]].to_numpy(dtype=float) / 10.0),
        columns=["snr1", "snr2"],
    )
    group_number = grouped.ngroup().to_numpy()
    linear_grouped = linear_snr.groupby(group_number, sort=False)
    linear_mean = linear_grouped.mean().to_numpy()
    linear_std = linear_grouped.std(ddof=0).to_numpy()
    s4 = np.divide(
        linear_std,
        linear_mean,
        out=np.full_like(linear_std, np.nan),
        where=linear_mean > 0,
    )
    summary["s4_1"] = s4[:, 0]
    summary["s4_2"] = s4[:, 1]
    return summary.reset_index().rename(columns={"minbin": "minute_timestamp_utc"})


def _empty_common_features() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "minute_timestamp_utc": pd.Series(dtype="datetime64[ns]")
    }
    columns.update({column: pd.Series(dtype="float64") for column in COMMON_COLUMNS})
    columns["n_common_delta_snr1"] = pd.Series(dtype="Int64")
    columns["n_common_delta_snr2"] = pd.Series(dtype="Int64")
    return pd.DataFrame(columns)


def compute_common_mode_minute_features(
    frame: pd.DataFrame,
    *,
    fs_hz: float = FS_HZ,
    rolling_window: str = COMMON_BASELINE,
    rolling_min_fraction: float = COMMON_BASELINE_MIN_FRACTION,
    pass_gap: str = COMMON_PASS_GAP,
    elevation_min_deg: float = COMMON_ELEVATION_MIN_DEG,
    min_satellites: int = COMMON_MIN_SATELLITES,
) -> pd.DataFrame:
    """Compute the centered rolling-median common delta-SNR minute features."""

    if frame.empty:
        return _empty_common_features()
    minimum_observations = max(
        1,
        math.ceil(
            pd.Timedelta(rolling_window).total_seconds() * fs_hz * rolling_min_fraction
        ),
    )
    pass_gap_ns = pd.Timedelta(pass_gap).value
    delta = {
        "snr1": np.full(len(frame), np.nan),
        "snr2": np.full(len(frame), np.nan),
    }

    times_all = frame["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    datetime_index = pd.DatetimeIndex(times_all)
    snr_all = frame[["snr1", "snr2"]].to_numpy(dtype=float)
    for positions in frame.groupby("prn", sort=False, observed=True).indices.values():
        positions = np.asarray(positions)
        satellite_times = times_all[positions]
        split_at = np.flatnonzero(np.diff(satellite_times) > pass_gap_ns) + 1
        for pass_positions in np.split(positions, split_at):
            if not len(pass_positions):
                continue
            indexed = pd.DataFrame(
                snr_all[pass_positions],
                index=datetime_index[pass_positions],
                columns=["snr1", "snr2"],
            )
            baseline = indexed.rolling(
                rolling_window,
                center=True,
                min_periods=minimum_observations,
            ).median()
            deviations = indexed.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
            delta["snr1"][pass_positions] = deviations[:, 0]
            delta["snr2"][pass_positions] = deviations[:, 1]

    elevation = pd.to_numeric(frame["elev"], errors="coerce").to_numpy()
    elevation_ok = (elevation > elevation_min_deg) & (elevation <= 90)
    epoch_ids = frame["_receiver_epoch"].to_numpy(dtype="int32")
    unique_epoch, first_position = np.unique(epoch_ids, return_index=True)
    epoch_time_ns = np.empty(int(unique_epoch.max()) + 1, dtype="int64")
    epoch_time_ns[unique_epoch] = times_all[first_position]

    channel_results: list[pd.DataFrame] = []
    for channel in ("snr1", "snr2"):
        valid = elevation_ok & np.isfinite(delta[channel])
        if not valid.any():
            continue
        residuals = pd.DataFrame(
            {
                "receiver_epoch": epoch_ids[valid],
                "delta": delta[channel][valid],
            }
        )
        common = residuals.groupby("receiver_epoch", sort=False)["delta"].agg(
            common="median", n_satellites="count"
        )
        common.loc[common["n_satellites"] < min_satellites, "common"] = np.nan
        common = common.dropna(subset=["common"])
        if common.empty:
            continue
        common["minute_timestamp_utc"] = pd.to_datetime(
            epoch_time_ns[common.index.to_numpy(dtype="int32")]
        ).floor("min")
        minute_group = common.groupby("minute_timestamp_utc", sort=True)["common"]
        prefix = f"common_delta_{channel}"
        result = minute_group.agg(
            **{
                f"{prefix}_std_dbhz": lambda values: float(
                    np.std(values.to_numpy(dtype=float), ddof=0)
                ),
                f"{prefix}_median_dbhz": "median",
                f"{prefix}_p95_dbhz": lambda values: values.quantile(0.95),
                f"{prefix}_p99_dbhz": lambda values: values.quantile(0.99),
                f"n_common_delta_{channel}": "size",
            }
        ).reset_index()
        channel_results.append(result)

    if not channel_results:
        return _empty_common_features()
    result = channel_results[0]
    for channel_result in channel_results[1:]:
        result = result.merge(
            channel_result,
            on="minute_timestamp_utc",
            how="outer",
        )
    for column in [*COMMON_COLUMNS, "n_common_delta_snr1", "n_common_delta_snr2"]:
        if column not in result.columns:
            result[column] = np.nan
    return result.loc[
        :,
        [
            "minute_timestamp_utc",
            *COMMON_COLUMNS,
            "n_common_delta_snr1",
            "n_common_delta_snr2",
        ],
    ]


def _add_spectral_values(
    record: dict[str, object],
    *,
    prefix: str,
    values: np.ndarray | None,
    preprocessing: str,
) -> None:
    columns = spectral_columns(prefix)
    if values is None:
        record.update(dict.fromkeys(columns, np.nan))
        return
    powers = integrate_periodogram_bands_db(
        values,
        preprocessing=preprocessing,
    )
    record.update(
        {
            column: powers[label]
            for column, (label, _, _) in zip(columns, SPECTRAL_BANDS)
        }
    )


def compute_features(
    input_path: str | Path,
    *,
    n_threshold: int = DEFAULT_N_THRESHOLD,
    s4_threshold: float = DEFAULT_S4_THRESHOLD,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Compute the final eligible satellite-minute feature table."""

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not 0 <= n_threshold < EXPECTED_MINUTE_SAMPLES:
        raise ValueError(f"n_threshold must be in [0, {EXPECTED_MINUTE_SAMPLES - 1}]")
    if not np.isfinite(s4_threshold) or s4_threshold < 0:
        raise ValueError("s4_threshold must be finite and nonnegative")

    if verbose:
        print(f"Reading {input_path}")
    raw = pd.read_parquet(input_path)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in raw.columns]
    if missing:
        raise KeyError(f"input is missing required columns: {missing}")

    if verbose:
        print("Reconstructing the in-memory 20 Hz receiver clock")
    frame, clock_report = reconstruct_receiver_clock(raw, sample_rate_hz=FS_HZ)
    if verbose and clock_report.sample_order_grid_fallback_used:
        print(
            "Raw timestamps could not define a strict grid; using sample order "
            "on an internal 20 Hz grid"
        )
    del raw
    valid_satellite = frame["svid"].ne(255)
    if not valid_satellite.all():
        frame = frame.loc[valid_satellite].copy().reset_index(drop=True)
    unneeded = [
        column
        for column in frame.columns
        if column not in {*REQUIRED_INPUT_COLUMNS, "_receiver_epoch"}
    ]
    frame.drop(columns=unneeded, inplace=True)
    frame = temp_formating(frame)

    if verbose:
        print("Computing S4 and valid sample counts")
    summary = compute_minute_summary(frame)

    # This intentionally uses the complete receiver file, not the eligible
    # satellite-minute subset, and is evaluated before applying that filter.
    if verbose:
        print("Computing full-file centered 5-minute common-mode delta-SNR")
    common_features = compute_common_mode_minute_features(frame)
    eligible = summary.loc[
        summary["n_samples"].gt(n_threshold)
        & summary["n_snr2_samples"].gt(n_threshold)
        & summary["s4_1"].gt(s4_threshold)
        & summary["svid"].ne(255)
    ].copy()
    if verbose:
        print(
            f"Eligible dual-frequency satellite-minutes: {len(eligible):,} "
            f"of {len(summary):,}"
        )
    eligible = eligible.merge(
        common_features,
        on="minute_timestamp_utc",
        how="left",
    )
    if eligible.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), clock_report.to_dict()

    if verbose:
        print("Computing ScintKit carrier-phase TEC for eligible satellite tracks")
    eligible_prns = set(eligible["prn"])
    tec_columns = [
        "datetime",
        "prn",
        "minbin",
        "cph1",
        "cph2",
        "rng1",
        "rng2",
        "freq_1",
        "freq_2",
    ]
    tec_frame = frame.loc[frame["prn"].isin(eligible_prns), tec_columns]
    tec_frame = tec_frame.reset_index(drop=True)
    tec_frame = add_tec_columns(
        tec_frame,
        pair="12",
        fs=FS_HZ,
        copy=False,
    )

    signal_groups = frame.groupby(["prn", "minbin"], sort=False, observed=True).indices
    signal_time = frame["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    snr1_all = frame["snr1"].to_numpy(dtype=float)
    snr2_all = frame["snr2"].to_numpy(dtype=float)
    tec_groups = tec_frame.groupby(["prn", "minbin"], sort=False, observed=True).indices
    tec_time = tec_frame["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    tec12_all = tec_frame["tec_cph12"].to_numpy(dtype=float)
    del frame, tec_frame

    records: list[dict[str, object]] = []
    if verbose:
        print("Computing FFT features for eligible satellite-minutes")
    for row_number, (_, row) in enumerate(eligible.iterrows(), start=1):
        key = (row["prn"], row["minute_timestamp_utc"])
        signal_positions = signal_groups[key]
        tec_positions = tec_groups[key]
        snr1, n_snr1 = regular_minute_values(
            signal_time[signal_positions],
            snr1_all[signal_positions],
            minute=row["minute_timestamp_utc"],
            n_threshold=n_threshold,
        )
        snr2, n_snr2 = regular_minute_values(
            signal_time[signal_positions],
            snr2_all[signal_positions],
            minute=row["minute_timestamp_utc"],
            n_threshold=n_threshold,
        )
        if snr1 is None or snr2 is None:
            raise AssertionError("eligible SNR bin failed regular-grid validation")
        tec12, n_tec12 = regular_minute_values(
            tec_time[tec_positions],
            tec12_all[tec_positions],
            minute=row["minute_timestamp_utc"],
            n_threshold=n_threshold,
        )

        record = row.to_dict()
        record["n_samples"] = n_snr1
        record["n_snr2_samples"] = n_snr2
        record["n_tec12_samples"] = n_tec12
        record["source_filename"] = input_path.name
        _add_spectral_values(
            record,
            prefix="snr1",
            values=snr1,
            preprocessing="median",
        )
        _add_spectral_values(
            record,
            prefix="snr2",
            values=snr2,
            preprocessing="median",
        )
        _add_spectral_values(
            record,
            prefix="tec12",
            values=tec12,
            preprocessing="linear",
        )
        records.append(record)
        if verbose and row_number % 500 == 0:
            print(f"  processed {row_number:,}/{len(eligible):,}")

    output = pd.DataFrame.from_records(records)
    for count_column in (
        "n_samples",
        "n_snr2_samples",
        "n_tec12_samples",
        "n_common_delta_snr1",
        "n_common_delta_snr2",
    ):
        output[count_column] = output[count_column].astype("Int64")
    output = (
        output.loc[:, OUTPUT_COLUMNS]
        .sort_values(["minute_timestamp_utc", "prn"], kind="stable")
        .reset_index(drop=True)
    )
    return output, clock_report.to_dict()


def write_features(
    features: pd.DataFrame,
    output_path: str | Path,
    *,
    input_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write the output Parquet without risking the source file."""

    output_path = Path(output_path).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()
    if output_path == input_path:
        raise ValueError("output path must not be the source data file")
    if output_path.suffix.lower() not in {".pq", ".parquet"}:
        raise ValueError("output path must end in .pq or .parquet")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        features.to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one raw ScintPi Parquet file")
    parser.add_argument(
        "--output",
        type=Path,
        help="output .pq/.parquet (default: INPUT_STEM_ml_features.pq)",
    )
    parser.add_argument("--n-thresh", type=int, default=DEFAULT_N_THRESHOLD)
    parser.add_argument("--s4-thresh", type=float, default=DEFAULT_S4_THRESHOLD)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file (the input can never be replaced)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_ml_features.pq")
    features, clock_report = compute_features(
        input_path,
        n_threshold=args.n_thresh,
        s4_threshold=args.s4_thresh,
    )
    written = write_features(
        features,
        output_path,
        input_path=input_path,
        overwrite=args.overwrite,
    )
    print(
        "Clock: "
        f"{clock_report['receiver_epochs_after_deduplication']:,} epochs, "
        f"{clock_report['missing_receiver_epochs']:,} missing epochs, "
        f"{clock_report['gps_week_rollovers_unwrapped']} week rollover(s)"
    )
    print(
        f"Wrote {len(features):,} rows and {len(features.columns)} columns to {written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
