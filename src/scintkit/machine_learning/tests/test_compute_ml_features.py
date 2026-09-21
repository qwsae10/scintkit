from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scintkit.machine_learning.assemble_labeled_features import select_and_label
from scintkit.machine_learning.batch_ml_features import (
    DEFAULT_COORDINATE_TOLERANCE_DEG,
    discover_input_files,
    error_path_for,
    filename_year,
    output_path_for,
    parse_filename_coordinates,
    parse_year_argument,
    select_file_shard,
)
from scintkit.machine_learning.compute_ml_features import (
    DEFAULT_S4_THRESHOLD,
    compute_common_mode_minute_features,
    compute_minute_summary,
    integrate_periodogram_bands_db,
    regular_minute_signal,
)
from scintkit.machine_learning.receiver_clock import reconstruct_receiver_clock
from scintkit.preprocessing.format import add_sigs, make_prn
from scintkit.services.compute import _repair_tec_pair, compute_s4
from scintkit.services.phase_detrend import repair_discontinuities_pos

EXAMPLE_SOURCE_NAME = "scintpi3_20240120_2000_359072.9062W_72126.7422S_v326d.pq"
DECIMAL_DEGREE_SOURCE_NAME = "scintpi3_20260408_2000_96.79263W_46.90714N_v326f.pq"


def test_ml_batch_import_does_not_import_matplotlib() -> None:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[3])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_root
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import scintkit.machine_learning.batch_ml_features; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_packed_filename_coordinates_decode_with_hemisphere_signs() -> None:
    latitude, longitude = parse_filename_coordinates(EXAMPLE_SOURCE_NAME)
    assert np.isclose(latitude, -7.21267422, rtol=0, atol=1e-14)
    assert np.isclose(longitude, -35.90729062, rtol=0, atol=1e-14)
    assert filename_year(EXAMPLE_SOURCE_NAME) == 2024
    assert abs(latitude - (-7.213)) <= DEFAULT_COORDINATE_TOLERANCE_DEG
    assert abs(longitude - (-35.907)) <= DEFAULT_COORDINATE_TOLERANCE_DEG

    for name, expected in (
        ("x_1203580.0000W_432710.0000N_y.pq", (43.271, -120.358)),
        ("x_967930.0000W_469070.0000N_y.pq", (46.907, -96.793)),
    ):
        assert np.allclose(
            parse_filename_coordinates(name),
            expected,
            rtol=0,
            atol=1e-14,
        )


def test_decimal_degree_filename_coordinates_are_not_divided_by_scale() -> None:
    assert np.allclose(
        parse_filename_coordinates(DECIMAL_DEGREE_SOURCE_NAME),
        (46.90714, -96.79263),
        rtol=0,
        atol=1e-14,
    )


def test_decimal_degree_multipath_file_is_labeled() -> None:
    frame = pd.DataFrame(
        {
            "svid": [1],
            "s4_1": [0.30],
            "elevation_deg": [30.0],
            "minute_timestamp_utc": [pd.Timestamp("2026-04-08 20:00:00")],
        }
    )
    feature_name = DECIMAL_DEGREE_SOURCE_NAME.replace(
        ".pq",
        "_ml_features.pq",
    )
    labeled = select_and_label(frame, Path(feature_name))
    assert labeled["label"].tolist() == ["Multipath"]


def test_year_argument_accepts_one_or_multiple_years() -> None:
    assert parse_year_argument("2024") == (2024,)
    assert parse_year_argument("2024,2025,2026") == (2024, 2025, 2026)
    assert parse_year_argument("2024, 2025,2024") == (2024, 2025)


def test_file_discovery_filters_year_coordinates_and_generated_outputs(
    tmp_path,
) -> None:
    station = tmp_path / "receiver_a"
    station.mkdir()
    matched = station / EXAMPLE_SOURCE_NAME
    matched.touch()
    (station / EXAMPLE_SOURCE_NAME.replace("20240120", "20230120")).touch()
    (station / EXAMPLE_SOURCE_NAME.replace("72126.7422S", "450000.0000N")).touch()

    output_dir = tmp_path / "ml_features"
    output_dir.mkdir()
    generated = output_dir / f"{matched.stem}_ml_features.pq"
    generated.touch()

    sources = discover_input_files(
        tmp_path,
        output_dir=output_dir,
        year=2024,
        coordinates=[(-7.213, -35.907)],
    )
    assert sources == [matched]
    assert output_path_for(matched, output_dir) == generated
    assert error_path_for(generated).name.endswith("_ml_features_err.txt")


def test_file_discovery_accepts_multiple_years_and_coordinate_formats(
    tmp_path,
) -> None:
    station = tmp_path / "receiver_a"
    station.mkdir()
    packed_2024 = station / EXAMPLE_SOURCE_NAME
    packed_2025 = station / EXAMPLE_SOURCE_NAME.replace("20240120", "20250120")
    decimal_2026 = station / DECIMAL_DEGREE_SOURCE_NAME
    for source in (packed_2024, packed_2025, decimal_2026):
        source.touch()

    output_dir = tmp_path / "ml_features"
    output_dir.mkdir()
    sources = discover_input_files(
        tmp_path,
        output_dir=output_dir,
        year=(2024, 2025, 2026),
        coordinates=[(-7.213, -35.907), (46.907, -96.793)],
    )
    assert sources == sorted((packed_2024, packed_2025, decimal_2026))


def test_file_shards_are_disjoint_and_cover_every_whole_file() -> None:
    sources = [Path(f"source_{number:02d}.pq") for number in range(11)]
    shards = [
        select_file_shard(sources, shard_index=index, shard_count=4)
        for index in range(4)
    ]
    assert shards[0] == sources[0::4]
    assert shards[1] == sources[1::4]
    assert shards[2] == sources[2::4]
    assert shards[3] == sources[3::4]
    assert sorted(path for shard in shards for path in shard) == sources
    assert sum(len(shard) for shard in shards) == len(
        {path for shard in shards for path in shard}
    )


def test_default_s4_threshold_is_point_fifteen() -> None:
    assert DEFAULT_S4_THRESHOLD == 0.15


def test_periodogram_places_tone_in_expected_band_and_keeps_first_bin() -> None:
    fs = 20.0
    time = np.arange(1200) / fs
    tone = np.sin(2 * np.pi * 0.2 * time)
    powers = integrate_periodogram_bands_db(tone, preprocessing="median")

    assert np.isfinite(powers["0p1_0p3_hz"])
    assert powers["0p1_0p3_hz"] > powers["0p05_0p1_hz"] + 20
    first_bin = np.sin(2 * np.pi * (1 / 60) * time)
    first_bin_powers = integrate_periodogram_bands_db(first_bin, preprocessing="median")
    assert np.isfinite(first_bin_powers["0p017_0p05_hz"])


def test_tec_linear_detrending_removes_ramp_without_removing_tone() -> None:
    fs = 20.0
    time = np.arange(1200) / fs
    tec = 20 + 0.4 * time + np.sin(2 * np.pi * 0.2 * time)
    powers = integrate_periodogram_bands_db(tec, preprocessing="linear")
    assert powers["0p1_0p3_hz"] > powers["0p05_0p1_hz"] + 20


def test_regular_minute_signal_fills_only_a_small_missing_fraction() -> None:
    time = pd.date_range("2024-01-01", periods=1200, freq="50ms")
    frame = pd.DataFrame({"datetime": time, "value": np.arange(1200, dtype=float)})
    frame = frame.drop(index=[0, 100, 101, 500, 700]).reset_index(drop=True)
    frame.loc[[20, 30, 40, 50], "value"] = np.nan

    values, count = regular_minute_signal(
        frame,
        "value",
        minute=pd.Timestamp("2024-01-01"),
        n_threshold=1190,
    )
    assert count == 1191
    assert values is not None
    assert np.isfinite(values).all()
    assert np.allclose(values, np.arange(1200, dtype=float))

    rejected, rejected_count = regular_minute_signal(
        frame.iloc[:-1],
        "value",
        minute=pd.Timestamp("2024-01-01"),
        n_threshold=1191,
    )
    assert rejected is None
    assert rejected_count == 1190


def test_minute_summary_uses_valid_snr_counts_and_scintkit_s4() -> None:
    samples = 1200
    frame = pd.DataFrame(
        {
            "prn": ["G01"] * samples,
            "minbin": [pd.Timestamp("2024-01-01")] * samples,
            "snr1": np.tile([30.0, 50.0], samples // 2),
            "snr2": np.full(samples, 42.0),
            "svid": np.ones(samples, dtype=int),
            "cons": ["GPS"] * samples,
            "elev": np.full(samples, 45),
            "azim": np.full(samples, 180),
            "sig_1": ["GPS_L1CA"] * samples,
            "sig_2": ["GPS_L2C"] * samples,
            "freq_1": np.full(samples, 1575.42),
            "freq_2": np.full(samples, 1227.60),
        }
    )
    frame.loc[0, "snr2"] = np.nan
    summary = compute_minute_summary(frame).iloc[0]
    assert summary["n_samples"] == 1200
    assert summary["n_snr2_samples"] == 1199
    assert summary["s4_1"] > 0.1
    assert np.isclose(summary["s4_2"], 0.0, atol=1e-12)


def test_vectorized_minute_s4_matches_scalar_reference_and_gate() -> None:
    generator = np.random.default_rng(8)
    rows = []
    for minute_number in range(4):
        for svid in (1, 2):
            snr1 = 40 + generator.normal(0, 2 + minute_number, 1200)
            snr2 = 42 + generator.normal(0, 1 + svid, 1200)
            for sample in range(1200):
                rows.append(
                    {
                        "prn": f"G{svid:02d}",
                        "minbin": pd.Timestamp("2024-01-01")
                        + pd.Timedelta(minutes=minute_number),
                        "snr1": snr1[sample],
                        "snr2": snr2[sample],
                        "svid": svid,
                        "cons": "GPS",
                        "elev": 45,
                        "azim": 180,
                        "sig_1": "GPS_L1CA",
                        "sig_2": "GPS_L2C",
                        "freq_1": 1575.42,
                        "freq_2": 1227.60,
                    }
                )
    frame = pd.DataFrame(rows)
    summary = compute_minute_summary(frame).set_index(["prn", "minute_timestamp_utc"])
    reference = frame.groupby(["prn", "minbin"], sort=False).agg(
        s4_1=("snr1", compute_s4),
        s4_2=("snr2", compute_s4),
    )
    assert np.allclose(summary[["s4_1", "s4_2"]], reference, rtol=0, atol=3e-15)
    assert summary["s4_1"].gt(0.15).equals(reference["s4_1"].gt(0.15))


def test_vectorized_format_mappings_match_previous_semantics() -> None:
    frame = pd.DataFrame(
        {
            "cons": ["GPS", "GLO", "GAL", "BDS", "QZSS", "SBS"],
            "svid": [1, 12, 3, 25, 7, 123],
        }
    )
    old_prn = frame["cons"].map(
        {
            "GPS": "G",
            "BDS": "C",
            "GAL": "E",
            "GLO": "R",
            "QZSS": "J",
            "IRNSS": "I",
            "SBAS": "S",
            "SBS": "S",
        }
    ) + frame["svid"].astype(int).astype(str).str.zfill(2)
    assert make_prn(frame).equals(old_prn)

    formatted = add_sigs(frame.copy())
    assert formatted["sig_1"].iloc[:5].tolist() == [
        "GPS_L1CA",
        "GLO_L1CA",
        "GAL_L1BC",
        "BDS_B1I",
        "QZS_L1CA",
    ]
    assert formatted["sig_2"].iloc[:5].tolist() == [
        "GPS_L2C",
        "GLO_L2C",
        "GAL_E5b",
        "BDS_B2I",
        "QZS_L2C",
    ]
    assert pd.isna(formatted.loc[5, "sig_1"])
    assert pd.isna(formatted.loc[5, "sig_2"])
    assert formatted["freq_1"].iloc[:5].tolist() == [
        1575.42,
        1602.0,
        1575.42,
        1561.098,
        1575.42,
    ]


def test_paired_tec_repair_matches_two_independent_repairs() -> None:
    generator = np.random.default_rng(12)
    values = np.column_stack(
        [
            np.cumsum(generator.normal(0, 0.05, 800)),
            np.cumsum(generator.normal(0, 0.08, 800)),
        ]
    )
    values[120:124, 0] = np.nan
    values[400:403, 1] = np.nan
    values[600:, 0] += 4.0
    paired = _repair_tec_pair(values, fs=20)
    for column_number, column in enumerate(("carrier", "pseudo")):
        reference, _, _ = repair_discontinuities_pos(
            pd.Series(values[:, column_number]), fs=20, threshold=1
        )
        assert np.allclose(paired[column], reference, equal_nan=True, rtol=0, atol=0)


def test_common_mode_uses_all_satellites_before_minute_aggregation() -> None:
    seconds = 10 * 60
    rows = []
    for epoch, timestamp in enumerate(
        pd.date_range("2024-01-01", periods=seconds, freq="s")
    ):
        event = 2.0 if 5 * 60 <= epoch < 5 * 60 + 20 else 0.0
        for svid in range(1, 5):
            rows.append(
                {
                    "datetime": timestamp,
                    "_receiver_epoch": epoch,
                    "prn": f"G{svid:02d}",
                    "elev": 45,
                    "snr1": 40.0 + svid + event,
                    "snr2": 42.0 + svid - event,
                }
            )
    frame = pd.DataFrame(rows)
    result = compute_common_mode_minute_features(frame, fs_hz=1.0)
    event_minute = result.loc[
        result["minute_timestamp_utc"].eq(pd.Timestamp("2024-01-01 00:05"))
    ].iloc[0]
    assert event_minute["common_delta_snr1_p95_dbhz"] > 1.5
    assert event_minute["common_delta_snr2_p95_dbhz"] >= 0.0
    assert event_minute["n_common_delta_snr1"] == 60
    assert event_minute["n_common_delta_snr2"] == 60


def test_empty_common_mode_retains_datetime_merge_key() -> None:
    frame = pd.DataFrame(
        columns=[
            "datetime",
            "_receiver_epoch",
            "prn",
            "elev",
            "snr1",
            "snr2",
        ]
    )
    result = compute_common_mode_minute_features(frame)
    assert result["minute_timestamp_utc"].dtype == "datetime64[ns]"


def test_receiver_clock_splits_overlap_unwraps_week_and_deduplicates() -> None:
    epochs = [
        (pd.Timestamp("2024-01-20 23:59:59.875000"), 10),
        (pd.Timestamp("2024-01-20 23:59:59.875000"), 20),
        (pd.Timestamp("2024-01-14 00:00:00"), 30),
        (pd.Timestamp("2024-01-14 00:00:00"), 30),  # exact copy
    ]
    rows = []
    for timestamp, value in epochs:
        for svid in range(1, 5):
            rows.append(
                {
                    "datetime": timestamp,
                    "cons": 0,
                    "svid": svid,
                    "snr1": value + svid,
                }
            )
    repaired, report = reconstruct_receiver_clock(pd.DataFrame(rows))
    assert len(repaired) == 12
    assert repaired["_receiver_epoch"].nunique() == 3
    assert repaired["datetime"].is_monotonic_increasing
    assert repaired["datetime"].max().date() == pd.Timestamp("2024-01-21").date()
    assert report.exact_duplicate_receiver_epochs_removed == 1
    assert report.gps_week_rollovers_unwrapped == 1
    assert not report.sample_order_grid_fallback_used


def test_receiver_clock_falls_back_to_sample_order_for_ambiguous_epochs() -> None:
    timestamp = pd.Timestamp("2024-01-20 20:00:00")
    rows = []
    for epoch, (epoch_time, value) in enumerate(
        [
            (timestamp, 10.0),
            (timestamp, 20.0),
            (timestamp + pd.Timedelta(milliseconds=50), 30.0),
        ]
    ):
        for svid in range(1, 5):
            rows.append(
                {
                    "datetime": epoch_time,
                    "cons": 0,
                    "svid": svid,
                    "snr1": value + svid + epoch / 10,
                }
            )

    repaired, report = reconstruct_receiver_clock(pd.DataFrame(rows))
    epoch_times = (
        repaired.groupby("_receiver_epoch", sort=True)["datetime"].first().sort_index()
    )
    assert report.sample_order_grid_fallback_used
    assert report.receiver_epochs_after_deduplication == 3
    assert epoch_times.diff().dropna().eq(pd.Timedelta(milliseconds=50)).all()
