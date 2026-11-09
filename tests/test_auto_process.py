from pathlib import Path

import pandas as pd
import pytest

from scintkit import schema
from scintkit.pipeline import find_recent_files, process

FIXTURES = Path(__file__).parent / "fixtures"
OLD_LVL0 = FIXTURES / "scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
RAW_V326F = FIXTURES / "scintpi3_20261101_0400_96.75784W_32.99184N_v326f.bin.zip"


@pytest.fixture(scope="module")
def old_fixture_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    summary = process(OLD_LVL0, out, mode="both")
    return out, summary


def test_process_writes_into_output_dir_with_derived_names(old_fixture_outputs):
    out, summary = old_fixture_outputs
    stem = "scintpi3_20241011_1200_96.7573W_32.9919N_v326f"
    assert sorted(summary["written"]) == [str(out / f"{stem}_lvl2.pq"), str(out / f"{stem}_lvl3.pq")]
    assert summary["failed"] == [] and summary["unsupported"] == []


def test_outputs_follow_the_schema(old_fixture_outputs):
    out, summary = old_fixture_outputs
    for path in summary["written"]:
        frame = pd.read_parquet(path)
        names = schema.LVL3_COLUMN_NAMES if path.endswith("_lvl3.pq") else schema.COLUMN_NAMES
        assert list(frame.columns) == names
        assert not frame.duplicated(["datetime", "prn"]).any()
        assert frame["datetime"].dt.microsecond.eq(0).all()
        assert not any(c.endswith("3") for c in frame.columns if c[-1].isdigit())

    lvl3 = pd.read_parquet(summary["written"][1])
    # 10 Hz detected: a full minute away from arc edges is not flagged.
    assert lvl3["s4_quality_flag_1"].eq(0).mean() > 0.5
    assert lvl3["sigma_phi_quality_flag_1"].eq(0).any()
    assert lvl3["elev"].min() >= -90 and lvl3["elev"].max() <= 90
    # UTC: the GPS-time file started at 12:00:18, i.e. 12:00:00 UTC.
    assert lvl3["datetime"].min() == pd.Timestamp("2024-10-11 12:00")


def test_lvl3_first_sample_values_come_from_one_full_rate_row(old_fixture_outputs):
    out, summary = old_fixture_outputs
    lvl3 = pd.read_parquet(summary["written"][1])
    raw = pd.read_parquet(OLD_LVL0)
    row = lvl3[(lvl3["prn"] == "G05") & (lvl3["datetime"] == pd.Timestamp("2024-10-11 12:05"))].iloc[0]
    match = raw[(raw["cons"] == 0) & (raw["svid"] == 5) & (raw["cph1"] == row["cph1"])]
    assert len(match) == 1
    assert match["rng2"].iloc[0] == row["rng2"]
    assert match["snr2"].iloc[0] == row["snr2"]


def test_metadata_records_rate_time_scale_and_station(old_fixture_outputs):
    out, summary = old_fixture_outputs
    meta = schema.read_metadata(summary["written"][0])
    assert meta["sampling_rate_hz"] == "10"
    assert meta["time_scale"] == "UTC"
    assert meta["firmware"] == "v326f"
    assert meta["station_code"] == "US-TX1"
    assert meta["source_file"] == OLD_LVL0.name


def test_raw_input_writes_lvl0_and_skips_existing_outputs(tmp_path):
    first = process(RAW_V326F, tmp_path, mode="lvl3")
    names = sorted(Path(p).name for p in first["written"])
    assert names == [RAW_V326F.name.replace(".bin.zip", "_lvl0.pq"),
                     RAW_V326F.name.replace(".bin.zip", "_lvl3.pq")]
    assert schema.read_metadata(tmp_path / names[0])["time_scale"] == "UTC"

    again = process(RAW_V326F, tmp_path, mode="lvl3")
    assert again["written"] == [] and again["skipped"] == [str(RAW_V326F)]


def test_unsupported_and_failed_files_are_reported(tmp_path):
    broken = tmp_path / "scintpi3_20240101_0000_96.7573W_32.9919N_v326f.bin"
    broken.write_bytes(b"")
    summary = process([tmp_path / "scintpi2_20240101_0000_v226.bin.zip", broken], tmp_path / "out")
    assert [Path(p).name for p, _ in summary["unsupported"]] == ["scintpi2_20240101_0000_v226.bin.zip"]
    assert "v226" in summary["unsupported"][0][1]
    assert [Path(p) for p, _ in summary["failed"]] == [broken]


def test_find_recent_files_uses_filename_start_time(tmp_path):
    names = {
        "sc000": ["scintpi3_20261108_2000_96.7573W_32.9919N_v326f.bin.zip",
                  "scintpi3_20261107_0000_96.7573W_32.9919N_v326f.bin.zip",
                  "scintpi3_20261108_2000_96.7573W_32.9919N_v326f_lvl0.pq"],
        "sc010": ["scintpi3_20261109_0400_1114207.5000W_419337.7188N_v325.bin.zip"],
    }
    for folder, files in names.items():
        (tmp_path / folder).mkdir()
        for name in files:
            (tmp_path / folder / name).touch()

    found = find_recent_files(tmp_path, days=1, now=pd.Timestamp("2026-11-09 06:00").to_pydatetime())

    assert [Path(p).name for p in found] == [names["sc000"][0], names["sc010"][0]]
