"""ScintPi 4 in the core pipeline: dispatch, names, kept signals, channel 3, clock fallback, recent files."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scintkit import reading, schema
from scintkit.filenames import is_lvl0_parquet, output_path, start_time
from scintkit.formatting import channels, format_lvl0
from scintkit.machine_learning.classify import unclassified
from scintkit.phase import estimate_clock
from scintkit.pipeline import _out_dir, find_recent_files, process_file
from scintkit.products import assemble, compute_products

FIXTURES = Path(__file__).parent / "fixtures"
SC3_LVL0 = FIXTURES / "scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
BLOCKS = ("SatVisibility1", "GPSUtc", "GLONav", "ReceiverSetup1")


@pytest.fixture
def fake_bin2asc(tmp_path, monkeypatch):
    """A stand-in for RXTools bin2asc that writes the saved tx01312a00 text excerpts."""
    script = tmp_path / "fake_bin2asc"
    lines = ["#!/bin/sh", 'name="$2"', f'cp "{FIXTURES}/sc4_tx01312a00_measurements.txt" "${{name}}_measurements.txt"']
    lines += [f'cp "{FIXTURES}/sc4_tx01312a00_{b}.txt" "${{name}}_SBF_{b}.txt"' for b in BLOCKS]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    monkeypatch.setenv("SCINTKIT_BIN2ASC", str(script))
    raw = tmp_path / "sc000" / "20261108" / "tx01312a00.26_"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")
    return raw


def test_to_lvl0_dispatches_sbf_files(fake_bin2asc, tmp_path):
    out = reading.to_lvl0(fake_bin2asc, tmp_path / "out")
    assert Path(out).name == "tx01_20261108_0000_lvl0.pq"
    df, meta = reading.read_lvl0(out)
    assert len(df) == 140 and meta["time_scale"] == "UTC" and meta["station_code"] == "US-TX3"
    assert channels(df) == (1, 2, 3)


def test_sbf_file_runs_through_process_file(fake_bin2asc, tmp_path):
    written = process_file(fake_bin2asc, tmp_path / "out", mode="lvl3", fs=50)
    lvl3 = pd.read_parquet(written[-1])
    assert [Path(p).name for p in written] == ["tx01_20261108_0000_lvl0.pq", "tx01_20261108_0000_lvl3.pq"]
    assert list(lvl3.columns) == [c[0] for c in schema.LVL3_COLUMNS]  # every channel-3 column too
    assert lvl3["ml_class"].eq("Unclassified").all()  # 50 Hz: the classifier is skipped
    assert lvl3.loc[lvl3["prn"] == "E09", "sig_2"].iloc[0] == "GAL_E5a"
    meta = schema.read_metadata(written[-1])
    assert meta["elevation_available"] == "true" and meta["ml_classifier_ran"] == "false"


def test_names_of_sbf_and_older_lvl0_files():
    assert output_path("/a/cr01312m15.26_", "/o", "lvl0") == "/o/cr01_20261108_1215_lvl0.pq"
    assert output_path("/a/cr01_20261108_1215_lvl0.pq", "/o", "lvl3") == "/o/cr01_20261108_1215_lvl3.pq"
    old = "/t/sc003/scintpi3_20241018_2000_359072.8438W_72127.0625S_v326d.pq"
    assert output_path(old, "/o", "lvl2") == "/o/scintpi3_20241018_2000_359072.8438W_72127.0625S_v326d_lvl2.pq"
    assert is_lvl0_parquet(old) and is_lvl0_parquet("x_lvl0.parquet")
    assert not is_lvl0_parquet("x_lvl3.pq") and not is_lvl0_parquet("x_lvl2.parquet")
    assert not is_lvl0_parquet("x_ml_features.pq") and not is_lvl0_parquet("x.bin.zip")
    assert start_time("cg01313a15.26_") == pd.Timestamp("2026-11-09 00:15").to_pydatetime()


def test_lvl0_without_suffix_is_processed_and_products_are_skipped(tmp_path):
    old = tmp_path / SC3_LVL0.name.replace("_lvl0.pq", ".pq")
    shutil.copy(SC3_LVL0, old)
    written = process_file(old, tmp_path / "out", mode="lvl3", classify=False)
    assert [Path(p).name for p in written] == [old.name.replace(".pq", "_lvl3.pq")]
    assert process_file(written[0], tmp_path / "again", mode="lvl3") == []


def test_format_lvl0_keeps_signals_from_the_file():
    sc4 = pd.DataFrame({
        "cons": np.array([2, 3, 1], dtype=np.uint8), "svid": np.array([9, 30, 131], dtype=np.uint8),
        "sig_1": ["GAL_E1BC", "BDS_B1I", "GEO_L1"], "freq_1": [1575.42, 1561.098, 1575.42],
        "sig_2": ["GAL_E5a", "BDS_B3I", "GEO_L5"], "freq_2": [1176.45, 1268.52, 1176.45],
        "sig_3": ["GAL_E5b", "BDS_B2a", None], "freq_3": [1207.14, 1176.45, np.nan],
        **{f"{n}{i}": [2.2e7 if n == "rng" else 45.0] * 3 for n in ("snr", "cph", "rng") for i in (1, 2, 3)},
    })
    sc4.loc[2, ["snr3", "cph3", "rng3"]] = np.nan
    out = format_lvl0(sc4.copy())
    assert out["sig_2"].tolist() == ["GAL_E5a", "BDS_B3I", "GEO_L5"]  # ScintPi 3 table: E5b, none, none
    assert out["freq_2"].tolist() == [1176.45, 1268.52, 1176.45]
    assert out["cph2"].notna().all() and out["cph3"].notna().tolist() == [True, True, False]
    assert out["prn"].tolist() == ["E09", "C30", "S31"]

    sc3 = sc4.drop(columns=[c for c in sc4 if c.startswith(("sig_", "freq_")) or c.endswith("3")])
    out = format_lvl0(sc3)
    assert out["sig_2"].tolist()[:1] == ["GAL_E5b"] and out["cph2"].isna().tolist() == [False, True, True]


def synthetic_frame(fs=10, minutes=3, third=True, elev=(45.0, 60.0)):
    """Two GPS satellites with smooth phase, a common clock wobble and a small scintillation."""
    rng = np.random.default_rng(0)
    t = np.arange(int(minutes * 60 * fs)) / fs
    clock = 0.3 * np.sin(2 * np.pi * t / 7)  # rad/MHz-scaled receiver clock, removed by the clock step
    freqs = {1: 1575.42, 2: 1227.60, 3: 1176.45}
    sigs = {1: "GPS_L1CA", 2: "GPS_L2C", 3: "GPS_L5"}
    rows = []
    for k, svid in enumerate((5, 7)):
        sat = {"datetime": pd.Timestamp("2026-11-08 12:00") + pd.to_timedelta(t, unit="s"),
               "cons": np.uint8(0), "svid": np.uint8(svid), "elev": np.float32(elev[k]), "azim": np.float32(100)}
        geometry = 2.2e7 + 300.0 * t + 1e3 * k
        tec = 20 + 0.01 * t
        for i in (1, 2, 3) if third else (1, 2):
            f = freqs[i] * 1e6
            iono = 40.3e16 * tec / f**2
            noise = 0.05 * rng.standard_normal(len(t))
            sat[f"cph{i}"] = (geometry - iono) * f / 299792458 + (clock * freqs[i] + noise) / (2 * np.pi)
            sat[f"rng{i}"] = geometry + iono + rng.standard_normal(len(t))
            sat[f"snr{i}"] = (45 + 0.5 * rng.standard_normal(len(t))).astype(np.float32)
            sat[f"sig_{i}"] = sigs[i]
            sat[f"freq_{i}"] = freqs[i]
        rows.append(pd.DataFrame(sat))
    return pd.concat(rows).sort_values(["datetime", "svid"], kind="stable").reset_index(drop=True)


def test_three_channel_products():
    frame, minute, fs = compute_products(synthetic_frame(), fs=10)
    assert fs == 10
    for column in ("detrended_noclk_cph3", "tec_cph13", "tec_rng13"):
        assert frame[column].notna().any(), column
    for column in ("s4_3", "sigma_phi_3", "tau_3", "n_s4_3", "n_cycleslip_3", "s4_quality_flag_3",
                   "sigma_phi_quality_flag_3"):
        assert column in minute, column
    middle = minute[minute["datetime"] == pd.Timestamp("2026-11-08 12:01")]
    assert middle["sigma_phi_3"].between(0, 0.2).all() and middle["s4_3"].between(0, 0.3).all()
    # TEC is levelled to the pseudorange TEC (20 TECU at the start) for both pairs.
    assert frame["tec_cph13"].median() == pytest.approx(frame["tec_cph12"].median(), abs=1)
    lvl3 = assemble(frame, unclassified(minute), "1min")
    assert list(lvl3.columns) == [c[0] for c in schema.LVL3_COLUMNS]


def test_two_channel_frame_keeps_the_scintpi3_columns():
    frame, minute, _ = compute_products(synthetic_frame(third=False), fs=10)
    assert "tec_cph13" not in frame and "s4_3" not in minute
    assert list(assemble(frame, unclassified(minute), "1min").columns) == schema.LVL3_COLUMN_NAMES
    assert list(assemble(frame, minute, "1s").columns) == schema.COLUMN_NAMES


def clock_frame(elev_low):
    return pd.DataFrame({
        "epoch": [0, 0, 1, 1], "elev": [50.0, elev_low, 50.0, elev_low],
        "detrended_cph1": [1.0, 3.0, 2.0, 4.0], "detrended_cph2": [1.0, 3.0, 2.0, 4.0],
        "freq_1": [1.0] * 4, "freq_2": [1.0] * 4,
    })


def test_clock_uses_the_elevation_mask_when_elevation_exists():
    assert estimate_clock(clock_frame(5.0))["clock_term"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_clock_falls_back_to_all_satellites_without_elevation(caplog):
    with caplog.at_level(logging.WARNING, logger="scintkit.phase"):
        clock = estimate_clock(clock_frame(np.nan).assign(elev=np.nan))["clock_term"]
    assert clock.tolist() == [2.0, 2.0, 3.0, 3.0]
    assert sum("no elevation" in r.message for r in caplog.records) == 1


def test_sigma_phi_is_finite_without_elevation():
    _, minute, _ = compute_products(synthetic_frame(elev=(np.nan, np.nan)), fs=10)
    middle = minute[minute["datetime"] == pd.Timestamp("2026-11-08 12:01")]
    assert middle[["sigma_phi_1", "sigma_phi_2", "sigma_phi_3"]].notna().all().all()


def test_recent_files_include_dated_sc4_folders(tmp_path):
    files = [
        "sc000/scintpi3_20261108_2000_96.7573W_32.9919N_v326f.bin.zip",
        "sc003/20261108/cg01312x45.26_",  # 2026-11-08 23:45
        "sc003/20261109/cg01313a00.26_",  # 2026-11-09 00:00
        "sc003/20261109/cg01313a00.txt",  # not raw
        "sc003/20261109/cg01313h00.26_",  # 07:00, after now
        "sc003/20261101/cg01305a00.26_",  # too old
    ]
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    found = find_recent_files(tmp_path, days=1, now=pd.Timestamp("2026-11-09 06:00").to_pydatetime())
    assert [str(Path(p).relative_to(tmp_path)) for p in found] == files[:3]
    assert _out_dir(found[2], "/out", mirror=True) == "/out/sc003/20261109"
    assert _out_dir(found[0], "/out", mirror=True) == "/out/sc000"
