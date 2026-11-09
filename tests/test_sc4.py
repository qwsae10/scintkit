"""SC4 (Septentrio SBF) reader tests that need no RXTools: names, signal map, text parsing."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scintkit import sc4

FIXTURES = Path(__file__).parent / "fixtures"
TXT = {name: FIXTURES / f"sc4_tx01312a00_{name}.txt"
       for name in ("measurements", "SatVisibility1", "GPSUtc", "GLONav", "ReceiverSetup1")}


@pytest.mark.parametrize("name", [
    "cr01001a00.26_", "tx01312m15.26_", "cg02312x45.25_", "/mfs/x/sc026/20261108/mx01312a00.26_",
])
def test_sc4_names_are_detected(name):
    assert sc4.is_sc4_file(name)


@pytest.mark.parametrize("name", [
    "zz01001a00.26_",  # unregistered prefix
    "cr01001a00.26",  # no trailing underscore
    "cr01001y00.26_",  # hour letter beyond x
    "cr01001a00.txt",
    "scintpi3_20241011_1200_96.7573W_32.9919N_v326f.bin.zip",
    "tx01_20261108_0000_lvl0.pq",
])
def test_other_names_are_rejected(name):
    assert not sc4.is_sc4_file(name)


def test_name_gives_station_and_start():
    assert sc4.parse_sc4_name("cr01312m15.26_") == ("cr01", datetime(2026, 11, 8, 12, 15))


def test_signal_map():
    by_cons = {}
    for sig, ch in sc4.CHANNEL.items():
        by_cons.setdefault(sig.split("_")[0], {})[ch] = sig
    assert by_cons["GPS"] == {1: "GPS_L1CA", 2: "GPS_L2C", 3: "GPS_L5"}
    assert by_cons["GAL"] == {1: "GAL_E1BC", 2: "GAL_E5a", 3: "GAL_E5b"}
    assert by_cons["BDS"] == {1: "BDS_B1I", 2: "BDS_B3I", 3: "BDS_B2a"}
    assert by_cons["GLO"] == {1: "GLO_L1CA", 2: "GLO_L2CA", 3: "GLO_L3"}
    assert by_cons["GEO"] == {1: "GEO_L1", 2: "GEO_L5"}
    for sig in ("GPS_L2PY", "GPS_L1PY", "GAL_E6BC", "GAL_E5", "BDS_B1C", "BDS_B2b", "GLO_L2P"):
        assert sig not in sc4.CHANNEL
    for sig in sc4.CHANNEL:
        assert sig in sc4.SIGNAL_MHZ or sig in sc4.GLONASS_FDMA_MHZ


def test_glonass_fdma_frequency():
    sig = pd.Series(["GLO_L1CA", "GLO_L2CA", "GLO_L3", "GLO_L1CA", "GPS_L1CA"], dtype="string")
    svid = pd.Series(["R09", "R09", "R09", "R99", "G01"])
    freq = sc4.frequency_mhz(sig, svid, {"R09": -2})
    np.testing.assert_allclose(freq[:3], [1600.875, 1245.125, 1202.025])
    assert np.isnan(freq[3])
    assert freq[4] == 1575.42


def test_find_bin2asc(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(sc4.BIN2ASC_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="RXTools"):
        sc4.find_bin2asc()
    wrapper = tmp_path / "rxtools_exec.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    assert sc4.find_bin2asc() == ["rxtools_exec.sh", "bin2asc"]
    monkeypatch.setenv(sc4.BIN2ASC_ENV, "rxtools_exec.sh bin2asc -v")
    assert sc4.find_bin2asc() == ["rxtools_exec.sh", "bin2asc", "-v"]
    with pytest.raises(FileNotFoundError):
        sc4.find_bin2asc("/nonexistent/bin2asc")


def test_block_helpers():
    assert sc4.leap_seconds(TXT["GPSUtc"]) == 18
    assert sc4.leap_seconds(None) == sc4.DEFAULT_LEAP_SECONDS
    assert sc4.receiver_info(TXT["ReceiverSetup1"]) == "mosaic-X5 4.14.0"
    k = sc4.glonass_k(TXT["GLONav"], TXT["SatVisibility1"])
    assert k["R09"] == -2 and k["R15"] == 0 and k["R19"] == 3


@pytest.fixture(scope="module")
def lvl0():
    meas = sc4.read_measurements(TXT["measurements"])
    vis = sc4.read_visibility(TXT["SatVisibility1"])
    k = sc4.glonass_k(TXT["GLONav"], TXT["SatVisibility1"])
    return sc4.to_lvl0_frame(meas, vis, k, sc4.leap_seconds(TXT["GPSUtc"]))


def test_lvl0_layout(lvl0):
    assert list(lvl0.columns) == [
        "datetime", "cons", "svid", "elev", "azim",
        "snr1", "snr2", "snr3", "cph1", "cph2", "cph3", "rng1", "rng2", "rng3",
        "sig_1", "sig_2", "sig_3", "freq_1", "freq_2", "freq_3",
    ]
    assert lvl0["cons"].dtype == np.uint8 and lvl0["svid"].dtype == np.uint8
    assert len(lvl0) == 5 * 28  # five 50 Hz epochs, 28 satellites with a mapped signal
    assert not lvl0.duplicated(["datetime", "cons", "svid"]).any()


def test_lvl0_time_is_utc_at_50hz(lvl0):
    # TOW 0 of GPS week 2444 is 2026-11-08 00:00:00 GPS = 23:59:42 UTC the day before.
    assert lvl0["datetime"].iloc[0] == pd.Timestamp("2026-11-07 23:59:42")
    assert lvl0["datetime"].nunique() == 5
    assert sc4.sampling_rate(lvl0) == 50


def test_lvl0_satellites_and_channels(lvl0):
    first = lvl0[lvl0["datetime"] == lvl0["datetime"].iloc[0]].set_index(["cons", "svid"])
    assert set(first.index.get_level_values("cons")) == {0, 1, 2, 3, 6}
    assert (1, 131) in first.index  # SBF S31 -> WAAS PRN 131, ScintPi 3 numbering
    g01 = first.loc[(0, 1)]
    assert (g01["sig_1"], g01["sig_2"], g01["sig_3"]) == ("GPS_L1CA", "GPS_L2C", "GPS_L5")
    assert g01["cph1"] == pytest.approx(117273748.078840)
    assert g01["rng2"] == pytest.approx(22316437.133)
    assert g01["snr3"] == pytest.approx(49.5625)
    assert first.loc[(2, 9), "sig_2"] == "GAL_E5a"
    r09 = first.loc[(6, 9)]
    assert r09["freq_1"] == pytest.approx(1600.875) and r09["freq_2"] == pytest.approx(1245.125)
    assert pd.isna(r09["sig_3"]) and np.isnan(r09["cph3"])
    # Every row has channel 1; a missing channel is NaN in all of its columns.
    assert lvl0["cph1"].notna().all()
    for ch in (2, 3):
        absent = lvl0[f"cph{ch}"].isna()
        assert lvl0.loc[absent, [f"snr{ch}", f"rng{ch}", f"freq_{ch}"]].isna().all().all()
        assert lvl0.loc[absent, f"sig_{ch}"].isna().all()


def test_lvl0_geometry_from_satvisibility(lvl0):
    first = lvl0[lvl0["datetime"] == lvl0["datetime"].iloc[0]].set_index(["cons", "svid"])
    assert first.loc[(0, 17), "elev"] == pytest.approx(72.74)
    assert first.loc[(0, 17), "azim"] == pytest.approx(274.46)
    sbas = lvl0["cons"] == 1  # SatVisibility1 has no SBAS entries
    assert lvl0.loc[sbas, "elev"].isna().all()
    assert lvl0.loc[~sbas, "elev"].notna().all()


def test_zero_measurements_are_missing():
    meas = pd.DataFrame({
        "gps_ms": [0, 0, 20, 20], "svid": ["G01"] * 4,
        "sig": ["GPS_L1CA", "GPS_L2C", "GPS_L1CA", "GPS_L2C"],
        "pr": [2.2e7, 2.2e7, 2.2e7, 0.0], "cph": [1.0e8, 0.8e8, 1.0e8, 0.8e8], "cn0": [45.0, 0.0, 45.0, 40.0],
    })
    df = sc4.to_lvl0_frame(meas)
    assert df["cph1"].notna().all()
    assert df["cph2"].isna().all() and df["sig_2"].isna().all()
    assert df["elev"].isna().all()
