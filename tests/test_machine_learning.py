import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scintkit import schema
from scintkit.machine_learning import classify
from scintkit.machine_learning.features import FEATURES, band_powers_db, compute_features, minute_grid
from scintkit.pipeline import process_file
from scintkit.products import compute_products

FS = 20


def synthetic_lvl0(minutes=4, seed=7):
    """Deterministic 20 Hz lvl0 frame: 7 satellites, scintillation on G01/G02 in minutes 1-2,
    a receiver-wide C/N0 dip at 100-108 s, implausible channel-2 pseudorange on G06."""
    rng = np.random.default_rng(seed)
    n = minutes * 60 * FS
    t = np.arange(n) / FS
    dip = np.where((t >= 100) & (t < 108), 3 + 2 * np.sin(np.pi * (t - 100) / 8), 0.0)
    tec = 20 + 0.8 * np.sin(2 * np.pi * t / 400) + 0.05 * np.sin(2 * np.pi * 0.2 * t)
    frames = []
    for k, (cons, svid) in enumerate([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (2, 11)]):
        f1, f2 = (1575.42e6, 1227.60e6) if cons == 0 else (1575.42e6, 1207.14e6)
        iono = 40.3e16 * tec
        out = {"cons": np.uint8(cons), "svid": np.uint8(svid),
               "elev": np.float32(20 + 8 * k) + np.float32(0.002) * t.astype(np.float32),
               "azim": np.float32(40 * k) + np.zeros(n, np.float32)}
        for i, f in ((1, f1), (2, f2)):
            snr = 42 + 2 * k % 5 + np.convolve(rng.normal(0, 0.5, n), np.ones(4) / 2, "same") - dip
            if svid in (1, 2):
                m = (t >= 60) & (t < 180)
                w = np.convolve(rng.normal(size=n), np.ones(10) / np.sqrt(10), "same")
                snr[m] += 10 * np.log10(np.clip(1 + 0.4 * w[m], 0.05, None))
            out[f"snr{i}"] = np.clip(np.round(snr), 1, 255).astype(np.uint8)
            out[f"cph{i}"] = 2.0e7 / 0.19 + 700.0 * (k + 1) * t - iono / f**2 * f / 299792458 \
                + rng.normal(0, 0.002, n)
            out[f"rng{i}"] = 2.1e7 + 133.0 * (k + 1) * t + iono / f**2 + rng.normal(0, 0.5, n)
        if svid == 6:
            out["rng2"] = np.full(n, 1.0e7)  # implausible: C/N0 kept, TEC NaN
        tow = 6 * 86400 + 7200 + t
        out["datetime"] = pd.Timestamp("2024-05-05") + pd.to_timedelta(np.float32(tow).astype(float), unit="s")
        out["_order"] = np.arange(n)
        frame = pd.DataFrame(out)
        keep = rng.random(n) > 0.002
        if svid == 3:
            keep &= ~((t >= 150) & (t < 155))  # minute 2 of G03 has too few samples
        frames.append(frame[keep])
    df = pd.concat(frames, ignore_index=True).sort_values(["_order", "cons", "svid"], kind="stable")
    return df.drop(columns="_order").reset_index(drop=True)


FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "ml_features_golden.csv"  # old feature code (commit 9a34c65) on synthetic_lvl0()
LVL0_10HZ = FIXTURES / "scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
MOTHERSDAY = Path(os.environ.get(
    "SCINTKIT_MOTHERSDAY_TEST",
    "/Users/isaac/Documents/ML_Analysis_2026/mothersday_ml_test_20240510_20240511_corrected.pq"))
MOTHERSDAY_CONFUSION = [[32, 0, 9], [6, 15, 0], [1, 0, 19]]  # rows: true Multipath, RFI, Scintillation

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@pytest.fixture(scope="module")
def products():
    return compute_products(synthetic_lvl0())


def test_features_match_the_definitions_the_model_was_trained_on(products):
    frame, _, fs = products
    assert fs == FS
    new = compute_features(frame)
    golden = pd.read_csv(GOLDEN, parse_dates=["datetime"])
    assert new[["prn", "datetime"]].astype(str).values.tolist() == golden[["prn", "datetime"]].astype(str).values.tolist()
    np.testing.assert_allclose(new[FEATURES].to_numpy(float), golden[FEATURES].to_numpy(float), rtol=1e-10, atol=0)


def test_band_power_puts_a_tone_in_its_band_and_keeps_the_first_bin():
    t = np.arange(1200) / FS
    powers = dict(zip(["b0", "b1", "b2", "b3", "b4"], band_powers_db(np.sin(2 * np.pi * 0.2 * t), "median")))
    assert powers["b2"] > powers["b1"] + 20
    assert np.isfinite(band_powers_db(np.sin(2 * np.pi * t / 60), "median")[0])
    ramp = band_powers_db(20 + 0.4 * t + np.sin(2 * np.pi * 0.2 * t), "linear")
    assert ramp[2] > ramp[1] + 20


def test_minute_grid_fills_small_gaps_and_rejects_sparse_minutes():
    time = np.arange(1200) * 50_000_000
    values = np.arange(1200, dtype=float)
    keep = np.ones(1200, bool)
    keep[[0, 100, 101, 500, 700]] = False
    values[[20, 30, 40, 50]] = np.nan
    filled, count = minute_grid(time[keep], values[keep], 0)
    assert count == 1191
    np.testing.assert_allclose(filled, np.arange(1200.0))
    keep[600] = False
    assert minute_grid(time[keep], values[keep], 0) == (None, 1190)


@pytest.mark.skipif(not HAVE_SKLEARN or not MOTHERSDAY.exists(), reason="needs scikit-learn and the Mother's Day test set")
def test_bundled_model_reproduces_the_mothersday_confusion_matrix():
    from sklearn.metrics import confusion_matrix

    test = pd.read_parquet(MOTHERSDAY)
    labels = classify.classify(test.rename(columns={"minute_timestamp_utc": "datetime"}))
    classes = list(classify.PROBABILITY_COLUMNS)
    assert confusion_matrix(test["label"], labels["ml_class"], labels=classes).tolist() == MOTHERSDAY_CONFUSION
    np.testing.assert_allclose(labels[list(classify.PROBABILITY_COLUMNS.values())].sum(axis=1), 1, rtol=1e-6)


def _write_lvl0(df, path):
    schema.write(df, path, {"time_scale": "UTC", "leap_seconds": "18", "firmware": "v326f",
                            "source_file": "synthetic.bin"})


def test_pipeline_classifies_only_dual_frequency_minutes_above_the_s4_threshold(tmp_path, products):
    lvl0 = tmp_path / "scintpi3_20240511_0200_96.75784W_32.99184N_v326f_lvl0.pq"
    _write_lvl0(synthetic_lvl0(), lvl0)
    lvl3_path = [p for p in process_file(lvl0, tmp_path, mode="both") if p.endswith("_lvl3.pq")][0]
    lvl3 = pd.read_parquet(lvl3_path)
    assert list(lvl3.columns) == schema.LVL3_COLUMN_NAMES
    assert lvl3["ml_class"].dtype == "string" and lvl3["ml_prob_rfi"].dtype == np.float32
    assert lvl3["ml_class"].notna().all()
    lvl2 = pd.read_parquet(lvl3_path.replace("_lvl3", "_lvl2"))
    assert list(lvl2.columns) == schema.COLUMN_NAMES

    features = compute_features(products[0])
    sent = set(map(tuple, classify.classifiable(features)[["prn", "datetime"]].astype(str).values))
    assert ("G06", "2024-05-11 02:01:00") not in sent  # featured, but no channel-2 TEC: single frequency
    classified = lvl3["ml_class"].ne(classify.UNCLASSIFIED)
    meta = schema.read_metadata(lvl3_path)
    if not HAVE_SKLEARN:
        assert not classified.any() and meta["ml_classifier_ran"] == "false"
        return
    assert set(map(tuple, lvl3.loc[classified, ["prn", "datetime"]].astype(str).values)) == sent
    assert 0 < len(sent) < len(features) < len(lvl3)
    assert set(lvl3.loc[classified, "ml_class"]) <= set(classify.PROBABILITY_COLUMNS)
    probabilities = lvl3.loc[classified, list(classify.PROBABILITY_COLUMNS.values())]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)
    assert lvl3.loc[~classified, "ml_prob_scintillation"].isna().all()
    assert meta["ml_classification"] == "on" and meta["ml_classifier_ran"] == "true"
    assert meta["ml_model"] == classify.MODEL_FILE and len(meta["ml_model_sha256"]) == 12


def test_low_s4_and_single_frequency_minutes_are_not_sent_to_the_model():
    table = pd.DataFrame({"s4_1": [0.19, 0.2, 0.5, 0.5], "s4_2": [0.3, 0.3, np.nan, 0.3],
                          classify.TEC_POWER: [-30.0, -30.0, -30.0, -30.0]})
    assert classify.classifiable(table).index.tolist() == [1, 3]
    table.loc[3, classify.TEC_POWER] = np.nan
    assert classify.classifiable(table).index.tolist() == [1]


def test_10hz_data_are_unclassified_without_failing(tmp_path):
    lvl3_path = process_file(LVL0_10HZ, tmp_path, mode="lvl3")[0]
    lvl3 = pd.read_parquet(lvl3_path)
    assert list(lvl3.columns) == schema.LVL3_COLUMN_NAMES
    assert lvl3["ml_class"].eq(classify.UNCLASSIFIED).all() and lvl3["ml_prob_multipath"].isna().all()
    meta = schema.read_metadata(lvl3_path)
    assert "10 Hz" in meta["ml_classification"] and meta["ml_classifier_ran"] == "false"


def test_missing_scikit_learn_leaves_every_minute_unclassified(monkeypatch, products):
    frame, minute, fs = products
    monkeypatch.setitem(sys.modules, "sklearn", None)
    classify.load_model.cache_clear()
    try:
        out, meta = classify.classify_minutes(frame, minute, fs)
    finally:
        classify.load_model.cache_clear()
    assert out["ml_class"].eq(classify.UNCLASSIFIED).all() and len(out) == len(minute)
    assert "scikit-learn could not be imported" in meta["ml_classification"]


def test_importing_the_pipeline_does_not_import_scikit_learn():
    code = "import sys, scintkit.pipeline; assert 'sklearn' not in sys.modules and 'joblib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
