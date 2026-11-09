from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scintkit.machine_learning.features import FEATURES, band_powers_db, compute_features, minute_grid
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
            keep &= ~((t >= 150) & (t < 155))  # minute 2 of G03 fails the sample gate
        frames.append(frame[keep])
    df = pd.concat(frames, ignore_index=True).sort_values(["_order", "cons", "svid"], kind="stable")
    return df.drop(columns="_order").reset_index(drop=True)


FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "ml_features_golden.csv"  # old feature code (commit 9a34c65) on synthetic_lvl0()


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
