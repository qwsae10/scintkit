import numpy as np
import pandas as pd
import pytest

from scintkit.formatting import format_lvl0, satellite
from scintkit.timing import detect_sampling_rate, fix_epochs


def _row(cons, svid, **values):
    row = {"cons": cons, "svid": svid, "snr1": 40, "snr2": 40,
           "cph1": 1e8, "cph2": 8e7, "rng1": 2e7, "rng2": 2e7}
    return {**row, **values}


def test_signal_table():
    assert satellite(6, 5) == ("GLO", "R05", "GLO_L1OF", 1602.5625, "GLO_L2OF", 1246.4375)
    assert satellite(6, 26)[3] == 1602 - 6 * 0.5625
    assert satellite(3, 11) == ("BDS", "C11", "BDS_B1I", 1561.098, "BDS_B2I", 1207.14)
    assert satellite(3, 23)[:4] == ("BDS", "C23", "BDS_B1I", 1561.098)
    assert satellite(3, 23)[4] is None and np.isnan(satellite(3, 23)[5])
    assert satellite(5, 194)[1:3] == ("J02", "QZS_L1CA")
    assert satellite(1, 131)[1:3] == ("S31", "SBAS_L1CA")
    assert satellite(2, 11)[1:] == ("E11", "GAL_E1", 1575.42, "GAL_E5b", 1207.14)


def test_absent_channels_and_bad_pseudoranges_become_nan():
    df = format_lvl0(pd.DataFrame([
        _row(0, 1),
        _row(0, 2, cph2=0.0),  # no L2C: channel 2 absent
        _row(3, 23),  # BeiDou-3: channel 2 is junk
        _row(1, 131),  # SBAS: no channel 2
        _row(2, 5, rng1=3.2e8),  # impossible pseudorange
        _row(4, 1),  # unknown constellation code: dropped
    ]))
    assert df["prn"].tolist() == ["G01", "G02", "C23", "S31", "E05"]
    assert df[["snr2", "cph2", "rng2"]].isna().all(axis=1).tolist() == [False, True, True, True, False]
    assert df[["snr1", "cph1"]].notna().all().all()
    assert np.isnan(df["rng1"].iloc[4])


def _synthetic(fs, seconds=5, n_sats=4, start="2024-01-01"):
    times = pd.Timestamp(start) + pd.to_timedelta(np.arange(int(seconds * fs)) / fs, unit="s")
    return pd.DataFrame({
        "datetime": np.repeat(times, n_sats),
        "cons": 0,
        "svid": np.tile(np.arange(1, n_sats + 1), len(times)),
    })


@pytest.mark.parametrize("fs", [10, 20, 25])
def test_rate_detection(fs):
    df = _synthetic(fs)
    df = df.drop(index=df.index[::97])  # a few dropped samples do not matter
    assert detect_sampling_rate(df) == fs


def test_rate_detection_fails_loudly():
    with pytest.raises(ValueError):
        detect_sampling_rate(_synthetic(20).iloc[:0])


def test_collapsed_and_week_wrapped_epochs_get_distinct_times():
    # Last seconds of a GPS week: float32 time-of-week has 62.5 ms steps, so
    # 20 Hz epochs collapse; after the boundary time-of-week restarts at 0.
    week_end = pd.Timestamp("2024-01-21")  # Sunday 00:00, GPS week boundary
    true = week_end - pd.Timedelta(seconds=2) + pd.to_timedelta(np.arange(80) * 50, unit="ms")
    tow = (true - pd.Timestamp("2024-01-14")).total_seconds().to_numpy()
    stamp = pd.Timestamp("2024-01-14") + pd.to_timedelta(np.round(tow / 0.0625) * 0.0625 % (7 * 86400), unit="s")
    df = pd.DataFrame({"datetime": np.repeat(stamp, 3), "cons": 0, "svid": np.tile([1, 2, 3], len(stamp))})
    assert df.groupby("datetime").ngroups < 80

    fixed = fix_epochs(df, 20)

    assert fixed["epoch"].nunique() == 80
    times = fixed.groupby("epoch")["datetime"].first()
    assert times.is_unique and times.is_monotonic_increasing
    assert (times.to_numpy() == true.to_numpy()).mean() > 0.9


def test_untrusted_reconstruction_keeps_raw_times(caplog):
    df = _synthetic(20, seconds=2)
    df.loc[df.index[-4:], "datetime"] -= pd.Timedelta(hours=1)  # time runs backwards
    fixed = fix_epochs(df, 20)
    assert fixed["datetime"].equals(df["datetime"])
    assert "keeping raw timestamps" in caplog.text
