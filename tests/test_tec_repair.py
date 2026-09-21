import numpy as np
import pandas as pd

from scintkit.services.compute import add_tec_columns
from scintkit.services.phase_detrend import repair_discontinuities_pos


def test_repair_discontinuities_anchors_at_first_finite_value():
    values = pd.Series([np.nan, 10.0, 11.0, 12.0])

    repaired, _, _ = repair_discontinuities_pos(values, fs=1, threshold=2)

    assert np.isnan(repaired.iloc[0])
    np.testing.assert_allclose(repaired.iloc[1:], [10.0, 11.0, 12.0])


def test_repair_discontinuities_preserves_valid_blocks_after_gaps():
    values = pd.Series([1.0, 2.0, np.nan, 10.0, np.nan, 20.0, 21.0])

    repaired, _, _ = repair_discontinuities_pos(values, fs=1)

    assert repaired.notna().equals(values.notna())


def test_carrier_tec_survives_missing_first_phase_pair():
    frame = pd.DataFrame(
        {
            "prn": ["G15"] * 4,
            "cph1": [1.0, 2.0, 3.0, 4.0],
            "cph2": [np.nan, 1.5, 2.5, 3.5],
            "rng1": [20_000_000.0] * 4,
            "rng2": [20_000_010.0] * 4,
            "freq_1": [1575.42] * 4,
            "freq_2": [1227.60] * 4,
        }
    )

    result = add_tec_columns(frame, pair="12", fs=1)

    assert np.isnan(result["tec_cph12"].iloc[0])
    assert result["tec_cph12"].iloc[1:].notna().all()


def test_zero_pseudorange_is_treated_as_missing():
    frame = pd.DataFrame(
        {
            "prn": ["C21"] * 3,
            "cph1": [1.0, 2.0, 3.0],
            "cph2": [1.0, 2.0, 3.0],
            "rng1": [20_000_000.0] * 3,
            "rng2": [0.0] * 3,
            "freq_1": [1561.098] * 3,
            "freq_2": [1207.14] * 3,
        }
    )

    result = add_tec_columns(frame, pair="12", fs=1)

    assert result["tec_rng12"].isna().all()
    assert result["tec_cph12"].isna().all()


def test_carrier_tec_median_is_leveled_to_pseudorange_by_time_segment():
    first_segment = pd.date_range("2024-01-01 00:00:00", periods=3, freq="s")
    second_segment = pd.date_range("2024-01-01 00:10:00", periods=3, freq="s")
    frame = pd.DataFrame(
        {
            "datetime": first_segment.append(second_segment),
            "prn": ["G15"] * 6,
            "cph1": [1_000.0] * 3 + [2_000.0] * 3,
            "cph2": [900.0] * 3 + [1_800.0] * 3,
            "rng1": [20_000_000.0] * 6,
            "rng2": [20_000_010.0] * 3 + [20_000_100.0] * 3,
            "freq_1": [1575.42] * 6,
            "freq_2": [1227.60] * 6,
        }
    )

    result = add_tec_columns(frame, pair="12", fs=1, max_gap="5min")
    segment = result["datetime"].diff().gt(pd.Timedelta("5min")).cumsum()

    for _, values in result.groupby(segment):
        assert np.isclose(
            values["tec_cph12"].median(),
            values["tec_rng12"].median(),
        )


def test_carrier_data_outage_over_five_minutes_starts_new_segment():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00:00", periods=10, freq="min"),
            "prn": ["G15"] * 10,
            "cph1": [1_000.0, 1_000.0] + [np.nan] * 6 + [2_000.0, 2_000.0],
            "cph2": [900.0, 900.0] + [np.nan] * 6 + [1_800.0, 1_800.0],
            "rng1": [20_000_000.0] * 10,
            "rng2": [20_000_010.0] * 8 + [20_000_100.0] * 2,
            "freq_1": [1575.42] * 10,
            "freq_2": [1227.60] * 10,
        }
    )

    result = add_tec_columns(frame, pair="12", fs=1, max_gap="5min")

    for rows in ([0, 1], [8, 9]):
        assert np.isclose(
            result.loc[rows, "tec_cph12"].median(),
            result.loc[rows, "tec_rng12"].median(),
        )
