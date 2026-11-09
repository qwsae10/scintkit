import numpy as np
import pandas as pd

from scintkit.products import minute_products

FS = 10.0


def _minute(prn, minute, phase_count, snr_count, edge_gap_channel=None):
    """One satellite-minute of a processed 10 Hz frame with the given valid-sample counts."""
    n = 600
    phase = np.r_[np.ones(phase_count), np.full(n - phase_count, np.nan)]
    snr = np.r_[np.full(snr_count, 40.0), np.full(n - snr_count, np.nan)]
    frame = pd.DataFrame({
        "datetime": pd.Timestamp(minute) + pd.to_timedelta(np.arange(n) / FS, unit="s"),
        "prn": prn,
    })
    for i in (1, 2):
        frame[f"detrended_noclk_cph{i}"] = phase
        frame[f"cycleslips_cph{i}"] = False
        frame[f"edgegap_mask_cph{i}"] = np.r_[edge_gap_channel == i, np.zeros(n - 1, dtype=bool)]
        frame[f"snr{i}"] = snr
    return frame


def _products(*minutes):
    result = minute_products(pd.concat(minutes, ignore_index=True), FS)
    return result.set_index(["prn", "datetime"])


def test_sigma_phi_and_s4_flags_are_separate_and_inclusive_at_thresholds():
    result = _products(
        # At both exact thresholds: 10 phase samples dropped and 80% C/N0.
        _minute("G01", "2024-01-01 00:00", 590, 480),
        _minute("G01", "2024-01-01 00:01", 589, 479),
        _minute("G02", "2024-01-01 00:00", 600, 600, edge_gap_channel=1),
        _minute("R03", "2024-01-01 00:00", 600, 600),
    )
    flags = result[["sigma_phi_quality_flag_1", "s4_quality_flag_1"]]
    assert flags.loc[("G01", pd.Timestamp("2024-01-01 00:00"))].tolist() == [0, 0]
    assert flags.loc[("G01", pd.Timestamp("2024-01-01 00:01"))].tolist() == [1, 1]
    assert flags.loc[("G02", pd.Timestamp("2024-01-01 00:00"))].tolist() == [1, 0]
    assert flags.loc[("R03", pd.Timestamp("2024-01-01 00:00"))].tolist() == [1, 0]

    first = result.loc[("G01", pd.Timestamp("2024-01-01 00:00"))]
    assert first["n_sigphi_1"] == 590
    assert first["n_s4_1"] == 480
    assert not any(c.startswith("_") for c in result.columns)


def test_channel_2_edge_gap_sets_only_channel_2_sigma_phi_flag():
    result = _products(_minute("G01", "2024-01-01 00:00", 600, 600, edge_gap_channel=2)).iloc[0]

    assert result["sigma_phi_quality_flag_1"] == 0
    assert result["sigma_phi_quality_flag_2"] == 1
