import numpy as np
import pandas as pd

from scintkit.services import compute


def _minute(prn, minute, phase_count, snr_count, edge_gap=False):
    n_rows = 600
    return pd.DataFrame(
        {
            "prn": prn,
            "minbin": pd.Timestamp(minute),
            "detrended_noclk_cph1": np.r_[
                np.ones(phase_count), np.full(n_rows - phase_count, np.nan)
            ],
            "edgegap_mask_cph1": np.r_[
                np.full(1, edge_gap), np.zeros(n_rows - 1, dtype=bool)
            ],
            "snr1": np.r_[
                np.full(snr_count, 40.0), np.full(n_rows - snr_count, np.nan)
            ],
        }
    )


def test_add_products_creates_separate_sigma_phi_and_s4_quality_flags(monkeypatch):
    frame = pd.concat(
        [
            # At both exact thresholds: 10 phase samples dropped and 80% C/N0.
            _minute("G01", "2024-01-01 00:00", 590, 480),
            _minute("G01", "2024-01-01 00:01", 589, 479),
            _minute("G02", "2024-01-01 00:00", 600, 600, edge_gap=True),
            _minute("R03", "2024-01-01 00:00", 600, 600),
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(compute, "temp_formating", lambda df: df.copy())
    monkeypatch.setattr(compute, "process_phases", lambda df: df)
    monkeypatch.setattr(compute, "detect_sampling_rate", lambda df: 10.0)

    result = compute.add_products(frame)
    minute_products = result.groupby(["prn", "minbin"], sort=False).first()

    assert (
        minute_products.loc[
            ("G01", pd.Timestamp("2024-01-01 00:00")), "sigma_phi_quality_flag_1"
        ]
        == 0
    )
    assert (
        minute_products.loc[
            ("G01", pd.Timestamp("2024-01-01 00:00")), "s4_quality_flag_1"
        ]
        == 0
    )

    assert (
        minute_products.loc[
            ("G01", pd.Timestamp("2024-01-01 00:01")), "sigma_phi_quality_flag_1"
        ]
        == 1
    )
    assert (
        minute_products.loc[
            ("G01", pd.Timestamp("2024-01-01 00:01")), "s4_quality_flag_1"
        ]
        == 1
    )

    assert (
        minute_products.loc[
            ("G02", pd.Timestamp("2024-01-01 00:00")), "sigma_phi_quality_flag_1"
        ]
        == 1
    )
    assert (
        minute_products.loc[
            ("G02", pd.Timestamp("2024-01-01 00:00")), "s4_quality_flag_1"
        ]
        == 0
    )

    assert (
        minute_products.loc[
            ("R03", pd.Timestamp("2024-01-01 00:00")), "sigma_phi_quality_flag_1"
        ]
        == 1
    )
    assert (
        minute_products.loc[
            ("R03", pd.Timestamp("2024-01-01 00:00")), "s4_quality_flag_1"
        ]
        == 0
    )

    assert (
        minute_products.loc[("G01", pd.Timestamp("2024-01-01 00:00")), "n_sigphi_1"]
        == 590
    )
    assert (
        minute_products.loc[("G01", pd.Timestamp("2024-01-01 00:00")), "n_s4_1"] == 480
    )
    assert "n_1" not in result.columns
    assert "_s4_sample_count_1" not in result.columns
    assert "quality_1" not in result.columns


def test_channel_2_edge_gap_mask_sets_channel_2_sigma_phi_flag(monkeypatch):
    n_rows = 600
    frame = pd.DataFrame(
        {
            "prn": "G01",
            "minbin": pd.Timestamp("2024-01-01 00:00"),
            "detrended_noclk_cph1": np.ones(n_rows),
            "detrended_noclk_cph2": np.ones(n_rows),
            "edgegap_mask_cph1": np.zeros(n_rows, dtype=bool),
            "edgegap_mask_cph2": np.r_[True, np.zeros(n_rows - 1, dtype=bool)],
        }
    )

    monkeypatch.setattr(compute, "temp_formating", lambda df: df.copy())
    monkeypatch.setattr(compute, "process_phases", lambda df: df)
    monkeypatch.setattr(compute, "detect_sampling_rate", lambda df: 10.0)

    result = compute.add_products(frame).iloc[0]

    assert result["sigma_phi_quality_flag_1"] == 0
    assert result["sigma_phi_quality_flag_2"] == 1


def test_missing_channel_edge_gap_mask_fails_closed():
    products = pd.DataFrame(
        {
            "prn": ["G01"],
            "n_sigphi_2": [600],
        }
    )

    result = compute._add_quality_flags(products, fs=10.0)

    assert result.loc[0, "sigma_phi_quality_flag_2"] == 1
