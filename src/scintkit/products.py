"""Per-minute scintillation products and their quality flags."""

import logging

import numpy as np
import pandas as pd

from scintkit.amplitude import add_detrended_snr, compute_s4, compute_s4_corrected, compute_tau
from scintkit.constants import GLONASS_PREFIX, S4_MIN_SAMPLE_FRACTION, SIGMA_PHI_MAX_DROPPED_SAMPLES
from scintkit.formatting import detect_sampling_rate, format_lvl0
from scintkit.phase import process_phases
from scintkit.tec import add_tec_columns

logger = logging.getLogger(__name__)

GROUP_COLUMNS = ["prn", "minbin"]


def compute_sigma_phi(phase):
    """Standard deviation of detrended phase (rad)."""
    phase = phase.dropna()
    return np.std(phase) if len(phase) > 0 else np.nan


def compute_n_samples(col):
    return int(col.notna().sum())


def compute_n_cycleslips(cycleslips):
    return int(cycleslips.fillna(False).sum())


def _any_edge_gap(mask):
    return int(mask.fillna(False).astype(bool).any())


def _add_quality_flags(products, fs):
    """Add 0/1 ``sigma_phi_quality_flag_*`` and ``s4_quality_flag_*`` (1 = bad).

    sigma-phi is bad near an arc edge or gap, with more than
    SIGMA_PHI_MAX_DROPPED_SAMPLES missing, or for GLONASS; S4 is bad with fewer
    than S4_MIN_SAMPLE_FRACTION of the expected samples.
    """
    expected_samples = fs * 60
    sigma_phi_min_samples = expected_samples - SIGMA_PHI_MAX_DROPPED_SAMPLES
    s4_min_samples = expected_samples * S4_MIN_SAMPLE_FRACTION
    is_glonass = products["prn"].astype(str).str.startswith(GLONASS_PREFIX)

    edge_gap_columns = []
    for i in ("1", "2", "3"):
        phase_count_col = f"n_sigphi_{i}"
        edge_gap_col = f"_sigma_phi_edge_gap_{i}"
        s4_count_col = f"n_s4_{i}"
        edge_gap_columns.append(edge_gap_col)

        if phase_count_col in products.columns:
            # Without a propagated edge/gap mask the channel cannot be trusted.
            if edge_gap_col in products.columns:
                has_edge_gap = products[edge_gap_col].astype(bool)
            else:
                has_edge_gap = pd.Series(True, index=products.index)
            sigma_phi_bad = (
                has_edge_gap
                | products[phase_count_col].lt(sigma_phi_min_samples)
                | is_glonass
            )
            products[f"sigma_phi_quality_flag_{i}"] = sigma_phi_bad.astype(np.int8)

        if s4_count_col in products.columns:
            products[f"s4_quality_flag_{i}"] = products[s4_count_col].lt(s4_min_samples).astype(np.int8)

    return products.drop(columns=edge_gap_columns, errors="ignore")


def add_products(df, fs=None):
    """Full-rate frame with TEC, detrended phase/SNR and per-minute products merged in.

    Per satellite and minute: sigma_phi_* (rad), n_sigphi_*, n_cycleslip_*,
    s4_*, s4_detrended_*, s4_corrected_*, tau_* (s), n_s4_* and quality flags.
    ``fs`` (Hz) is detected from the data when None. See docs/lvl3.md.
    """
    logger.info("Formatting")
    df = format_lvl0(df)
    logger.info("Processing phases")
    df = process_phases(df)

    if fs is None:
        fs = detect_sampling_rate(df)
    if fs is None:
        raise ValueError("Could not determine sampling rate.")

    logger.info("Computing TEC")
    if "cph1" in df.columns and "cph2" in df.columns:
        df = add_tec_columns(df, fs=fs, pair="12")
    if "cph1" in df.columns and "cph3" in df.columns:
        df = add_tec_columns(df, fs=fs, pair="13")

    logger.info("Detrending SNR")
    for i in ("1", "2", "3"):
        if f"snr{i}" in df.columns:
            df = add_detrended_snr(df, i=i)

    logger.info("Computing minute products")
    agg_dict = {}
    for i in ("1", "2", "3"):
        phase_col = f"detrended_noclk_cph{i}"
        cycleslip_col = f"cycleslips_cph{i}"
        edgegap_col = f"edgegap_mask_cph{i}"
        snr_col = f"snr{i}"

        if phase_col in df.columns:
            agg_dict[f"sigma_phi_{i}"] = (phase_col, compute_sigma_phi)
            agg_dict[f"n_sigphi_{i}"] = (phase_col, compute_n_samples)
        if cycleslip_col in df.columns:
            agg_dict[f"n_cycleslip_{i}"] = (cycleslip_col, compute_n_cycleslips)
        if edgegap_col in df.columns:
            agg_dict[f"_sigma_phi_edge_gap_{i}"] = (edgegap_col, _any_edge_gap)
        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_detrended_{i}"] = (f"snr{i}_new", compute_s4)
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
            agg_dict[f"tau_{i}"] = (snr_col, lambda x, fs=fs: compute_tau(x, fs))
            agg_dict[f"n_s4_{i}"] = (snr_col, compute_n_samples)

    if not agg_dict:
        return df

    products = df.groupby(GROUP_COLUMNS, sort=False).agg(**agg_dict).reset_index()
    products = _add_quality_flags(products, fs=fs)
    return df.merge(products, on=GROUP_COLUMNS, how="left")
