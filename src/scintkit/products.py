"""Per-minute scintillation products, their quality flags, and lvl2/lvl3 assembly."""

import logging

import numpy as np

from scintkit import schema
from scintkit.amplitude import add_detrended_snr, compute_s4, compute_s4_corrected, compute_tau
from scintkit.constants import S4_MIN_SAMPLE_FRACTION, SIGMA_PHI_MAX_DROPPED_SAMPLES
from scintkit.formatting import first_samples, format_lvl0
from scintkit.phase import process_phases
from scintkit.tec import add_tec_columns
from scintkit.timing import detect_sampling_rate, fix_epochs

logger = logging.getLogger(__name__)


def compute_sigma_phi(phase):
    """Standard deviation of detrended phase (rad)."""
    phase = phase.dropna()
    return np.std(phase) if len(phase) > 0 else np.nan


def compute_n_samples(col):
    return int(col.notna().sum())


def compute_n_cycleslips(cycleslips):
    return int(cycleslips.sum())


def _add_quality_flags(minute, fs):
    """Add 0/1 ``sigma_phi_quality_flag_*`` and ``s4_quality_flag_*`` (1 = bad); ``fs`` in Hz.

    sigma-phi is bad near an arc edge or gap, with more than
    SIGMA_PHI_MAX_DROPPED_SAMPLES missing, or for GLONASS; S4 is bad with fewer
    than S4_MIN_SAMPLE_FRACTION of the expected samples.
    """
    expected = fs * 60
    is_glonass = minute["prn"].str.startswith("R")
    for i in (1, 2):
        sigma_phi_bad = (
            minute.pop(f"_edge_gap_{i}")
            | minute[f"n_sigphi_{i}"].lt(expected - SIGMA_PHI_MAX_DROPPED_SAMPLES)
            | is_glonass
        )
        minute[f"sigma_phi_quality_flag_{i}"] = sigma_phi_bad.astype(np.int8)
        minute[f"s4_quality_flag_{i}"] = minute[f"n_s4_{i}"].lt(expected * S4_MIN_SAMPLE_FRACTION).astype(np.int8)
    return minute


def minute_products(df, fs):
    """One row per satellite and UTC minute (``datetime`` = minute start) with the channel 1/2 products.

    ``df`` is the full-rate frame after process_phases; ``fs`` in Hz. See docs/lvl3.md.
    """
    for i in (1, 2):
        df = add_detrended_snr(df, i=i)
    agg = {}
    for i in (1, 2):
        phase, snr = f"detrended_noclk_cph{i}", f"snr{i}"
        agg[f"s4_{i}"] = (snr, compute_s4)
        agg[f"s4_corrected_{i}"] = (snr, compute_s4_corrected)
        agg[f"s4_detrended_{i}"] = (f"snr{i}_new", compute_s4)
        agg[f"tau_{i}"] = (snr, lambda x: compute_tau(x, fs))
        agg[f"sigma_phi_{i}"] = (phase, compute_sigma_phi)
        agg[f"n_s4_{i}"] = (snr, compute_n_samples)
        agg[f"n_sigphi_{i}"] = (phase, compute_n_samples)
        agg[f"n_cycleslip_{i}"] = (f"cycleslips_cph{i}", compute_n_cycleslips)
        agg[f"_edge_gap_{i}"] = (f"edgegap_mask_cph{i}", "any")
    minute = df.groupby([df["prn"], df["datetime"].dt.floor("1min")], sort=False).agg(**agg)
    return _add_quality_flags(minute.reset_index(), fs)


def compute_products(df, fs=None):
    """lvl0 frame -> ``(frame, minute, fs)``.

    ``frame`` is the full-rate frame (receiver epochs fixed, signals decoded,
    detrended phase and TEC added), ``minute`` the per-minute products, and
    ``fs`` the sampling rate in Hz (detected from the data when None).
    """
    fs = detect_sampling_rate(df) if fs is None else fs
    logger.info("sampling rate %g Hz", fs)
    df = format_lvl0(fix_epochs(df, fs))
    logger.info("processing phases")
    df = process_phases(df, fs)
    logger.info("computing TEC")
    df = add_tec_columns(df, pair="12", fs=fs, copy=False)
    logger.info("computing minute products")
    return df, minute_products(df, fs), fs


def assemble(frame, minute, interval):
    """lvl3 (``interval="1min"``) or lvl2 (``"1s"``) table: first sample per satellite and interval
    plus that satellite-minute's products, conformed to schema.LVL3_COLUMNS or schema.COLUMNS.

    For lvl3, ``minute`` must carry the classification columns (see machine_learning.classify).
    """
    rows = first_samples(frame, interval)
    rows["_minute"] = rows["datetime"].dt.floor("1min")
    rows = rows.merge(minute.rename(columns={"datetime": "_minute"}), on=["prn", "_minute"], how="left")
    return schema.conform(rows, schema.LVL3_COLUMNS if interval == "1min" else schema.COLUMNS)
