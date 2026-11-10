"""Amplitude scintillation: SNR detrending, S4, bias-corrected S4 and tau."""

import numpy as np
import pandas as pd

from scintkit.constants import (
    DETRENDED_SNR_OFFSET_DB,
    MIN_TAU_SECONDS,
    MIN_UNIQUE_SNR,
    SNR_DETREND_WINDOW,
    TAU_CORRELATION_LEVEL,
)


def add_detrended_snr(df, i=1):
    """Add ``snr{i}_avg`` (centered moving average, dB) and ``snr{i}_new`` (detrended SNR, dB).

    The detrended SNR is offset by DETRENDED_SNR_OFFSET_DB so it stays positive.
    Requires a RangeIndex.
    """
    snr_col, avg_col, new_col = f"snr{i}", f"snr{i}_avg", f"snr{i}_new"
    for _, group in df.groupby('prn', sort=False, observed=True):
        group = group.sort_values('datetime')
        average = (
            group.set_index('datetime')[snr_col]
            .rolling(SNR_DETREND_WINDOW, center=True)
            .mean()
            .to_numpy()
        )
        df.loc[group.index, avg_col] = average
        df.loc[group.index, new_col] = group[snr_col].to_numpy() - average + DETRENDED_SNR_OFFSET_DB
    return df


def compute_s4(snr):
    """S4 from SNR samples in dB; NaN if empty or the lowest sample is exactly 0 dB."""
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan
    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)
    return std / mean if ((mean > 0) & (np.min(snr) != 0)) else np.nan


def compute_s4_corrected(snr):
    """S4 with the C/N0 noise contribution removed (Van Dierendonck 1993)."""
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan
    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)
    if mean <= 0:
        return np.nan
    s4 = std / mean
    # S4 that receiver noise alone produces at this mean C/N0 (linear, Hz).
    s4_noise = np.sqrt(100 / mean * (1 + 500 / (19 * mean)))
    val = s4**2 - s4_noise**2
    return np.sqrt(val) if val > 0 else 0


def compute_tau(snr, fs):
    """Decorrelation time (s): the first lag at which the normalized autocorrelation of the
    linear intensity 10**(snr/10) drops below 0.5.

    NaN with fewer than MIN_TAU_SECONDS * fs samples, flat input, or no such lag.
    """
    snr = snr.dropna()
    if len(snr) < MIN_TAU_SECONDS * fs or snr.nunique() < MIN_UNIQUE_SNR:
        return np.nan
    intensity = 10 ** (snr.to_numpy(dtype=float) / 10)
    x = intensity - intensity.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    below = np.flatnonzero(ac < TAU_CORRELATION_LEVEL * ac[0])
    return below[0] / fs if len(below) else np.nan
