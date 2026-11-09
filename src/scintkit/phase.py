"""Carrier-phase detrending: cycle-slip repair, high-pass filter, edge mask, receiver clock."""

import logging

import numpy as np
import pandas as pd
from scipy import signal

from scintkit.constants import (
    CLOCK_ELEVATION_MASK_DEG,
    CYCLE_SLIP_THRESHOLD_RAD,
    EDGE_GAP_S,
    EDGE_PAD_S,
    HIGHPASS_CUTOFF_HZ,
    L1_FREQUENCY_MHZ,
    MAX_SLIP_FRACTION,
    REPAIR_WINDOW_S,
)
from scintkit.formatting import detect_sampling_rate

logger = logging.getLogger(__name__)


def repair_discontinuities_pos(vec, fs, threshold=1, svid=None):
    """Remove steps larger than ``threshold`` from a series sampled at ``fs`` Hz.

    Returns ``(repaired, slip_mask, n_slips)``. Each contiguous finite block is
    rebuilt from its first value; arcs with too many slips are returned as is.
    """
    y = pd.Series(vec).copy()
    finite = y.notna().to_numpy()
    if not finite.any():
        return y, pd.Series(False, index=y.index), 0

    window = int(REPAIR_WINDOW_S * fs)
    delt = y.diff()
    trend = (
        delt
        .rolling(window=window, center=True, min_periods=max(3, window // 10))
        .median()
        .bfill()
        .ffill()
    )
    residual = (delt - trend).abs()
    good = residual.le(threshold) | residual.isna()
    slip_mask = ~good
    n_slips = int(slip_mask.sum())

    if n_slips / len(vec) > MAX_SLIP_FRACTION and len(vec) > 10:
        logger.debug("many cycle slips for %s: %d/%d", svid, n_slips, len(vec))
        return pd.Series(vec), slip_mask, n_slips

    delt_clean = delt.where(good, np.nan)
    result = pd.Series(np.nan, index=y.index, dtype=float)
    padded = np.r_[False, finite, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, stop in zip(starts, stops):
        result.iloc[start] = y.iloc[start]
        if start + 1 < stop:
            increments = delt_clean.iloc[start + 1:stop].interpolate(
                limit_direction="both"
            ).fillna(0.0)
            result.iloc[start + 1:stop] = (y.iloc[start] + increments.cumsum()).to_numpy()

    return result, slip_mask, n_slips


def filter_signal_cascaded(x, f_N=HIGHPASS_CUTOFF_HZ, fs=10):
    """Causal 6th-order Butterworth high-pass at ``f_N`` Hz, as three biquads."""
    omega_N = 2 * np.pi * f_N
    y = x
    for a in (np.sqrt(2 + np.sqrt(3)), np.sqrt(2), np.sqrt(2 - np.sqrt(3))):
        b_z, a_z = signal.bilinear([1, 0, 0], [1, a * omega_N, omega_N**2], fs)
        y = signal.lfilter(b_z, a_z, y)
    return y


def make_edge_gap_mask(time, phase, fs, gap_seconds=EDGE_GAP_S, pad_seconds=EDGE_PAD_S):
    """True within ``pad_seconds`` of arc ends, time gaps and missing phase samples."""
    phase = np.asarray(phase, dtype=float)
    n = len(phase)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    pad = int(round(fs * pad_seconds))
    dt = pd.Series(pd.to_datetime(time)).diff().dt.total_seconds().to_numpy()
    break_idx = np.where((dt > gap_seconds) | ~np.isfinite(phase))[0]
    for p in [0, n - 1, *break_idx.tolist()]:
        mask[max(0, p - pad):min(n, p + pad + 1)] = True
    return mask


def highpass_phase(df, in_col="cph1", out_col="detrended_cph1", tr=CYCLE_SLIP_THRESHOLD_RAD,
                   fs=None, f_N=HIGHPASS_CUTOFF_HZ):
    """Add slip-repaired, high-pass filtered phase in radians, plus slip and edge masks.

    Writes ``out_col``, ``cycleslips_*`` and ``edgegap_mask_*``. Phase is all NaN
    when the sampling rate is unknown.
    """
    slip_col = out_col.replace("detrended", "cycleslips")
    mask_col = out_col.replace("detrended", "edgegap_mask")
    df[out_col] = np.nan
    df[slip_col] = False
    df[mask_col] = False
    if fs is None:
        fs = detect_sampling_rate(df)
    if fs is None:
        return df

    phase_all = df[in_col].to_numpy(dtype=float)
    time_all = df["datetime"].to_numpy()
    out_loc, slip_loc, mask_loc = (df.columns.get_loc(c) for c in (out_col, slip_col, mask_col))

    for prn, pos in df.groupby("prn", sort=False).indices.items():
        phase = phase_all[pos]
        finite = np.isfinite(phase)
        if not finite.any():
            continue

        # Interpolate across missing samples so the filter sees a continuous arc.
        phase_rad = pd.Series(phase).interpolate(limit_direction="both") * 2 * np.pi
        repaired, cycleslips, _ = repair_discontinuities_pos(phase_rad, fs=fs, threshold=tr, svid=prn)

        filtered = np.asarray(
            filter_signal_cascaded(repaired.to_numpy(dtype=float), f_N=f_N, fs=fs), dtype=float
        )
        filtered[~finite] = np.nan
        slip_mask = cycleslips.to_numpy(dtype=bool)
        slip_mask[~finite] = False

        df.iloc[pos, out_loc] = filtered
        df.iloc[pos, slip_loc] = slip_mask
        df.iloc[pos, mask_loc] = make_edge_gap_mask(time_all[pos], phase, fs=fs)

    return df


def highpass_all_phases(df, fs=None, tr=CYCLE_SLIP_THRESHOLD_RAD):
    """Detrend every ``cph1..3`` column; the slip threshold scales with frequency."""
    for i in range(1, 4):
        if f"cph{i}" not in df.columns:
            continue
        freq_col = f"freq_{i}"
        scaled_tr = tr * df[freq_col].median() / L1_FREQUENCY_MHZ if freq_col in df.columns else tr
        df = highpass_phase(df, in_col=f"cph{i}", out_col=f"detrended_cph{i}", tr=scaled_tr, fs=fs)
    return df


def estimate_clock(df, elev_mask=CLOCK_ELEVATION_MASK_DEG):
    """Add ``clock_term``: per-epoch median of detrended phase / frequency (rad/MHz).

    Uses only satellites above ``elev_mask`` degrees; also adds ``v1..3``.
    """
    value_cols = []
    for i in (1, 2, 3):
        cph_col, freq_col = f"detrended_cph{i}", f"freq_{i}"
        if cph_col in df.columns and freq_col in df.columns:
            df[f"v{i}"] = df[cph_col] / df[freq_col]
            value_cols.append(f"v{i}")

    if not value_cols:
        df["clock_term"] = np.nan
        return df

    clock_df = df[(df["elev"] > elev_mask) & (df["elev"] < 90)]
    median_curve = (
        clock_df.melt(id_vars="datetime", value_vars=value_cols)
        .groupby("datetime")["value"]
        .median()
    )
    df["clock_term"] = df["datetime"].map(median_curve)
    return df


def clock_correction(df):
    """Add ``detrended_noclk_cph1..3``: detrended phase minus the receiver clock (rad)."""
    for i in (1, 2, 3):
        cph_col, freq_col = f"detrended_cph{i}", f"freq_{i}"
        if cph_col in df.columns and "clock_term" in df.columns and freq_col in df.columns:
            df[f"detrended_noclk_cph{i}"] = df[cph_col] - df["clock_term"] * df[freq_col]
    return df


def process_phases(df, fs=None, tr=CYCLE_SLIP_THRESHOLD_RAD):
    """Detrend all phases and remove the common receiver clock."""
    fs = detect_sampling_rate(df) if fs is None else fs
    df = highpass_all_phases(df, fs, tr)
    df = estimate_clock(df)
    return clock_correction(df)
