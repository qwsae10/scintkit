"""Decode lvl0 receiver codes into constellation, PRN, signal and frequency columns."""

import pandas as pd

from scintkit.constants import (
    CONSTELLATION_PREFIX,
    FREQUENCY_MHZ_BY_SIGNAL,
    INVALID_SVID,
    NUMERIC_CONSTELLATION,
    SAMPLES_PER_MINUTE,
    SAMPLES_PER_MINUTE_TOLERANCE,
    SIGNALS_BY_CONSTELLATION,
)


def make_prn(df):
    """PRN labels such as ``G05`` from ``cons`` and ``svid``."""
    # Map the few unique SVIDs rather than formatting millions of rows.
    svid_labels = {value: str(int(value)).zfill(2) for value in df["svid"].unique()}
    return df["cons"].map(CONSTELLATION_PREFIX) + df["svid"].map(svid_labels)


def zero_cph_snr_to_nan(df):
    """Treat zero carrier phase and zero SNR as missing."""
    for column in df.columns:
        if column.startswith(("cph", "snr")) and column[3:].isdigit():
            df[column] = df[column].mask(df[column].eq(0))
    return df


def add_sigs(df):
    """Add ``sig_1..3`` (if absent) and ``freq_1..3`` in MHz."""
    if 'sig_1' not in df.columns:
        for channel in (1, 2, 3):
            signal_map = {
                constellation: signals[channel - 1]
                for constellation, signals in SIGNALS_BY_CONSTELLATION.items()
            }
            df[f'sig_{channel}'] = df['cons'].map(signal_map)

    for channel in (1, 2, 3):
        df[f'freq_{channel}'] = df[f'sig_{channel}'].map(FREQUENCY_MHZ_BY_SIGNAL)
    return df


def format_lvl0(df):
    """Decode a lvl0 frame and add ``minbin``, ``prn``, signal and frequency columns.

    Returns a frame with a fresh RangeIndex, which later stages rely on.
    """
    codes = pd.to_numeric(df['cons'], errors='coerce')
    if codes.notna().all():
        df['cons'] = codes.map(NUMERIC_CONSTELLATION)

    invalid_glonass = (df['cons'] == 'GLO') & (df['svid'] == INVALID_SVID)
    if invalid_glonass.any():
        df = df.loc[~invalid_glonass]
    if not df.index.equals(pd.RangeIndex(len(df))):
        df = df.reset_index(drop=True)

    df['minbin'] = df['datetime'].dt.floor('1min')
    df['prn'] = make_prn(df)
    df = add_sigs(df)
    return zero_cph_snr_to_nan(df)


def detect_sampling_rate(df):
    """Sampling rate in Hz from the busiest satellite-minute, or None if unrecognised."""
    n = df.groupby(['minbin', 'prn']).size().max()
    for samples in SAMPLES_PER_MINUTE:
        if abs(n - samples) < SAMPLES_PER_MINUTE_TOLERANCE:
            return samples / 60
    return None


def make_1sec(df):
    """First non-null value per column for each satellite and second (adds ``secbin``)."""
    df['secbin'] = df['datetime'].dt.floor('1s')
    return df.groupby(['secbin', 'prn']).first().reset_index()


def make_1min(df):
    """First non-null value per column for each satellite and minute."""
    return df.groupby(['minbin', 'prn']).first().reset_index()
