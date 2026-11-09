"""Decode lvl0 receiver codes into constellation, PRN, signal and frequency columns."""

import logging

import numpy as np
import pandas as pd

from scintkit.constants import (
    CONSTELLATIONS,
    GLONASS_K,
    GLONASS_L1_MHZ,
    GLONASS_L1_STEP_MHZ,
    GLONASS_L2_MHZ,
    GLONASS_L2_STEP_MHZ,
    PSEUDORANGE_MAX_M,
    PSEUDORANGE_MIN_M,
    RINEX_PREFIX,
    RINEX_SVID_OFFSET,
    SIGNALS,
)

logger = logging.getLogger(__name__)

SATELLITE_COLUMNS = ["cons", "prn", "sig_1", "freq_1", "sig_2", "freq_2"]


def satellite(code, svid):
    """``(cons, prn, sig_1, freq_1, sig_2, freq_2)`` for a constellation code and svid; MHz, None/NaN if no signal."""
    cons = CONSTELLATIONS[code]
    prn = f"{RINEX_PREFIX[cons]}{svid - RINEX_SVID_OFFSET.get(cons, 0):02d}"
    for name, first, last, sig_1, freq_1, sig_2, freq_2 in SIGNALS:
        if name == cons and first <= svid <= last:
            if cons == "GLO":
                k = GLONASS_K.get(svid, np.nan)
                freq_1 = GLONASS_L1_MHZ + GLONASS_L1_STEP_MHZ * k
                freq_2 = GLONASS_L2_MHZ + GLONASS_L2_STEP_MHZ * k
            return cons, prn, sig_1, freq_1, sig_2, np.nan if freq_2 is None else freq_2
    return cons, prn, None, np.nan, None, np.nan


def format_lvl0(df):
    """Add constellation names, RINEX ``prn``, ``sig_1/2`` and ``freq_1/2`` (MHz); blank absent channels.

    A channel is absent in a row (snr, cph, rng -> NaN) when the satellite has no
    such signal or any of its snr, cph, rng is 0; pseudoranges outside
    1.8e7-4.5e7 m are NaN. Keeps row order; returns a fresh RangeIndex.
    """
    known = df["cons"].isin(list(CONSTELLATIONS))
    if not known.all():
        logger.warning("dropping %d rows with unknown constellation codes", (~known).sum())
    df = df[known].reset_index(drop=True)

    key = df["cons"].to_numpy(np.int64) * 1000 + df["svid"].to_numpy(np.int64)
    codes = np.unique(key)
    table = pd.DataFrame([satellite(int(k // 1000), int(k % 1000)) for k in codes],
                         index=codes, columns=SATELLITE_COLUMNS)
    rows = table.reindex(key)
    for column in SATELLITE_COLUMNS:
        df[column] = rows[column].to_numpy()

    for i in (1, 2):
        snr, cph, rng = (df[f"{name}{i}"] for name in ("snr", "cph", "rng"))
        absent = df[f"sig_{i}"].isna() | snr.eq(0) | cph.eq(0) | rng.eq(0)
        df[f"snr{i}"] = snr.astype(np.float32).mask(absent)
        df[f"cph{i}"] = cph.astype(float).mask(absent)
        df[f"rng{i}"] = rng.astype(float).mask(absent | ~rng.between(PSEUDORANGE_MIN_M, PSEUDORANGE_MAX_M))
    return df


def first_samples(df, interval):
    """First full-rate row of every satellite and ``interval`` ("1s", "1min"); ``datetime`` = interval start.

    Rows must be in time order within each satellite. The input is not modified.
    """
    start = df["datetime"].dt.floor(interval)
    keep = ~pd.MultiIndex.from_arrays([start, df["prn"]]).duplicated()
    out = df.loc[keep].copy()
    out["datetime"] = start[keep]
    return out.reset_index(drop=True)
