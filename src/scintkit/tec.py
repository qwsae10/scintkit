"""Slant TEC from dual-frequency carrier phase and pseudorange."""

import numpy as np
import pandas as pd

from scintkit.constants import (
    IONO_CONSTANT,
    SPEED_OF_LIGHT,
    TEC_MAX_GAP,
    TEC_SLIP_THRESHOLD_TECU,
    TECU,
)
from scintkit.phase import repair_discontinuities_pos


def _tec_factor(f1_hz, f2_hz):
    return (f1_hz**2 * f2_hz**2) / (IONO_CONSTANT * (f1_hz**2 - f2_hz**2))


def carrier_phase_tec(phi1_cyc, phi2_cyc, f1_hz, f2_hz):
    """Relative TEC in TECU from carrier phases in cycles (ambiguous offset)."""
    phi1_m = phi1_cyc * (SPEED_OF_LIGHT / f1_hz)
    phi2_m = phi2_cyc * (SPEED_OF_LIGHT / f2_hz)
    return _tec_factor(f1_hz, f2_hz) * (phi1_m - phi2_m) / TECU


def pseudorange_tec(P1_m, P2_m, f1_hz, f2_hz):
    """Absolute but noisy TEC in TECU from pseudoranges in metres."""
    return _tec_factor(f1_hz, f2_hz) * (P2_m - P1_m) / TECU


def _repair_tec_pair(values, fs, threshold=TEC_SLIP_THRESHOLD_TECU):
    """Repair the carrier and pseudorange TEC columns of an (n, 2) array independently."""
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({
        name: repair_discontinuities_pos(pd.Series(values[:, k]), fs=fs, threshold=threshold)[0]
        for k, name in enumerate(("carrier", "pseudo"))
    })


def _float_column(df, column):
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def add_tec_columns(df, pair="13", *, fs, max_gap=TEC_MAX_GAP, copy=True):
    """Add ``tec_cph{pair}`` and ``tec_rng{pair}`` in TECU for signals ``pair``, e.g. "12".

    Per PRN, data are split where the time step (or carrier outage) exceeds
    ``max_gap``. Within each segment both series are slip-repaired and the
    carrier TEC is shifted so its median matches the pseudorange TEC median at
    common epochs; it is left NaN where there is no pseudorange overlap.
    Zero pseudoranges are set to NaN in ``df``. ``fs`` is the sampling rate in Hz.
    """
    if copy:
        df = df.copy()
    gap_ns = pd.to_timedelta(max_gap).value
    a, b = pair

    for column in (f"rng{a}", f"rng{b}"):
        df[column] = df[column].mask(df[column].eq(0))
    phi1_all = _float_column(df, f"cph{a}")
    phi2_all = _float_column(df, f"cph{b}")
    rng1_all = _float_column(df, f"rng{a}")
    rng2_all = _float_column(df, f"rng{b}")
    f1_all = _float_column(df, f"freq_{a}") * 1e6
    f2_all = _float_column(df, f"freq_{b}") * 1e6
    if "datetime" in df.columns:
        time_all = pd.to_datetime(df["datetime"], errors="coerce").to_numpy(
            dtype="datetime64[ns]"
        ).astype("int64")
        nat = np.datetime64("NaT", "ns").astype("int64")
    else:
        time_all = None

    carrier_output = np.full(len(df), np.nan)
    pseudo_output = np.full(len(df), np.nan)
    for positions in df.groupby("prn", sort=False, observed=True).indices.values():
        positions = np.asarray(positions, dtype="int64")
        if time_all is not None:
            positions = positions[np.argsort(time_all[positions], kind="stable")]

        f1_hz, f2_hz = f1_all[positions], f2_all[positions]
        phi1, phi2 = phi1_all[positions], phi2_all[positions]
        rng1, rng2 = rng1_all[positions], rng2_all[positions]
        frequency_valid = np.isfinite(f1_hz) & np.isfinite(f2_hz) & (f1_hz != f2_hz)
        carrier_valid = np.isfinite(phi1) & np.isfinite(phi2) & frequency_valid
        pseudo_valid = np.isfinite(rng1) & np.isfinite(rng2) & frequency_valid

        carrier_raw = np.full(len(positions), np.nan)
        pseudo_raw = np.full(len(positions), np.nan)
        carrier_raw[carrier_valid] = carrier_phase_tec(
            phi1[carrier_valid], phi2[carrier_valid], f1_hz[carrier_valid], f2_hz[carrier_valid]
        )
        pseudo_raw[pseudo_valid] = pseudorange_tec(
            rng1[pseudo_valid], rng2[pseudo_valid], f1_hz[pseudo_valid], f2_hz[pseudo_valid]
        )

        new_segment = np.zeros(len(positions), dtype=bool)
        if time_all is not None and len(positions) > 1:
            time = time_all[positions]
            new_segment[1:] = (np.diff(time) > gap_ns) | (time[1:] == nat)
            # A carrier outage longer than max_gap also starts a new segment.
            last_valid = np.maximum.accumulate(
                np.where(carrier_valid, np.arange(len(positions)), -1)
            )
            previous_valid = np.r_[-1, last_valid[:-1]]
            has_previous = previous_valid >= 0
            new_segment[has_previous] |= carrier_valid[has_previous] & (
                time[has_previous] - time[previous_valid[has_previous]] > gap_ns
            )
            new_segment[0] = False

        boundaries = np.r_[0, np.flatnonzero(new_segment), len(positions)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            repaired = _repair_tec_pair(
                np.column_stack([carrier_raw[start:stop], pseudo_raw[start:stop]]), fs=fs
            )
            carrier, pseudo = repaired["carrier"], repaired["pseudo"]
            common = carrier.notna() & pseudo.notna()
            if common.any():
                carrier = carrier + (pseudo[common].median() - carrier[common].median())
            else:
                carrier[:] = np.nan
            carrier_output[positions[start:stop]] = carrier.to_numpy()
            pseudo_output[positions[start:stop]] = pseudo.to_numpy()

    df[f"tec_cph{pair}"] = carrier_output
    df[f"tec_rng{pair}"] = pseudo_output
    return df
