"""Sampling rate and receiver epochs recovered from damaged float32 time-of-week stamps.

The receiver stores time-of-week as float32: late in the GPS week adjacent
20 Hz epochs share one value, and a file crossing the week boundary jumps back
7 days. Rows are written epoch by epoch, so row order still separates epochs.
"""

import logging

import numpy as np
import pandas as pd

from scintkit.constants import MAX_CLOCK_CORRECTION_S, MIN_MODAL_FRACTION

logger = logging.getLogger(__name__)

WEEK_NS = 7 * 86400 * 10**9


def detect_sampling_rate(df):
    """Sampling rate (Hz): the modal number of samples per satellite-second.

    Raises ValueError when the frame is empty or no count dominates.
    """
    counts = df.groupby([df["datetime"].dt.floor("1s"), "cons", "svid"], observed=True).size()
    if counts.empty:
        raise ValueError("cannot determine the sampling rate of an empty frame")
    share = counts.value_counts(normalize=True)
    if share.iloc[0] < MIN_MODAL_FRACTION:
        raise ValueError(f"no dominant samples-per-second count: {share.head(3).to_dict()}")
    return float(share.index[0])


def receiver_epochs(df):
    """Receiver epoch number (0, 1, ...) of every row, from row order and raw ``datetime``.

    A new epoch starts when the time changes, or when a satellite reappears
    under the same time (two epochs collapsed onto one float32 value).
    """
    keys = [df["datetime"], df["cons"], df["svid"]]
    occurrence = df.groupby(keys, sort=False, observed=True).cumcount()
    level = occurrence.groupby(df["datetime"], sort=False).cummax().to_numpy()
    time = df["datetime"].to_numpy()
    boundary = np.r_[True, (time[1:] != time[:-1]) | (level[1:] > level[:-1])]
    return np.cumsum(boundary) - 1


def unwrap_weeks(times):
    """Undo the 7-day jump back at a GPS week boundary (ns); ValueError if time still runs backwards."""
    times = times + np.r_[0, np.cumsum(np.diff(times) < -WEEK_NS // 2)] * WEEK_NS
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps run backwards")
    return times


def _grid_slots(times, period):
    """Strictly increasing grid slots (multiples of ``period`` ns) nearest ``times`` (ns).

    Each epoch takes the grid slot just below or above its time; dynamic
    programming picks the path with the fewest skipped slots, then the
    smallest squared residual.
    """
    lower = times // period
    cand = np.stack([lower, lower + 1], axis=1)
    resid = np.abs(times[:, None] - cand * period).astype(float)
    ok = resid <= max(0.625 * period + 1000, period / 2)  # float32 error is up to 31.25 ms
    if not ok.any(axis=1).all():
        raise ValueError("timestamps too far from the sampling grid")

    cand_l, err_l, ok_l = cand.tolist(), (resid**2).tolist(), ok.tolist()
    none = (np.inf, np.inf)
    cost = [(0, err_l[0][s]) if ok_l[0][s] else none for s in (0, 1)]
    back = np.zeros((len(times), 2), dtype=np.int8)
    for e in range(1, len(times)):
        new = [none, none]
        for s in (0, 1):
            if not ok_l[e][s]:
                continue
            for p in (0, 1):
                step = cand_l[e][s] - cand_l[e - 1][p]
                if step >= 1 and cost[p] != none:
                    c = (cost[p][0] + step - 1, cost[p][1] + err_l[e][s])
                    if c < new[s]:
                        new[s], back[e, s] = c, p
        if new == [none, none]:
            raise ValueError(f"no increasing grid path at epoch {e}")
        cost = new

    slots = np.empty(len(times), dtype=np.int64)
    s = 0 if cost[0] <= cost[1] else 1
    for e in range(len(times) - 1, -1, -1):
        slots[e] = cand_l[e][s]
        s = back[e, s]
    return slots


def grid_times(times, fs):
    """``(grid times, fallback used)``: strictly increasing times (ns) on the 1/fs grid nearest ``times`` (ns).

    When no path stays within the float32 error, row order wins: each epoch
    takes its nearest slot or, if that is taken, the slot after the previous epoch.
    """
    period = round(1e9 / fs)
    if period * fs != 1e9:
        raise ValueError(f"{fs:g} Hz has no whole-nanosecond period")
    try:
        return _grid_slots(times, period) * period, False
    except ValueError:
        slots = np.rint(times / period).astype(np.int64)
        k = np.arange(len(slots))
        return (np.maximum.accumulate(slots - k) + k) * period, True


def fix_epochs(df, fs):
    """Add ``epoch`` (receiver epoch number) and move ``datetime`` onto the 1/fs grid, one time per epoch.

    ``df`` must be in receiver row order. If the reconstruction cannot be
    trusted the raw times are kept and a warning is logged.
    """
    df = df.reset_index(drop=True)
    epoch = receiver_epochs(df)
    df["epoch"] = epoch
    starts = np.r_[0, np.flatnonzero(np.diff(epoch)) + 1]
    raw = df["datetime"].to_numpy("datetime64[ns]").astype(np.int64)[starts]
    try:
        unwrapped = unwrap_weeks(raw)
        fixed, fallback = grid_times(unwrapped, fs)
        worst = np.abs(fixed - unwrapped).max() / 1e9
        if worst > MAX_CLOCK_CORRECTION_S:
            raise ValueError(f"an epoch would move by {worst:.2f} s")
    except ValueError as error:
        logger.warning("keeping raw timestamps, epoch reconstruction failed: %s", error)
        return df
    logger.info("%d receiver epochs, %d shared a timestamp%s", len(raw), len(raw) - len(np.unique(raw)),
                "; placed by row order" if fallback else "")
    df["datetime"] = pd.to_datetime(fixed[epoch])
    return df
