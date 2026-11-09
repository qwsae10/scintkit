"""Reconstruct the ScintPi receiver clock without rewriting source data.

ScintPi v326 stores GPS time-of-week as ``float32``.  Near the end of a GPS
week, adjacent 20 Hz epochs can therefore share a timestamp, and a file that
crosses the week boundary can jump backward by seven days.  The functions in
this module recover one ordered receiver epoch clock from the original row
order.  Missing receiver epochs remain missing; only an in-memory timestamp
column is produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from scintkit.timing import WEEK_NS, grid_times, receiver_epochs, unwrap_weeks

NAT_INT = np.datetime64("NaT", "ns").astype("int64")


@dataclass(frozen=True)
class ReceiverClockReport:
    """Summary of the in-memory receiver-clock reconstruction."""

    input_rows: int
    output_rows: int
    receiver_epochs_before_deduplication: int
    receiver_epochs_after_deduplication: int
    exact_duplicate_receiver_epochs_removed: int
    exact_duplicate_rows_removed: int
    gps_week_rollovers_unwrapped: int
    sample_order_grid_fallback_used: bool
    missing_receiver_epochs: int
    receiver_gap_count: int
    first_timestamp: str
    last_timestamp: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _epoch_bounds(epoch_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(np.diff(epoch_ids)) + 1].astype("int64")
    ends = np.r_[starts[1:], len(epoch_ids)].astype("int64")
    return starts, ends


def _find_exact_duplicate_epochs(
    frame: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    epoch_times: np.ndarray,
    *,
    time_column: str,
) -> np.ndarray:
    """Mark later copies of adjacent, exactly identical receiver epochs."""

    payload_columns = [column for column in frame.columns if column != time_column]
    row_hashes = pd.util.hash_pandas_object(
        frame[payload_columns], index=False
    ).to_numpy(dtype="uint64")
    duplicate = np.zeros(len(starts), dtype=bool)

    for epoch in range(1, len(starts)):
        previous = epoch - 1
        if epoch_times[epoch] != epoch_times[previous]:
            continue
        if ends[epoch] - starts[epoch] != ends[previous] - starts[previous]:
            continue
        previous_hashes = row_hashes[starts[previous] : ends[previous]]
        current_hashes = row_hashes[starts[epoch] : ends[epoch]]
        if np.array_equal(np.sort(previous_hashes), np.sort(current_hashes)):
            duplicate[epoch] = True
    return duplicate


def reconstruct_receiver_clock(
    frame: pd.DataFrame,
    *,
    sample_rate_hz: float = 20.0,
    time_column: str = "datetime",
    satellite_columns: tuple[str, str] = ("cons", "svid"),
) -> tuple[pd.DataFrame, ReceiverClockReport]:
    """Return a row-order-preserving frame with a valid in-memory clock.

    The returned frame contains ``_receiver_epoch``.  Exact duplicate receiver
    epochs are removed, but the input dataframe and source file are untouched.
    """

    if frame.empty:
        raise ValueError("cannot reconstruct the clock of an empty dataframe")
    raw_time = pd.to_datetime(frame[time_column], errors="coerce")
    raw_ns = raw_time.to_numpy(dtype="datetime64[ns]").astype("int64")
    if np.any(raw_ns == NAT_INT):
        raise ValueError(f"{time_column!r} contains invalid timestamps")

    if time_column != "datetime" or tuple(satellite_columns) != ("cons", "svid"):
        raise ValueError("only datetime/cons/svid columns are supported")
    epoch_ids = receiver_epochs(frame.reset_index(drop=True))
    starts, ends = _epoch_bounds(epoch_ids)
    epoch_times = raw_ns[starts]
    duplicate_epoch = _find_exact_duplicate_epochs(
        frame,
        starts,
        ends,
        epoch_times,
        time_column=time_column,
    )
    keep_epoch = ~duplicate_epoch
    keep_row = keep_epoch[epoch_ids]

    kept_times = epoch_times[keep_epoch]
    rollover_count = int(np.sum(np.diff(kept_times) < -(WEEK_NS // 2)))
    unwrapped = unwrap_weeks(kept_times)
    fixed_epoch_times, sample_order_fallback = grid_times(unwrapped, sample_rate_hz)

    fixed_by_original_epoch = np.full(len(starts), NAT_INT, dtype="int64")
    fixed_by_original_epoch[keep_epoch] = fixed_epoch_times
    new_epoch_by_original = np.full(len(starts), -1, dtype="int32")
    new_epoch_by_original[keep_epoch] = np.arange(
        len(fixed_epoch_times), dtype="int32"
    )

    repaired = frame.loc[keep_row].copy().reset_index(drop=True)
    kept_original_epoch = epoch_ids[keep_row]
    repaired[time_column] = pd.to_datetime(
        fixed_by_original_epoch[kept_original_epoch]
    )
    repaired["_receiver_epoch"] = new_epoch_by_original[kept_original_epoch]

    period_ns = int(round(1_000_000_000 / sample_rate_hz))
    steps = np.diff(fixed_epoch_times) // period_ns
    report = ReceiverClockReport(
        input_rows=int(len(frame)),
        output_rows=int(len(repaired)),
        receiver_epochs_before_deduplication=int(len(starts)),
        receiver_epochs_after_deduplication=int(len(fixed_epoch_times)),
        exact_duplicate_receiver_epochs_removed=int(duplicate_epoch.sum()),
        exact_duplicate_rows_removed=int((~keep_row).sum()),
        gps_week_rollovers_unwrapped=rollover_count,
        sample_order_grid_fallback_used=sample_order_fallback,
        missing_receiver_epochs=int(np.maximum(steps - 1, 0).sum()),
        receiver_gap_count=int(np.count_nonzero(steps > 1)),
        first_timestamp=str(pd.Timestamp(fixed_epoch_times[0])),
        last_timestamp=str(pd.Timestamp(fixed_epoch_times[-1])),
    )
    return repaired, report
