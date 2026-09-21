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

try:  # Support package imports and direct execution from this directory.
    from .fast_common_signal import receiver_epoch_ids_fast
except ImportError:  # pragma: no cover - direct-script fallback
    from fast_common_signal import receiver_epoch_ids_fast


WEEK_NS = 7 * 24 * 60 * 60 * 1_000_000_000
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


def _unwrap_gps_weeks(epoch_times: np.ndarray) -> tuple[np.ndarray, int]:
    rollover = np.diff(epoch_times) < -(WEEK_NS // 2)
    rollover_number = np.r_[0, np.cumsum(rollover, dtype="int64")]
    unwrapped = epoch_times + rollover_number * WEEK_NS
    if np.any(np.diff(unwrapped) < 0):
        where = np.flatnonzero(np.diff(unwrapped) < 0)[:5].tolist()
        raise ValueError(
            "timestamps still move backward after GPS-week unwrapping at "
            f"receiver epochs {where}"
        )
    return unwrapped, int(rollover.sum())


def _assign_regular_grid(
    epoch_times: np.ndarray,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """Select the closest strictly increasing sample-grid slot per epoch."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    period_float = 1_000_000_000 / sample_rate_hz
    period_ns = int(round(period_float))
    if not np.isclose(period_ns, period_float):
        raise ValueError("sample rate must have an integer-nanosecond period")

    # v326 float32 time-of-week values can be 31.25 ms from the true time.
    tolerance_ns = max(int(round(0.625 * period_ns)) + 1_000, period_ns // 2)
    lower = np.floor_divide(epoch_times, period_ns)
    candidates = np.column_stack([lower, lower + 1]).astype("int64")
    residuals = np.abs(epoch_times[:, None] - candidates * np.int64(period_ns))
    valid = residuals <= tolerance_ns
    if np.any(~valid.any(axis=1)):
        bad = np.flatnonzero(~valid.any(axis=1))[:5].tolist()
        raise ValueError(f"no feasible {sample_rate_hz:g} Hz grid slot at {bad}")

    count = len(epoch_times)
    unreachable = np.iinfo("int64").max
    back = np.full((count, 2), -1, dtype="int8")
    missing_previous = np.full(2, unreachable, dtype="int64")
    error_previous = np.full(2, np.inf)
    for state in range(2):
        if valid[0, state]:
            missing_previous[state] = 0
            error_previous[state] = float(residuals[0, state]) ** 2

    for epoch in range(1, count):
        missing_current = np.full(2, unreachable, dtype="int64")
        error_current = np.full(2, np.inf)
        for state in range(2):
            if not valid[epoch, state]:
                continue
            for previous_state in range(2):
                if missing_previous[previous_state] == unreachable:
                    continue
                step = candidates[epoch, state] - candidates[epoch - 1, previous_state]
                if step < 1:
                    continue
                proposed = (
                    missing_previous[previous_state] + step - 1,
                    error_previous[previous_state]
                    + float(residuals[epoch, state]) ** 2,
                )
                current = (missing_current[state], error_current[state])
                if proposed < current:
                    missing_current[state], error_current[state] = proposed
                    back[epoch, state] = previous_state
        if np.all(missing_current == unreachable):
            raise ValueError(
                f"no strictly increasing sample-grid path at receiver epoch {epoch:,}"
            )
        missing_previous = missing_current
        error_previous = error_current

    final_state = min(
        (state for state in range(2) if missing_previous[state] != unreachable),
        key=lambda state: (missing_previous[state], error_previous[state]),
    )
    states = np.empty(count, dtype="int8")
    states[-1] = final_state
    for epoch in range(count - 1, 0, -1):
        states[epoch - 1] = back[epoch, states[epoch]]
    slots = candidates[np.arange(count), states]
    return slots * np.int64(period_ns)


def _assign_sample_order_grid(
    epoch_times: np.ndarray,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    """Return a strictly increasing grid when raw timestamps are ambiguous.

    Each observed epoch is first rounded to the nearest 20 Hz grid slot.  Row
    order is authoritative: when an observed slot repeats or moves backward,
    it is advanced to one slot after the preceding epoch.  Forward timestamp
    jumps are retained, so trustworthy coarse gaps and minute boundaries still
    anchor the internal clock.  This fallback is used only when the stricter
    bounded-residual grid solver cannot find a path.
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    period_float = 1_000_000_000 / sample_rate_hz
    period_ns = int(round(period_float))
    if not np.isclose(period_ns, period_float):
        raise ValueError("sample rate must have an integer-nanosecond period")

    observed_slots = np.rint(epoch_times / period_ns).astype("int64")
    epoch_number = np.arange(len(observed_slots), dtype="int64")
    # A strictly increasing integer sequence has nondecreasing (slot - index).
    # Cumulative maximum is the vectorized form of max(observed, previous + 1).
    slots = np.maximum.accumulate(observed_slots - epoch_number) + epoch_number
    if np.any(np.diff(slots) < 1):
        raise AssertionError("sample-order grid construction was not increasing")
    return slots * np.int64(period_ns)


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

    epoch_ids = receiver_epoch_ids_fast(
        frame,
        time_column=time_column,
        satellite_columns=satellite_columns,
    )
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

    unwrapped, rollover_count = _unwrap_gps_weeks(epoch_times[keep_epoch])
    sample_order_fallback = False
    try:
        fixed_epoch_times = _assign_regular_grid(
            unwrapped,
            sample_rate_hz=sample_rate_hz,
        )
    except ValueError as error:
        if not (
            str(error).startswith("no feasible ")
            or str(error).startswith("no strictly increasing sample-grid path")
        ):
            raise
        fixed_epoch_times = _assign_sample_order_grid(
            unwrapped,
            sample_rate_hz=sample_rate_hz,
        )
        sample_order_fallback = True

    fixed_by_original_epoch = np.full(len(starts), NAT_INT, dtype="int64")
    fixed_by_original_epoch[keep_epoch] = fixed_epoch_times
    new_epoch_by_original = np.full(len(starts), -1, dtype="int32")
    new_epoch_by_original[keep_epoch] = np.arange(len(fixed_epoch_times), dtype="int32")

    repaired = frame.loc[keep_row].copy().reset_index(drop=True)
    kept_original_epoch = epoch_ids[keep_row]
    repaired[time_column] = pd.to_datetime(fixed_by_original_epoch[kept_original_epoch])
    repaired["_receiver_epoch"] = new_epoch_by_original[kept_original_epoch]

    period_ns = int(round(1_000_000_000 / sample_rate_hz))
    steps = np.diff(fixed_epoch_times) // period_ns
    report = ReceiverClockReport(
        input_rows=len(frame),
        output_rows=len(repaired),
        receiver_epochs_before_deduplication=len(starts),
        receiver_epochs_after_deduplication=len(fixed_epoch_times),
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
