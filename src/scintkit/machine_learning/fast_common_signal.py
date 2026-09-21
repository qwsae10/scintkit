"""Fast common-signal median for ScintPi files with overlapping timestamps.

This deliberately does less than ``repair_20hz_timestamps.py``.  It does not
rewrite timestamps or a Parquet file.  It labels the receiver epochs already
encoded by row order and uses that integer label for the common-signal median.

The input must still be in its original receiver/file row order.  Calling
``temp_formating`` is fine because it preserves that order.  Do not sort by
the damaged datetime column before assigning receiver epoch IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def receiver_epoch_ids_fast(
    frame: pd.DataFrame,
    *,
    time_column: str = "datetime",
    satellite_columns: tuple[str, str] = ("cons", "svid"),
) -> np.ndarray:
    """Return one integer receiver-epoch ID per row, without fixing time.

    A new receiver epoch starts whenever the raw time changes.  If float32
    quantization gives consecutive epochs the same timestamp, the occurrence
    number of persistent satellite keys rises from 0 to 1 (or 1 to 2).  The
    cumulative maximum is important: a satellite newly appearing in the later
    epoch can have occurrence 0 and must not create an additional boundary.

    This vectorized result was checked against the full sequential epoch
    reconstruction on the supplied 10.3-million-row file.
    """

    required = [time_column, *satellite_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if frame.empty:
        return np.empty(0, dtype="int32")
    if frame[time_column].isna().any():
        raise ValueError(f"{time_column!r} contains NaT")

    occurrence = (
        frame.groupby(
            [time_column, *satellite_columns],
            sort=False,
            observed=True,
        )
        .cumcount()
        .to_numpy(dtype="int16")
    )
    occurrence_level = (
        pd.Series(occurrence, index=frame.index)
        .groupby(frame[time_column], sort=False, observed=True)
        .cummax()
        .to_numpy(dtype="int16")
    )

    raw_time = frame[time_column].to_numpy(dtype="datetime64[ns]").astype("int64")
    boundary = np.empty(len(frame), dtype=bool)
    boundary[0] = True
    boundary[1:] = (raw_time[1:] != raw_time[:-1]) | (
        occurrence_level[1:] > occurrence_level[:-1]
    )
    return np.cumsum(boundary, dtype="int32") - 1


def common_median_by_epoch(
    frame: pd.DataFrame,
    value_columns: list[str] | tuple[str, ...],
    *,
    epoch_ids: np.ndarray | None = None,
    elevation_column: str = "elev",
    elevation_mask_degrees: float = 10,
) -> pd.Series:
    """Compute the median across satellites/signals for every receiver epoch."""

    value_columns = list(value_columns)
    if not value_columns:
        return pd.Series(dtype=float, name="clock_term")
    missing = [
        column
        for column in [elevation_column, *value_columns]
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    if epoch_ids is None:
        epoch_ids = receiver_epoch_ids_fast(frame)
    epoch_ids = np.asarray(epoch_ids)
    if len(epoch_ids) != len(frame):
        raise ValueError("epoch_ids must contain one value per dataframe row")

    elevation = pd.to_numeric(frame[elevation_column], errors="coerce").to_numpy()
    selected = (elevation > elevation_mask_degrees) & (elevation < 90)
    selected_epoch_ids = epoch_ids[selected]
    values = frame.loc[selected, value_columns].to_numpy(dtype=float)

    long_epoch_ids = np.repeat(selected_epoch_ids, len(value_columns))
    long_values = values.ravel()
    finite = np.isfinite(long_values)
    curve = (
        pd.Series(long_values[finite], index=long_epoch_ids[finite], name="clock_term")
        .groupby(level=0, sort=False)
        .median()
    )
    curve.index.name = "receiver_epoch"
    return curve


def estimate_clock_fast(
    frame: pd.DataFrame,
    *,
    elevation_mask_degrees: float = 10,
    keep_epoch_id: bool = False,
) -> pd.DataFrame:
    """Drop-in replacement for ScintKit's datetime-grouped ``estimate_clock``.

    It creates the same ``v1``/``v2``/``v3`` and ``clock_term`` columns, but
    groups by receiver epoch rather than the damaged datetime values.
    """

    value_columns: list[str] = []
    for signal_number in (1, 2, 3):
        phase_column = f"detrended_cph{signal_number}"
        frequency_column = f"freq_{signal_number}"
        value_column = f"v{signal_number}"
        if phase_column in frame.columns and frequency_column in frame.columns:
            frame[value_column] = frame[phase_column] / frame[frequency_column]
            value_columns.append(value_column)

    if not value_columns:
        frame["clock_term"] = np.nan
        return frame

    epoch_ids = receiver_epoch_ids_fast(frame)
    curve = common_median_by_epoch(
        frame,
        value_columns,
        epoch_ids=epoch_ids,
        elevation_mask_degrees=elevation_mask_degrees,
    )
    frame["clock_term"] = curve.reindex(epoch_ids).to_numpy()
    if keep_epoch_id:
        frame["receiver_epoch"] = epoch_ids
    return frame
