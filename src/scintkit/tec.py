import numpy as np
import pandas as pd


def carrier_phase_tec(phi1_cyc, phi2_cyc, f1_hz, f2_hz):
    c = 299792458  # m/s

    lambda1 = c / f1_hz
    lambda2 = c / f2_hz

    phi1_m = phi1_cyc * lambda1 
    phi2_m = phi2_cyc * lambda2 

    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (phi1_m - phi2_m)/1e16


def pseudorange_tec(P1_m, P2_m, f1_hz, f2_hz):
    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (P2_m - P1_m)/1e16

def _repair_tec_pair(values, fs, threshold=1):
    """Repair carrier and pseudorange TEC with one paired rolling median.

    This is equivalent to calling ``repair_discontinuities_pos`` once for each
    column, but Pandas evaluates the shared two-column rolling operation in one
    pass.  The reconstruction remains independent for each TEC series.
    """
    values = pd.DataFrame(values, columns=["carrier", "pseudo"], dtype=float)
    repaired = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    finite = values.notna().to_numpy()
    if not finite.any():
        return repaired

    window = int(10 * fs)
    deltas = values.diff()
    trends = (
        deltas
        .rolling(window=window, center=True, min_periods=max(3, window // 10))
        .median()
        .bfill()
        .ffill()
    )
    residuals = (deltas - trends).abs()

    for column_number, column in enumerate(values.columns):
        column_finite = finite[:, column_number]
        if not column_finite.any():
            continue
        good = residuals[column].le(threshold) | residuals[column].isna()
        slip_count = int((~good).sum())
        if slip_count / len(values) > 0.2 and len(values) > 10:
            repaired[column] = values[column]
            continue

        clean_deltas = deltas[column].where(good, np.nan)
        padded = np.r_[False, column_finite, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        result = repaired[column]
        source = values[column]
        for start, stop in zip(starts, stops):
            result.iloc[start] = source.iloc[start]
            if start + 1 < stop:
                increments = clean_deltas.iloc[start + 1:stop].interpolate(
                    limit_direction="both"
                ).fillna(0.0)
                result.iloc[start + 1:stop] = (
                    source.iloc[start] + increments.cumsum()
                ).to_numpy()
    return repaired

def add_tec_columns(df, pair="13", fs=None, max_gap="5min", *, copy=True):
    """Add carrier-phase and pseudorange TEC for a frequency pair.
    Each PRN is split into continuous time segments. A time gap strictly
    greater than ``max_gap`` starts a new segment. Carrier and pseudorange TEC
    are repaired independently within each segment, then the carrier TEC is
    shifted so that its segment median matches the pseudorange TEC median at
    common valid epochs. Carrier TEC is left missing when a segment has no
    pseudorange overlap and therefore cannot be leveled.
    fs=None, max_gap="5min", *, copy=True
    """
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive sampling rate")
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("pair must contain two different signal numbers")

    expected_columns = [
        "prn",
        f"cph{pair[0]}",
        f"cph{pair[1]}",
        f"rng{pair[0]}",
        f"rng{pair[1]}",
        f"freq_{pair[0]}",
        f"freq_{pair[1]}",
    ]
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required TEC columns: {missing}")

    if not isinstance(df.index, pd.RangeIndex) or not df.index.equals(
        pd.RangeIndex(len(df))
    ):
        df = df.reset_index(drop=True)
        copy = False
    elif copy:
        df = df.copy()
    gap_threshold = pd.to_timedelta(max_gap)
    gap_ns = gap_threshold.value
    number_1, number_2 = pair

    phi1_all = pd.to_numeric(df[f"cph{number_1}"], errors="coerce").to_numpy(
        dtype=float
    )
    phi2_all = pd.to_numeric(df[f"cph{number_2}"], errors="coerce").to_numpy(
        dtype=float
    )
    rng1_all = pd.to_numeric(df[f"rng{number_1}"], errors="coerce").to_numpy(
        dtype=float
    )
    rng2_all = pd.to_numeric(df[f"rng{number_2}"], errors="coerce").to_numpy(
        dtype=float
    )
    rng1_all[rng1_all == 0] = np.nan
    rng2_all[rng2_all == 0] = np.nan
    f1_all = pd.to_numeric(df[f"freq_{number_1}"], errors="coerce").to_numpy(
        dtype=float
    ) * 1e6
    f2_all = pd.to_numeric(df[f"freq_{number_2}"], errors="coerce").to_numpy(
        dtype=float
    ) * 1e6
    if "datetime" in df.columns:
        time_all = pd.to_datetime(df["datetime"], errors="coerce").to_numpy(
            dtype="datetime64[ns]"
        ).astype("int64")
        nat = np.datetime64("NaT", "ns").astype("int64")
    else:
        time_all = None
        nat = None

    carrier_output = np.full(len(df), np.nan)
    pseudo_output = np.full(len(df), np.nan)
    groups = df.groupby("prn", sort=False, observed=True).indices
    for key, group_positions in groups.items():
        positions = np.asarray(group_positions, dtype="int64")
        if time_all is not None:
            order = np.argsort(time_all[positions], kind="stable")
            positions = positions[order]
            time = time_all[positions]
        else:
            time = None

        phi1 = phi1_all[positions]
        phi2 = phi2_all[positions]
        rng1 = rng1_all[positions]
        rng2 = rng2_all[positions]
        f1_hz = f1_all[positions]
        f2_hz = f2_all[positions]
        frequency_valid = np.isfinite(f1_hz) & np.isfinite(f2_hz) & (f1_hz != f2_hz)
        carrier_valid = np.isfinite(phi1) & np.isfinite(phi2) & frequency_valid
        pseudo_valid = np.isfinite(rng1) & np.isfinite(rng2) & frequency_valid

        carrier_raw = np.full(len(positions), np.nan)
        pseudo_raw = np.full(len(positions), np.nan)
        carrier_raw[carrier_valid] = carrier_phase_tec(
            phi1[carrier_valid],
            phi2[carrier_valid],
            f1_hz[carrier_valid],
            f2_hz[carrier_valid],
        )
        pseudo_raw[pseudo_valid] = pseudorange_tec(
            rng1[pseudo_valid],
            rng2[pseudo_valid],
            f1_hz[pseudo_valid],
            f2_hz[pseudo_valid],
        )

        new_segment = np.zeros(len(positions), dtype=bool)
        if time is not None and len(time) > 1:
            new_segment[1:] = (np.diff(time) > gap_ns) | (time[1:] == nat)
            last_valid_index = np.maximum.accumulate(
                np.where(carrier_valid, np.arange(len(positions)), -1)
            )
            previous_valid_index = np.r_[-1, last_valid_index[:-1]]
            has_previous = previous_valid_index >= 0
            carrier_gap = np.zeros(len(positions), dtype=bool)
            carrier_gap[has_previous] = (
                carrier_valid[has_previous]
                & (
                    time[has_previous]
                    - time[previous_valid_index[has_previous]]
                    > gap_ns
                )
            )
            new_segment |= carrier_gap
            new_segment[0] = False

        boundaries = np.r_[0, np.flatnonzero(new_segment), len(positions)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            repaired = _repair_tec_pair(
                np.column_stack(
                    [carrier_raw[start:stop], pseudo_raw[start:stop]]
                ),
                fs=fs,
                threshold=1,
            )
            carrier_segment = repaired["carrier"]
            pseudo_segment = repaired["pseudo"]
            common_valid = carrier_segment.notna() & pseudo_segment.notna()
            if common_valid.any():
                carrier_segment = carrier_segment + (
                    pseudo_segment.loc[common_valid].median()
                    - carrier_segment.loc[common_valid].median()
                )
            else:
                carrier_segment[:] = np.nan
            segment_positions = positions[start:stop]
            carrier_output[segment_positions] = carrier_segment.to_numpy()
            pseudo_output[segment_positions] = pseudo_segment.to_numpy()

    df[f"tec_cph{pair}"] = carrier_output
    df[f"tec_rng{pair}"] = pseudo_output
    return df
    
