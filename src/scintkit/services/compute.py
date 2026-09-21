# %%
import numpy as np
import pandas as pd

from scintkit.preprocessing.format import temp_formating
from scintkit.services.phase_detrend import detect_sampling_rate, process_phases


def carrier_phase_tec(phi1_cyc, phi2_cyc, f1_hz, f2_hz):
    c = 299792458  # m/s

    lambda1 = c / f1_hz
    lambda2 = c / f2_hz

    phi1_m = phi1_cyc * lambda1
    phi2_m = phi2_cyc * lambda2

    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (phi1_m - phi2_m) / 1e16


def pseudorange_tec(P1_m, P2_m, f1_hz, f2_hz):
    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (P2_m - P1_m) / 1e16


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
        deltas.rolling(window=window, center=True, min_periods=max(3, window // 10))
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
                increments = (
                    clean_deltas.iloc[start + 1 : stop]
                    .interpolate(limit_direction="both")
                    .fillna(0.0)
                )
                result.iloc[start + 1 : stop] = (
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
    f1_all = (
        pd.to_numeric(df[f"freq_{number_1}"], errors="coerce").to_numpy(dtype=float)
        * 1e6
    )
    f2_all = (
        pd.to_numeric(df[f"freq_{number_2}"], errors="coerce").to_numpy(dtype=float)
        * 1e6
    )
    if "datetime" in df.columns:
        time_all = (
            pd.to_datetime(df["datetime"], errors="coerce")
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
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
            carrier_gap[has_previous] = carrier_valid[has_previous] & (
                time[has_previous] - time[previous_valid_index[has_previous]] > gap_ns
            )
            new_segment |= carrier_gap
            new_segment[0] = False

        boundaries = np.r_[0, np.flatnonzero(new_segment), len(positions)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            repaired = _repair_tec_pair(
                np.column_stack([carrier_raw[start:stop], pseudo_raw[start:stop]]),
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


def compute_s4(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    return std / mean if mean > 0 else np.nan


MIN_TAU_SAMPLES = 1000
MIN_UNIQUE_SNR = 2


def compute_tau(snr, fs):
    snr = snr.dropna()

    if len(snr) < MIN_TAU_SAMPLES or snr.nunique() < MIN_UNIQUE_SNR:
        return np.nan

    amp = snr.to_numpy(dtype=float)

    # remove mean
    amp = amp - np.nanmean(amp)

    # autocorrelation
    # ac = np.correlate(amp, amp, mode="full")
    ac = np.correlate(amp, amp, mode="full")
    ac = ac[len(ac) // 2 :]

    peak_idx = np.argmax(ac)
    peak = ac[peak_idx]

    half = peak * 0.5

    mask = np.where(ac > half, ac, np.nan)

    left = np.nanargmax(mask)
    right = left + np.nanargmin(mask[left:])

    dt = 1.0 / fs

    return (right - peak_idx) * dt


def compute_s4_corrected(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    if mean <= 0:
        return np.nan

    s4 = std / mean
    s4_correction = np.sqrt(100 / mean * (1 + 500 / (19 * mean)))
    val = s4**2 - s4_correction**2
    return np.sqrt(val) if val > 0 else 0


def compute_n_cycleslips(cycleslips):
    return int(cycleslips.fillna(False).sum())


def compute_n_samples(col):
    return int(col.notna().sum())


def compute_sigma_phi(phase):
    phase = phase.dropna()
    return np.std(phase) if len(phase) > 0 else np.nan


SIGMA_PHI_MAX_DROPPED_SAMPLES = 10
S4_MIN_SAMPLE_FRACTION = 0.8


def _add_quality_flags(products, fs):
    """Add binary, per-frequency quality flags to minute products."""
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "A positive sampling rate is required to compute quality flags."
        )

    expected_samples = fs * 60
    sigma_phi_min_samples = expected_samples - SIGMA_PHI_MAX_DROPPED_SAMPLES
    s4_min_samples = expected_samples * S4_MIN_SAMPLE_FRACTION
    is_glonass = products["prn"].astype(str).str.startswith("R")

    internal_columns = []
    for i in ("1", "2", "3"):
        phase_count_col = f"n_sigphi_{i}"
        edge_gap_col = f"_sigma_phi_edge_gap_{i}"
        s4_count_col = f"n_s4_{i}"

        if phase_count_col in products.columns:
            if edge_gap_col in products.columns:
                has_edge_gap = products[edge_gap_col].astype(bool)
            else:
                # The phase product is not trustworthy if its edge/gap mask
                # was not propagated from phase detrending. Treat a missing
                # per-frequency mask as bad instead of silently assuming that
                # the channel contains no edge or gap contamination.
                has_edge_gap = pd.Series(True, index=products.index)

            sigma_phi_bad = (
                has_edge_gap
                | products[phase_count_col].lt(sigma_phi_min_samples)
                | is_glonass
            )
            products[f"sigma_phi_quality_flag_{i}"] = sigma_phi_bad.astype(np.int8)

        if s4_count_col in products.columns:
            products[f"s4_quality_flag_{i}"] = (
                products[s4_count_col].lt(s4_min_samples).astype(np.int8)
            )

        internal_columns.append(edge_gap_col)

    return products.drop(columns=internal_columns, errors="ignore")


def add_products(df, verbose=False, fs=None, merge=True):
    """
    This function takes a full-rate dataframe (fs=20 or 10 Hz) at and computes various products:
    - tec12 and tec13: differences between detrended phases to estimate TEC (WIP)
    - sigma_phi_1, sigma_phi_2, sigma_phi_3: standard deviation of detrended phases with clock noise removed, for each frequency
    - n_sigphi_1, n_sigphi_2, n_sigphi_3: number of valid detrended phase samples used for sigma-phi
    - n_s4_1, n_s4_2, n_s4_3: number of valid SNR samples used for S4
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - sigma_phi_quality_flag_1/2/3: binary sigma-phi quality flags; 0 is good and 1 marks an edge/gap, too many dropped samples, or GLONASS
    - s4_quality_flag_1/2/3: binary S4 quality flags; 0 is good and 1 marks fewer than 80% of the expected samples
    - s4_1, s4_2, s4_3: S4 index computed from SNR values for each frequency
    - s4_corrected_1, s4_corrected_2, s4_corrected_3: S4 index corrected for bias based on Van Dierendonck (1993) method
    The function groups the data by PRN and 1-minute bins to compute these products, and then merges the results back to the original dataframe in the same time bins (if merge=True).
    """

    if verbose:
        print("Ensuring format...")
    df = temp_formating(df)
    if verbose:
        print("Processing phases...")
    df = process_phases(df)

    if fs is None:
        fs = detect_sampling_rate(df)

    if fs is None:
        raise ValueError("Could not determine sampling rate.")

    if verbose:
        print("Computing TEC...")

    if "cph1" in df.columns and "cph2" in df.columns:
        df = add_tec_columns(df, fs=fs, pair="12")
    if "cph1" in df.columns and "cph3" in df.columns:
        df = add_tec_columns(df, fs=fs, pair="13")

    if verbose:
        print("Computing products...")

    group_cols = ["prn", "minbin"]
    agg_dict = {}
    for i in ("1", "2", "3"):
        detrended_noclk_col = f"detrended_noclk_cph{i}"
        cycleslip_col = f"cycleslips_cph{i}"
        edgegap_col = f"edgegap_mask_cph{i}"
        snr_col = f"snr{i}"

        if detrended_noclk_col in df.columns:
            agg_dict[f"sigma_phi_{i}"] = (detrended_noclk_col, compute_sigma_phi)
            agg_dict[f"n_sigphi_{i}"] = (
                detrended_noclk_col,
                compute_n_samples,
            )

        if cycleslip_col in df.columns:
            agg_dict[f"n_cycleslip_{i}"] = (cycleslip_col, compute_n_cycleslips)

        # Preserve the edge/gap result until it can be combined with the
        # sample-count and constellation checks below.
        if edgegap_col in df.columns:
            agg_dict[f"_sigma_phi_edge_gap_{i}"] = (
                edgegap_col,
                lambda x: int(x.fillna(False).astype(bool).any()),
            )

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
            agg_dict[f"tau_{i}"] = (snr_col, lambda x, fs=fs: compute_tau(x, fs))
            agg_dict[f"n_s4_{i}"] = (
                snr_col,
                compute_n_samples,
            )

    if not agg_dict:
        return df if merge else (df, None)

    products = df.groupby(group_cols, sort=False).agg(**agg_dict).reset_index()
    products = _add_quality_flags(products, fs=fs)

    if merge:
        if verbose:
            print("Merging products back to original dataframe...")
        df = df.merge(products, on=group_cols, how="left")
        return df

    return df, products


# %%
