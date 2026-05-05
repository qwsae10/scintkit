#%%%
import numpy as np
import pandas as pd
from scipy import signal    

def detect_sampling_rate(df):
    """
    detect the sampling rate of the data by looking at the number of samples per minute per PRN.
    """
    # samples per (minute, prn)
    counts = (
        df
        .groupby(['minbin', 'prn'])
        .size()
        .reset_index(name='n_samples')
    )


    n = counts.n_samples.max()

    threshold = 10
    if abs(n-600) < threshold:
        return 600/60
    elif abs(n-1200) < threshold:
        return 1200/60
    elif abs(n-2400) < threshold:
        return 2400/60
    elif abs(n-3000) < threshold:
        return 3000/60
    else:
        return None

def make_prn_local(dfin):
    constellation_map = {
        "GPS": "G",
        "BDS": "C",
        "GAL": "E",
        "GLO": "R",
        "QZSS": "J",
        "IRNSS": "I",
        "SBAS": "S",
        "SBS": "S",
    }
    return dfin["SIG"].map(constellation_map) + dfin["SVID"].astype(int).astype(str).str.zfill(2)


def repair_discontinuities_pos(vec, fs, threshold=1,svid=None,verbose=False):
    y = pd.Series(vec).copy()

    window = int(10 * fs)

    delt = y.diff()

    trend = (
        delt
        .rolling(window=window, center=True, min_periods=max(3, window // 10))
        .median()
        .bfill()
        .ffill()
    )

    good = ((delt - trend).abs() <= threshold).fillna(True)
    slip_mask = ~good
    n_slips = int(slip_mask.sum())
 
    if n_slips/len(vec) > 0.2 and len(vec) > 10:
        if verbose:
            print(f"many cycle slips detected for SVID {svid}, {n_slips}/{len(vec)}.")
    
        return pd.Series(vec), slip_mask, n_slips
    delt_clean = delt.where(good, np.nan)

    if len(delt_clean) > 1:
        delt_clean.iloc[0] = delt_clean.iloc[1]

    delt_clean = delt_clean.interpolate(limit_direction="both")

    result = y.iloc[0] + delt_clean.cumsum()


    return pd.Series(result, index=y.index), slip_mask, n_slips

def filter_signal_cascaded(x, f_N=0.1, fs=10):
        # To do: design a non-causal filter 
        omega_N = 2 * np.pi * f_N
        l = 3

        a1 = np.sqrt(2 + np.sqrt(l))
        a2 = np.sqrt(2)
        a3 = np.sqrt(2 - np.sqrt(l))

        num = [1, 0, 0]
        den1 = [1, a1 * omega_N, omega_N**2]
        den2 = [1, a2 * omega_N, omega_N**2]
        den3 = [1, a3 * omega_N, omega_N**2]

        bz1, az1 = signal.bilinear(num, den1, fs)
        bz2, az2 = signal.bilinear(num, den2, fs)
        bz3, az3 = signal.bilinear(num, den3, fs)

        y_stage1 = signal.lfilter(bz1, az1, x)
        y_stage2 = signal.lfilter(bz2, az2, y_stage1)
        y_stage3 = signal.lfilter(bz3, az3, y_stage2)

        return y_stage3




def highpass_phase(
    df,
    in_col="cph1",
    out_col="detrended_cph1",
    tr=1,
    fs=None,
    f_N=0.1
):

    slip_col = out_col.replace("detrended", "cycleslips")
    df[out_col] = np.nan
    df[slip_col] = False
    mask_col = out_col.replace("detrended", "edgegap_mask")
    df[mask_col] = False


    if fs is None:
        fs = detect_sampling_rate(df)

    if fs is None or fs <= 0:
        return df

    phase_all = df[in_col].to_numpy(dtype=float)

    for prn, pos in df.groupby("prn", sort=False).indices.items():
        phase = phase_all[pos]

        try:
            finite = np.isfinite(phase)
            if not finite.any():
                continue

            phase_s = pd.Series(phase)
            phase_s = phase_s.interpolate(limit_direction="both")


            repaired, cycleslips, _ = repair_discontinuities_pos(
                phase_s * 2 * np.pi,
                fs=fs,
                threshold=tr,
                svid=prn
            )

            slip_mask = cycleslips.to_numpy(dtype=bool)

            filtered = filter_signal_cascaded(
                repaired.to_numpy(dtype=float),
                f_N=f_N,
                fs=fs,
            )
            time_vec = df.iloc[pos]["datetime"].to_numpy()

            edge_gap_mask = make_edge_gap_mask(
                time_vec,
                phase,
                fs=fs,
                gap_seconds=2,
                pad_seconds=180,
            )

            filtered = np.asarray(filtered, dtype=float)
            filtered[~finite] = np.nan
            slip_mask[~finite] = False

            df.iloc[pos, df.columns.get_loc(out_col)] = filtered
            df.iloc[pos, df.columns.get_loc(slip_col)] = slip_mask
            df.iloc[pos, df.columns.get_loc(mask_col)] = edge_gap_mask
            

        except Exception as exc:
            print(f"highpass_phase failed for PRN {prn}: {type(exc).__name__}: {exc}")
            df.iloc[pos, df.columns.get_loc(out_col)] = np.nan
            df.iloc[pos, df.columns.get_loc(slip_col)] = False
            df.iloc[pos, df.columns.get_loc(mask_col)] = False
            raise

    return df



def make_edge_gap_mask(time, phase, fs, gap_seconds=1, pad_seconds=180):
    time = pd.to_datetime(time)
    phase = np.asarray(phase, dtype=float)

    n = len(phase)
    mask = np.zeros(n, dtype=bool)

    if n == 0:
        return mask

    pad = int(round(fs * pad_seconds))

    dt = pd.Series(time).diff().dt.total_seconds().to_numpy()
    finite = np.isfinite(phase)

    break_idx = np.where((dt > gap_seconds) | (~finite))[0]

    points = [0, n - 1]
    points.extend(break_idx.tolist())

    for p in points:
        i0 = max(0, p - pad)
        i1 = min(n, p + pad + 1)
        mask[i0:i1] = True

    return mask

def highpass_all_phases(df,fs=None):
    #wrapper to add detrended phases for all available phase columns
    for i in range(1, 4):
        if f"cph{i}" not in df.columns:
            continue
        in_col = f"cph{i}"
        out_col = f"detrended_cph{i}"
        frequency_col = f"freq_{i}"

        scaled_tr = 1*df[frequency_col].median()/1575.42 if frequency_col in df.columns else 1

        df = highpass_phase(df, in_col=in_col, out_col=out_col, tr=scaled_tr, fs=fs)
    return df


def estimate_clock(df, elev_mask=0):

    value_cols = []

    for i in [1, 2, 3]:
        cph_col = f"detrended_cph{i}"
        freq_col = f"freq_{i}"
        v_col = f"v{i}"

        if cph_col in df.columns and freq_col in df.columns:
            df[v_col] = df[cph_col] / df[freq_col]
            value_cols.append(v_col)

    if not value_cols:
        df["clock_term"] = np.nan
        return df

    # only use high-elevation data to estimate clock
    clock_df = df[(df["elev"] > elev_mask) & (df["elev"] < 90)]

    median_curve = (
        clock_df.melt(
            id_vars="datetime",
            value_vars=value_cols
        )
        .groupby("datetime")["value"]
        .median()
    )

    # apply clock estimate back to full df
    df["clock_term"] = df["datetime"].map(median_curve)

    return df

def clock_correction(df,out_col="detrended_noclk_cph"):

    for i in [1, 2, 3]:
        cph_col = f"detrended_cph{i}"
        out_coli = out_col+f"{i}"
        
        if cph_col in df.columns and "clock_term" in df.columns and f"freq_{i}" in df.columns:
            df[out_coli] = df[cph_col] - df["clock_term"] * df[f"freq_{i}"]

    return df


def process_phases(df,fs=None):
    fs=detect_sampling_rate(df) if fs is None else fs

    df = highpass_all_phases(df,fs)
    df = estimate_clock(df)
    df = clock_correction(df)

    return df

# %%
