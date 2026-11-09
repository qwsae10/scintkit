import numpy as np
import pandas as pd


def add_snr_columnsJM(df,fs=None,i=1, max_gap="5min", *, copy=True):
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
    expected_columns = [
        "prn",
        f"snr{i}"
    ]
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required SNR columns: {missing}")

    if not isinstance(df.index, pd.RangeIndex) or not df.index.equals(
        pd.RangeIndex(len(df))
    ):
        df = df.reset_index(drop=True)
        copy = False
    elif copy:
        df = df.copy()
    gap_threshold = pd.to_timedelta(max_gap)
    gap_ns = gap_threshold.value
    ##############
    for prn, groupi in df.groupby('prn', sort=False, observed=True):
        # Get the indices corresponding to this PRN
        idx = groupi.index
    
        # Make sure data are ordered by datetime
        temp = groupi.sort_values('datetime').copy()
    
        temp[f"snr{i}_avg"] = (
            temp.set_index('datetime')[f"snr{i}"]
                .rolling("10s", center=True)
                .mean()
                .to_numpy()
        )
    
        temp[f"snr{i}_new"] = (temp[f"snr{i}"] - temp[f"snr{i}_avg"] + 40)

        # Put the calculated values back into the original dataframe
        df.loc[temp.index, f"snr{i}_avg"] = temp[f"snr{i}_avg"]
        df.loc[temp.index, f"snr{i}_new"] = temp[f"snr{i}_new"]

    return df

def compute_s4(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    min_snr = np.min(snr)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    return std / mean if ((mean > 0)&(min_snr!=0)) else np.nan

def compute_s4_from_detrended_snr(snr):
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

    if (
        len(snr) < MIN_TAU_SAMPLES
        or snr.nunique() < MIN_UNIQUE_SNR
    ):
        return np.nan

    amp = snr.to_numpy(dtype=float)

    # remove mean
    amp = amp - np.nanmean(amp)

    # autocorrelation
    #ac = np.correlate(amp, amp, mode="full")
    ac = np.correlate(amp, amp, mode="full")
    ac = ac[len(ac)//2:]

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


