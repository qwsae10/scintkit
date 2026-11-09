#%%
import numpy as np
import pandas as pd

from scintkit.amplitude import add_snr_columnsJM, compute_s4, compute_s4_corrected, compute_tau
from scintkit.formatting import temp_formating
from scintkit.phase import detect_sampling_rate, process_phases
from scintkit.tec import _repair_tec_pair, add_tec_columns

def detrended_snr(snr,mov_avg_seconds = 20):
    
    tec_factor = (f1_hz**2 * f2_hz**2) / (40.3 * (f1_hz**2 - f2_hz**2))

    return tec_factor * (P2_m - P1_m)/1e16


def add_dsnr_columns(df,i=1,fs=None, copy=True):
    dsnr_all = pd.to_numeric(df[f"snr{i}"], errors="coerce").to_numpy(dtype=float)
    df[f"dsnr{i}"] = dsnr_all
    return df

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
            products[f"sigma_phi_quality_flag_{i}"] = sigma_phi_bad.astype(
                np.int8
            )

        if s4_count_col in products.columns:
            products[f"s4_quality_flag_{i}"] = products[s4_count_col].lt(
                s4_min_samples
            ).astype(np.int8)

        internal_columns.append(edge_gap_col)

    return products.drop(columns=internal_columns, errors="ignore")


def add_products(df,verbose=False,fs=None):
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
    The function groups the data by PRN and 1-minute bins to compute these products, and then merges the results back to the original dataframe in the same time bins.
    """

    if verbose:
        print("Ensuring format...")
    df=temp_formating(df)
    if verbose:
        print("Processing phases...")   
    df = process_phases(df)
    
    if fs is None:
        #fs = 20
        fs = detect_sampling_rate(df)

    if fs is None:
        raise ValueError("Could not determine sampling rate.")
    
    if verbose:
        print("Computing TEC...")

    if f"cph1" in df.columns and f"cph2" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="12")
    if f"cph1" in df.columns and f"cph3" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="13")
    if verbose:
        print("Computing detrended SNR")
    for i in ['1','2','3']:
        if f"snr{i}" in df.columns:
            df=add_snr_columnsJM(df,i=i,fs=fs)
    if verbose:
        print("Computing products...")

    group_cols = ["prn", "minbin"]
    agg_dict = {}
    for i in ("1", "2", "3"):


        detrended_noclk_col = f"detrended_noclk_cph{i}"
        cycleslip_col = f"cycleslips_cph{i}"
        edgegap_col = f"edgegap_mask_cph{i}"
        snr_col = f"snr{i}"
        snr_new = f"snr{i}_new" #snr1_new

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
                lambda x: int(x.fillna(False).astype(bool).any())
            )

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_detrended_{i}"] = (snr_new, compute_s4) #this one shoyld be snr_dtrn and use the same func to comps4
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
            agg_dict[f"tau_{i}"] = (
                snr_col,
                lambda x, fs=fs: compute_tau(x, fs)
            )
            agg_dict[f"n_s4_{i}"] = (
                snr_col,
                compute_n_samples,
            )
           
    if not agg_dict:
        return df

    products = (
        df.groupby(group_cols, sort=False)
        .agg(**agg_dict)
        .reset_index()
    )
    products = _add_quality_flags(products, fs=fs)
    if verbose:
        print("Merging products back to original dataframe...")
    df = df.merge(products, on=group_cols, how="left")

    return df

# %%
