#%%
import pandas as pd
import numpy as np

from scintkit.services.phase_detrend import process_phases,repair_discontinuities_pos,detect_sampling_rate
from scintkit.preprocessing.format import temp_formating


import numpy as np
import pdb

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

def add_tec_columns(df, pair="13", fs=None):
    df = df.reset_index(drop=True).copy()



    def _per_prn(key, g):
        g=g.copy()
        N1 = pair[0]
        N2 = pair[1]

        phi1 = g[f"cph{N1}"]
        phi2 = g[f"cph{N2}"]
        rng1 = g[f"rng{N1}"]
        rng2 = g[f"rng{N2}"]
        # if carrier inputs invalid
        if phi1.isna().all() or phi2.isna().all():
            carrier = np.full(len(g), np.nan)
            n_slip_carrier = 0
        else:
            f1_hz = g[f"freq_{N1}"] * 1e6
            f2_hz = g[f"freq_{N2}"] * 1e6

            carrier = carrier_phase_tec(
                phi1_cyc=phi1,
                phi2_cyc=phi2,
                f1_hz=f1_hz,
                f2_hz=f2_hz,
            )

            carrier, _, n_slip_carrier = repair_discontinuities_pos(
                carrier, fs=fs, threshold=1, svid=key, verbose=True
            )

            carrier = carrier - np.nanmean(carrier)

        # if pseudorange inputs invalid
        if rng1.isna().all() or rng2.isna().all():
            pseudo = np.full(len(g), np.nan)
            n_slip_pseudo = 0
        else:
            f1_hz = g[f"freq_{N1}"] * 1e6
            f2_hz = g[f"freq_{N2}"] * 1e6

            pseudo = pseudorange_tec(
                P1_m=rng1,
                P2_m=rng2,
                f1_hz=f1_hz,
                f2_hz=f2_hz,
            )

            pseudo, _, n_slip_pseudo = repair_discontinuities_pos(
                pseudo, fs=fs, threshold=0.1, svid=key, verbose=False
            )



        g[f"tec_cph{pair}"] = carrier
        g[f"tec_rng{pair}"] = pseudo

        return g
    out = (
        pd.concat(
            [_per_prn(key, g) for key, g in df.groupby("prn", sort=False)], ignore_index=True
        )
    )

    return out

def compute_s4(snr):
    snr = snr.dropna()
    if len(snr) == 0:
        return np.nan

    lin_snr = 10 ** (snr / 10)
    mean = np.mean(lin_snr)
    std = np.std(lin_snr)

    return std / mean if mean > 0 else np.nan


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


def add_products(df,verbose=False):
    """
    This function takes a full-rate dataframe (fs=20 or 10 Hz) at and computes various products:
    - tec12 and tec13: differences between detrended phases to estimate TEC (WIP)
    - sigma_phi_1, sigma_phi_2, sigma_phi_3: standard deviation of detrended phases with clock noise removed, for each frequency
    - n_1, n_2, n_3: number of valid samples for each frequency
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - quality_1, quality_2, quality_3: binary flags indicating potential quality issues (0 means no issue, 1 or more means issue) 
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

    fs=detect_sampling_rate(df)
    
    if verbose:
        print("Computing TEC...")

    if f"cph1" in df.columns and f"cph2" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="12")
    if f"cph1" in df.columns and f"cph3" in df.columns:
        df=add_tec_columns(df,fs=fs, pair="13")

    
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
            agg_dict[f"n_{i}"] = (detrended_noclk_col, compute_n_samples)

        if cycleslip_col in df.columns:
            agg_dict[f"n_cycleslip_{i}"] = (cycleslip_col, compute_n_cycleslips)

        #if close to edge of pass, mark as potential quality issue
        if edgegap_col in df.columns:
            agg_dict[f"quality_{i}"] = (
                edgegap_col,
                lambda x: int(x.fillna(False).astype(bool).any())
            )

        if snr_col in df.columns:
            agg_dict[f"s4_{i}"] = (snr_col, compute_s4)
            agg_dict[f"s4_corrected_{i}"] = (snr_col, compute_s4_corrected)
    if not agg_dict:
        return df

    products = (
        df.groupby(group_cols, sort=False)
        .agg(**agg_dict)
        .reset_index()
    )
    if verbose:
        print("Merging products back to original dataframe...")
    df = df.merge(products, on=group_cols, how="left")

    return df

# %%
