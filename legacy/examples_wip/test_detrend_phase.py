
#%%
from cmath import phase
from datetime import datetime
import glob
from itertools import permutations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal




import numpy as np
import pandas as pd
from scipy import signal



def add_detrended_phase_exact(
    df,
    phase_col="cph1",
    out_col="detrended_cph1",
    tr=1,
    fs=20,
    f_N=0.1,
):
    """
    mimic the original script as closely as possible

    input: raw mosaic-style dataframe with columns like
        cons, svid, week, towe, elev, azim, snr1, cph1, cph2

    output:
        same dataframe with new column `out_col`
    """

    gnssdic = {0: "GPS", 1: "SBS", 2: "GAL", 3: "BDS", 6: "GLO"}

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

    def repair_discontinuities(vec, threshold=10):
        first_order = pd.DataFrame({"delt": vec.diff()})
        first_order_full = first_order["delt"]

        points = first_order.resample("10s").median().shift(freq="5s")

        if len(points) == 1:
            defa = vec.copy()
            return defa, ~(defa == defa)

        final_t = points.index[-1] + (points.index[-1] - points.index[-2])
        final_val = points.iloc[-1] + (points.iloc[-1] - points.iloc[-2])

        first_t = points.index[0] - (points.index[1] - points.index[0])
        first_val = points.iloc[0] - (points.iloc[1] - points.iloc[0])

        points.loc[final_t] = final_val
        points.loc[first_t] = first_val
        points.sort_index(inplace=True)

        doppler = points.delt.reindex_like(first_order_full).interpolate("time")
        cycleslips = np.abs(doppler - first_order_full) <= threshold

        first_order_full = first_order_full.where(cycleslips, np.nan)

        if len(first_order_full) > 1:
            first_order_full.iloc[0] = first_order_full.iloc[1]

        first_order_full = first_order_full.bfill()
        result = np.cumsum(first_order_full)

        return pd.Series(result, index=first_order_full.index), ~(cycleslips)

    def filter_signal_cascaded(x, f_N=0.1, fs=10):
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

    def detrend_phase_like_original(sig, tr=10):
        sig = sig.dropna(subset=["SNR", "Elevation", "Azimuth", "Phase"])

        snr = 10 ** (sig["SNR"] / 10)
        el = sig["Elevation"]
        az = sig["Azimuth"]
        phase = sig["Phase"]

        sig_repaired, cycleslips = repair_discontinuities(phase * 2 * np.pi, threshold=tr)
        sig_filtered = filter_signal_cascaded(sig_repaired, f_N=f_N, fs=fs)

        out = pd.DataFrame(
            {
                "dphi": sig_filtered,
                "cycleslips": cycleslips.astype(int),
                "SNR": snr,
                "az": az.reindex(sig_repaired.index),
                "el": el.reindex(sig_repaired.index),
            },
            index=sig_repaired.index,
        )
        return out

    # make a working copy
    out = df.copy()
    out["_row_id"] = np.arange(len(out))

    # build datetime exactly like original worker
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
    elif isinstance(out.index, pd.DatetimeIndex):
        out["datetime"] = out.index
    else:
        gps_epoch = pd.Timestamp("1980-01-06")
        out["datetime"] = (
            gps_epoch
            + pd.to_timedelta(out["week"] * 7, unit="D")
            + pd.to_timedelta(out["towe"], unit="s")
        )

    # original code maps cons to strings, renames, then builds SVID string
    work = out.copy()

    if np.issubdtype(work["cons"].dtype, np.number):
        work["cons"] = work["cons"].map(gnssdic)

    work = work.rename(
        columns={
            "elev": "Elevation",
            "azim": "Azimuth",
            phase_col: "Phase",
            "cons": "SIG",
            "svid": "SVID",
            "snr1": "SNR",
        }
    )

    work["SVID_num"] = work["SVID"]
    work["SVID"] = make_prn_local(work)

    # original code removes this satellite
    work = work[work["SVID"] != "R255"].copy()

    # preserve original rows for final merge
    pieces = []

    for sat, g in work.groupby("SVID", sort=False):
        g = g.sort_values("datetime").copy()

        # original code sets datetime as index before groupby apply
        # duplicate datetimes break reindex/interpolate logic, so keep first only
        g_unique = g.drop_duplicates(subset=["datetime"], keep="first").copy()
        g_unique = g_unique.set_index("datetime")

        try:
            detr = detrend_phase_like_original(g_unique, tr=tr)
            detr = detr[["dphi"]].rename(columns={"dphi": out_col}).reset_index()
        except Exception:
            detr = pd.DataFrame(
                {
                    "datetime": g_unique.index,
                    out_col: np.nan,
                }
            )

        # map back to every original row with that timestamp
        g_back = g[["_row_id", "datetime"]].merge(detr, on="datetime", how="left")
        pieces.append(g_back[["_row_id", out_col]])

    if pieces:
        detr_all = pd.concat(pieces, ignore_index=True)
        out = out.merge(detr_all, on="_row_id", how="left")
    else:

        out[out_col] = np.nan

    out = out.sort_values("_row_id").drop(columns="_row_id")
    return out
    # Example usage as a script.




files = glob.glob('/Users/isaac/Documents/ScintPi/Mothersday/Data/*scintpi3*')
ismr_files = glob.glob('/Users/isaac/Documents/ScintPi/Mothersday/Data/*CSS*')


df = pd.read_parquet(files[0])
#%%
#gnssdic_loop = {0: 'GPS', 1: 'SBS', 2: 'GAL', 3: 'BDS', 6: 'GLO'}
#df['cons'] = df['cons'].map(gnssdic_loop)
df = df[~((df['cons'] == 'GLO') & (df['svid'] == 255))].copy()

df=df[df['elev']>30]



gps_epoch = datetime(1980, 1, 6)
if 'week' in df.columns and 'towe' in df.columns:
    df['datetime'] = gps_epoch + pd.to_timedelta(df['week'].astype(int), unit='W') \
                        + pd.to_timedelta(df['towe'].astype(float), unit='s')
    df.drop(columns=['week', 'towe'], inplace=True, errors='ignore')

#%%

detr=add_detrended_phase_exact(df, phase_col="cph1", out_col="detrended_cph1", tr=1,fs=10)
detr=add_detrended_phase_exact(detr, phase_col="cph2", out_col="detrended_cph2", tr=1,fs=10)



#%%
ismr_tot=[]
for i in ismr_files:
    df_ismr = pd.read_parquet(i)
    ismr_tot.append(df_ismr)
ismr_tot=pd.concat(ismr_tot,ignore_index=True)

#%%


detr['med1'] = detr.groupby('datetime')['detrended_cph1'].transform('median')

# %%
def make_prn(df, cons_col='cons', svid_col='svid'):
    # map constellations to prn prefix
    prefix_map = {
        'GPS': 'G',
        'GAL': 'E',
        'GLO': 'R',
        'BDS': 'C',
        'SBAS': 'S',

    }
    prn = (
        df[cons_col].map(prefix_map) +
        df[svid_col].astype(int).astype(str).str.zfill(2)
    )
    return prn

detr['PRN'] = make_prn(detr)
detr['minute']= detr['datetime'].dt.floor('T')

detr['detrended_cph1_noclock']= detr['detrended_cph1'] - detr['med1']
#%%


import numpy as np
sigphi_df = (
    detr
    .groupby(['minute', 'PRN'])['detrended_cph1_noclock']
    .agg(
        sigphi_1=lambda x: np.nanstd(x),
        n_samples='count'
    )
    .reset_index()
)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# choose which ismr column to compare against
ismr_col = 'Phi60_1'
ismr_tot['Phi60_1'] = pd.to_numeric(ismr_tot['Phi60_1'], errors='coerce')
# optional: make sure minute is datetime
sigphi_df['minute'] = pd.to_datetime(sigphi_df['minute'])

# if ismr_tot has more than one row per PRN, reduce it first
ismr_use = (
    ismr_tot[['PRN', ismr_col,'Elevation']]
    .dropna()
    .groupby('PRN', as_index=False)
    .mean()
)

# merge by PRN
df = sigphi_df.merge(ismr_use, on='PRN', how='inner')

# remove bad values and keep plotting range
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['sigphi_1', ismr_col])
df = df[(df['sigphi_1'] >= 0) & (df[ismr_col] >= 0)]
df_plot = df[(df['Elevation'] >= 30) 
             & (df['minute'] >= df['minute'].min()+pd.Timedelta(minutes=5)) 
             & (df['minute'] <= df['minute'].max()-pd.Timedelta(minutes=5))
             & (df['n_samples'] >= 590)  # optional: only keep points with enough samples]

# all-satellite scatter
plt.figure(figsize=(6, 6))
plt.scatter(df_plot['sigphi_1'], df_plot[ismr_col], s=12)
plt.plot([0, 2], [0, 2], 'k--')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('ScintPi $\sigma_\phi$')
plt.ylabel(f'ISMR {ismr_col}')
plt.title('All satellites')
plt.tight_layout()
plt.show()
#%%

plt.plot(detr['datetime'], detr['med1']);plt.ylim(-2,3)
