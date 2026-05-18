
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
#hi



import numpy as np
import pandas as pd
from scipy import signal


import numpy as np
import pandas as pd
from scipy import signal




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



def add_detrended_phase_exact(
    df,
    phase_col="cph1",
    out_col="detrended_cph1",
    tr=1,
    fs=20,
    f_N=0.1,
):
  
    gnssdic = {0: "GPS", 1: "SBS", 2: "GAL", 3: "BDS", 6: "GLO"}
    def repair_discontinuities_pos(vec, fs, threshold=10):
        y = pd.Series(vec).copy()

        window = int(10 * fs) #10sec window for median trend estimation

        delt = y.diff()

        trend = (
            delt
            .rolling(window=window, center=True, min_periods=max(3, window // 10))
            .median()
        )

        trend = trend.bfill().ffill()

        cycleslips = (delt - trend).abs() <= threshold

        delt_clean = delt.where(cycleslips, np.nan)

        if len(delt_clean) > 1:
            delt_clean.iloc[0] = delt_clean.iloc[1]

        delt_clean = delt_clean.bfill().ffill()

        result = y.iloc[0] + delt_clean.cumsum()

        slip_mask = ~cycleslips
        n_slips = int(slip_mask.sum())

        return pd.Series(result, index=y.index), slip_mask, n_slips
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

        phase = sig["Phase"]

        sig_repaired, cycleslips, n_slips = repair_discontinuities_pos(phase * 2 * np.pi, fs=fs, threshold=tr)
        sig_filtered = filter_signal_cascaded(sig_repaired, f_N=f_N, fs=fs)

        out = pd.DataFrame(
            {
                "dphi": sig_filtered,
                "n_slips": n_slips,
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

file = '/Users/isaac/Documents/scintpi3_20240511_0400_1203575.6250W_432707.1250N_v325.pq'

df = pd.read_parquet(file)

gnssdic_loop = {0: 'GPS', 1: 'SBS', 2: 'GAL', 3: 'BDS', 6: 'GLO'}
df['cons'] = df['cons'].map(gnssdic_loop)
df = df[~((df['cons'] == 'GLO') & (df['svid'] == 255))].copy()

df=df.reset_index()
#%%
df['minbin'] = df['datetime'].dt.floor('1min')
df['prn']=df['cons'] + df['svid'].astype(str).str.zfill(2)


#%%
# df has columns: 'prn', 'minbin', 'datetime'
import pandas as pd

df['datetime'] = pd.to_datetime(df['datetime'])

# sort first
df = df.sort_values(['prn', 'datetime'])

# compute dt per PRN
df['dt'] = (
    df.groupby('prn')['datetime']
      .diff()
      .dt.total_seconds()
)

print(df[['prn','datetime','dt']].head())

#%%
# keep default unique row index here
# keep datetime as a normal column
detr=add_detrended_phase_exact(df, phase_col="cph1", out_col="detrended_cph1", tr=1)
detr=add_detrended_phase_exact(df, phase_col="cph2", out_col="detrended_cph2", tr=1)


#%%
# plot some detrended phases
phases = detr.set_index('datetime')
phases = phases[phases['elev'] > 20]
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

windows = [
    pd.Timestamp('2024-05-11 07:50:00'),
    pd.Timestamp('2024-05-11 06:43:41'),
]

rng = np.random.default_rng(3)

for i, (ax, t1) in enumerate(zip(axes, windows)):
    t2 = t1 + pd.Timedelta(minutes=1)

    subset = phases.loc[t1:t2]

    svids = subset['svid'].dropna().unique()
    if len(svids) == 0:
        continue

    k = min(3, len(svids))
    chosen = rng.choice(svids,  size=k, replace=False)

    sub = subset[subset['svid'].isin(chosen)]

    # plot individual satellites
    for (cons, svid), g in sub.groupby(['cons', 'svid']):
        ax.plot(
            g.index,
            g['detrended_cph1'],
            label=rf'$\delta \phi$ {cons} {int(svid):02d}'
        )

    # median curve (yellow dots)
    med = sub['detrended_cph1'].groupby(sub.index).median()
    ax.plot(
        med.index,
        med.values,
        color='black',
        markersize=1,
        label='Clock reference (median)'
    )

    ax.set_xlabel('Universal Time')
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 4)
    ax.set_xlim(t1, t2)

    ax.text(
        0.02, 0.95,
        '(a)' if i == 0 else '(b)',
        transform=ax.transAxes,
        ha='left',
        va='top',
        weight='bold',
    )

axes[0].set_ylabel(r'$\delta \phi$ (radians)')
axes[0].legend(loc='upper right', fontsize='small')
axes[1].legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.show()
# %%
