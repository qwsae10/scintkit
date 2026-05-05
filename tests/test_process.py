
#%%
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import scintkit

data_dir = Path("data")
raw_files = sorted(data_dir.glob("scintpi3_*lvl0.pq"))
#These can also be .bin or .bin.zip files

#if taking too long, specify subset
raw_files = raw_files[0]

if not raw_files:
    raise FileNotFoundError(
        f"No ScintPi files found in {data_dir.resolve()}. "
        "Start Jupyter from the repository root, or update data_dir."
    )

lvl3_files = scintkit.pipelines.auto.process(raw_files, verbose=True,mode='lvl2')

# %%
import matplotlib.pyplot as plt
file=lvl3_files[0]

df=pd.read_parquet(file)
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

df = df.sort_values("secbin")

valid = df.groupby("prn")["tec_cph12"].apply(lambda x: x.notna().sum())
prns = valid[valid > 300].index.tolist()

n = len(prns)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

fig = plt.figure(figsize=(4*ncols, 5*nrows))
outer = fig.add_gridspec(nrows, ncols, wspace=0.3, hspace=0.35)

for i, prn in enumerate(prns):
    r = i // ncols
    c = i % ncols

    inner = outer[r, c].subgridspec(2, 1, hspace=0.0)

    ax1 = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1], sharex=ax1)

    g = df[df["prn"] == prn]

    # top
    ax1.plot(g["datetime"], g["tec_cph12"],'.', color="black", lw=0.8)
    ax1.set_ylabel("TEC")
    ax1.tick_params(labelbottom=False)

    # title inside plot
    ax1.annotate(
        f"PRN {prn} frequencies {g['freq_1'].iloc[0]}, {g['freq_2'].iloc[0]}",
        xy=(0.02, 0.9),
        xycoords="axes fraction",
        fontsize=9,
        ha="left",
        va="top"
    )

    # bottom

    l1, = ax2.plot(g["datetime"], g["s4_1"], color="blue", lw=0.8)
    l2, = ax2.plot(g["datetime"], g["s4_2"], color="blue", ls="--", lw=0.8)
    l3, = ax2.plot(g["datetime"], g["sigma_phi_1"], color="red", lw=0.8)
    l4, = ax2.plot(g["datetime"], g["sigma_phi_2"], color="red", ls="--", lw=0.8)

    ax2.set_ylabel("S4 / σφ")
    ax2.set_xlabel("UT")
    ax2.set_ylim(0,1)

    
    # limit ticks
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # format as UT time only
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # legend only once
    if i == 0:
        ax1.legend(
            [l1, l2, l3, l4],
            ["S4 L1", "S4 L2", "σφ L1", "σφ L2"],
            fontsize=8,
            loc="upper right"
        )
        
day=df["datetime"].dt.floor("D").median().strftime("%Y-%m-%d")
fig.suptitle(
    f"ScintPi 3.0 Data - {day}",
    fontsize=12
    )
plt.show()
# %%
