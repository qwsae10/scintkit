
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def individual_summary_plot(df,timecol='secbin',elmask=10,n_thresh=0.98):

    df = df.sort_values(timecol)

    valid = df.groupby("prn")["s4_1"].apply(lambda x: x.notna().sum())
    prns = valid[valid > 300].index.tolist()
    tot_n=np.nanmax(df['n_1'])


    n = len(prns)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(4*ncols, 5*nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.3, hspace=0.35)
    axes = []
    for i, prn in enumerate(prns):
        r = i // ncols
        c = i % ncols

        inner = outer[r, c].subgridspec(2, 1, hspace=0.0)

        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)

        g = df[df["prn"] == prn].copy()

        # top
        ax1.plot(g[timecol], g["tec_cph12"],'.', color="black", lw=0.8)
        ax1.set_ylabel("TEC")
        ax1.tick_params(labelbottom=False)

        # title inside plot
        ax1.annotate(
            f"PRN {prn} ",
            xy=(0.02, 0.9),
            xycoords="axes fraction",
            fontsize=9,
            ha="left",
            va="top"
        )
        #elevation on twin axis
        ax1b = ax1.twinx()
        ax1b.plot(g[timecol], g["elev"], color="green", lw=1, alpha=0.7)

        #optional formatting
        ax1b.set_ylabel("Elev (deg)", color="green")
        ax1b.tick_params(axis="y", colors="green")

        ax1b.set_ylim(0, 90)

        sp_mask1=(
            (g["quality_1"] ==0)&
            (g['elev']>elmask)&(g['elev']<91)&
            (g['n_1']/tot_n>n_thresh)
            )
        
        g.loc[~sp_mask1, 'sigma_phi_1'] = np.nan

        sp_mask2=(
            (g["quality_2"] ==0)&
            (g['elev']>elmask)&
            (g['elev']<91)&
            (g['n_2']/tot_n>n_thresh)
            )
            
        g.loc[~sp_mask2, 'sigma_phi_2'] = np.nan

        s4_mask1=(
            (g['elev']>elmask)&
            (g['elev']<91)&
            (g['n_1']/tot_n>n_thresh)
            )
        g.loc[~s4_mask1, 's4_1'] = np.nan

        s4_mask2=(
            (g['elev']>elmask)&
            (g['elev']<91)&
            (g['n_2']/tot_n>n_thresh)
            )
        
        g.loc[~s4_mask2, 's4_2'] = np.nan

        ax1.grid(True, which='both', ls='--', alpha=0.5)
        ax2.grid(True, which='both', ls='--', alpha=0.5)
        
        # bottom

        l1, = ax2.plot(g[timecol], g["s4_1"], color="blue", lw=0.8)
        l2, = ax2.plot(g[timecol], g["s4_2"], color="blue", ls="--", lw=0.8)
        l3, = ax2.plot(g[timecol], g["sigma_phi_1"], color="red", lw=0.8)
        l4, = ax2.plot(g[timecol], g["sigma_phi_2"], color="red", ls="--", lw=0.8)

        ax2.set_ylabel("$S4 / \sigma_{\phi}$")
        ax2.set_xlabel("UT")
        ax2.set_ylim(0,1)

        
        # limit ticks
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))

        # format as UT time only
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # legend only once
        if i == 0:

            ax1.legend(
                [l1, l2, l3, l4],
                ["S4 L1", "S4 L2", "σφ L1", "σφ L2"],
                fontsize=8,
                loc="upper right"
            )
        axes.append((ax1, ax2))

    day=df[timecol].dt.floor("D").median().strftime("%Y-%m-%d")
    
    fig.suptitle(
        f"ScintPi Data - {day}",
        fontsize=12,
        y=0.95
        )
    
    fig.tight_layout()
    plt.show()
    return fig, axes

