"""Quick-look plots of lvl2/lvl3 products."""

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

MIN_VALID_S4 = 300  # satellites with fewer valid s4_1 rows are not plotted


def individual_summary_plot(df, timecol='datetime', elmask=10):
    """One panel per satellite: TEC and elevation on top, S4 and sigma-phi (flag 0, above ``elmask`` deg) below.

    Returns ``(fig, axes)``; the caller shows or saves the figure.
    """
    df = df.sort_values(timecol)
    valid = df.groupby("prn")["s4_1"].apply(lambda x: x.notna().sum())
    prns = valid[valid > MIN_VALID_S4].index.tolist()
    ncols = int(np.ceil(np.sqrt(len(prns))))
    nrows = int(np.ceil(len(prns) / ncols))

    fig = plt.figure(figsize=(4 * ncols, 5 * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.3, hspace=0.35)
    axes = []
    for k, prn in enumerate(prns):
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, hspace=0.0)
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)
        g = df[df["prn"] == prn].copy()

        ax1.plot(g[timecol], g["tec_cph12"], '.', color="black", lw=0.8)
        ax1.set_ylabel("TEC")
        ax1.tick_params(labelbottom=False)
        ax1.annotate(f"PRN {prn} ", xy=(0.02, 0.9), xycoords="axes fraction",
                     fontsize=9, ha="left", va="top")

        ax1b = ax1.twinx()
        ax1b.plot(g[timecol], g["elev"], color="green", lw=1, alpha=0.7)
        ax1b.set_ylabel("Elev (deg)", color="green")
        ax1b.tick_params(axis="y", colors="green")
        ax1b.set_ylim(0, 90)

        above_mask = g['elev'] > elmask
        for product in ("sigma_phi", "s4"):
            for i in ("1", "2"):
                good = (g[f"{product}_quality_flag_{i}"] == 0) & above_mask
                g.loc[~good, f"{product}_{i}"] = np.nan

        ax1.grid(True, which='both', ls='--', alpha=0.5)
        ax2.grid(True, which='both', ls='--', alpha=0.5)

        l1, = ax2.plot(g[timecol], g["s4_1"], color="blue", lw=0.8)
        l2, = ax2.plot(g[timecol], g["s4_2"], color="blue", ls="--", lw=0.8)
        l3, = ax2.plot(g[timecol], g["sigma_phi_1"], color="red", lw=0.8)
        l4, = ax2.plot(g[timecol], g["sigma_phi_2"], color="red", ls="--", lw=0.8)
        ax2.set_ylabel(r"$S4 / \sigma_{\phi}$")
        ax2.set_xlabel("UT")
        ax2.set_ylim(0, 1)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        if k == 0:
            ax1.legend([l1, l2, l3, l4], ["S4 L1", "S4 L2", "σφ L1", "σφ L2"],
                       fontsize=8, loc="upper right")
        axes.append((ax1, ax2))

    day = df[timecol].dt.floor("D").median().strftime("%Y-%m-%d")
    fig.suptitle(f"ScintPi Data - {day}", fontsize=12, y=0.95)
    fig.tight_layout()
    return fig, axes
