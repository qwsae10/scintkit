import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd


def generate_availability_plot(plot_targets, cutoff, now, output_path):
    """Generates and saves the availability plot."""
    n = len(plot_targets)
    if n == 0:
        raise ValueError("No valid targets to plot.")

    fig_height = max(2.5, min(1.2 * n, 24))
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, fig_height), sharex=True, squeeze=False)
    axes = axes.ravel()

    COLOR_MISSING = '#ff9999'
    COLOR_PRESENT = '#66cc66'

    for ax, t in zip(axes, plot_targets):
        ax.axvspan(cutoff, now, color=COLOR_MISSING, alpha=1.0, lw=0, zorder=0)
        
        for d in t['valid_times']:
            end_d = min(d + pd.Timedelta(days=1), now)
            if d < now: 
                ax.axvspan(d, end_d, color=COLOR_PRESENT, alpha=1.0, lw=0, zorder=2)
                
        ax.set_xlim(cutoff, now)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=3)
            
        disp_code = t['code'] if t['code'] else "N/A"
        lat_val, lon_val = t['lat'], t['lon']
        lat_str = f"{abs(lat_val):.3f}°{'N' if lat_val >= 0 else 'S'}"
        lon_str = f"{abs(lon_val):.3f}°{'E' if lon_val >= 0 else 'W'}"
        
        display_label = f"[{t['version']}] {disp_code}, {t['name']} ({lat_str}, {lon_str})"
        ax.text(0.005, 0.8, display_label, transform=ax.transAxes, ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8), zorder=4)

    green_patch = mpatches.Patch(color=COLOR_PRESENT, label='Data present')
    red_patch   = mpatches.Patch(color=COLOR_MISSING, label='No data')
    fig.legend(handles=[green_patch, red_patch], loc='upper right', ncols=2)

    axes[-1].set_xlabel('Time')
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator())

    fig.suptitle(f"ScintPi Data Availability (last 3 months)\nGenerated {now:%Y-%m-%d}", fontsize=14, y=0.995)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {output_path}")
    plt.close(fig)