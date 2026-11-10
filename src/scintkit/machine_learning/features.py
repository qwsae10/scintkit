"""Per satellite-minute features of the signal classifier, and the ``scintkit-ml-features`` export.

The bundled model was trained on exactly these definitions at 20 Hz: band
power of C/N0 and TEC in five bands, statistics of the receiver-wide
common-mode C/N0 deviation, and S4 of both channels. Do not change them
without retraining.
"""

import argparse
import glob
import logging
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import signal

from scintkit.filenames import output_path
from scintkit.formatting import first_samples
from scintkit.products import compute_products
from scintkit.reading import read_lvl0, read_raw

logger = logging.getLogger(__name__)

FS_HZ = 20  # the features and the model assume 20 Hz data
MINUTE_NS = 60 * 10**9
MINUTE_SAMPLES = 60 * FS_HZ

# A satellite-minute gets features only when both channels have more than
# FEATURE_MIN_SAMPLES valid C/N0 samples and channel-1 S4 exceeds FEATURE_MIN_S4.
# The training set was built with the same thresholds.
FEATURE_MIN_SAMPLES = 1190
FEATURE_MIN_S4 = 0.15

# Common mode: per satellite pass, C/N0 minus its centred rolling median; then the
# median over satellites above the elevation mask at each receiver epoch.
COMMON_WINDOW = "5min"
COMMON_MIN_FRACTION = 0.5  # of the window's samples needed for a median
COMMON_PASS_GAP = "30min"  # a longer gap starts a new pass
COMMON_ELEVATION_MIN_DEG = 10
COMMON_MIN_SATELLITES = 4

# (label, low Hz, high Hz); the first positive bin of a 60 s FFT (1/60 Hz) is included.
BANDS = (
    ("0p017_0p05_hz", 1 / 60, 0.05),
    ("0p05_0p1_hz", 0.05, 0.1),
    ("0p1_0p3_hz", 0.1, 0.3),
    ("0p3_1_hz", 0.3, 1.0),
    ("1_10_hz", 1.0, 10.0),
)
COMMON_STATS = ("std", "median", "p95", "p99")

# Model input, in model order.
FEATURES = (
    [f"{p}_power_db_{label}" for p in ("snr1", "snr2", "tec12") for label, _, _ in BANDS]
    + [f"common_delta_snr{i}_{stat}_dbhz" for i in (1, 2) for stat in COMMON_STATS]
    + ["s4_1", "s4_2"]
)
COUNTS = ["n_samples", "n_snr2_samples", "n_tec12_samples", "n_common_delta_snr1", "n_common_delta_snr2"]


def band_powers_db(values, detrend):
    """Periodogram power (dB) integrated over BANDS; ``detrend`` is "median" (C/N0) or "linear" (TEC)."""
    values = np.asarray(values, dtype=float)
    prepared = values - np.median(values) if detrend == "median" else signal.detrend(values, type="linear")
    freqs, psd = signal.periodogram(prepared, fs=FS_HZ, window="boxcar", detrend=False, scaling="density")
    tolerance = np.finfo(float).eps * FS_HZ * 8  # band edges are shared by neighbouring bands
    powers = []
    for _, low, high in BANDS:
        selected = (freqs >= low - tolerance) & (freqs <= high + tolerance)
        power = float(np.trapezoid(psd[selected], freqs[selected])) if selected.sum() >= 2 else np.nan
        powers.append(10 * np.log10(power) if power > 0 else np.nan)
    return powers


def minute_grid(time_ns, values, minute_ns):
    """``(values on the 1200-slot grid with gaps filled linearly, valid count)``; None if too few or bad times."""
    period = 10**9 // FS_HZ
    slots = np.rint((time_ns - minute_ns) / period).astype(int)
    if np.any((slots < 0) | (slots >= MINUTE_SAMPLES)) or len(np.unique(slots)) != len(slots):
        return None, 0
    grid = np.full(MINUTE_SAMPLES, np.nan)
    grid[slots] = values
    valid = np.flatnonzero(np.isfinite(grid))
    if len(valid) <= FEATURE_MIN_SAMPLES:
        return None, len(valid)
    target = np.arange(MINUTE_SAMPLES)
    filled = np.interp(target, valid, grid[valid])
    # np.interp holds the edge values; extend the first and last segments linearly instead.
    (a, b), (c, d) = valid[:2], valid[-2:]
    filled[:a] = grid[a] + (grid[b] - grid[a]) / (b - a) * (target[:a] - a)
    filled[d + 1:] = grid[d] + (grid[d] - grid[c]) / (d - c) * (target[d + 1:] - d)
    return filled, len(valid)


def _common_stats(values):
    return pd.Series({
        "std": float(np.std(values.to_numpy(dtype=float), ddof=0)),
        "median": values.median(),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
        "n": len(values),
    })


def common_mode(time_ns, epoch, elev, snr, passes):
    """Per UTC minute (ns index): stats of the receiver-wide common C/N0 deviation (dB-Hz) of each channel.

    ``snr`` is (n, 2) float; ``passes`` lists the time-sorted row positions of each satellite pass.
    """
    min_periods = max(1, math.ceil(pd.Timedelta(COMMON_WINDOW).total_seconds() * FS_HZ * COMMON_MIN_FRACTION))
    delta = np.full(snr.shape, np.nan)
    for pos in passes:
        series = pd.DataFrame(snr[pos], index=pd.DatetimeIndex(time_ns[pos]))
        baseline = series.rolling(COMMON_WINDOW, center=True, min_periods=min_periods).median()
        delta[pos] = snr[pos] - baseline.to_numpy()

    elevation_ok = (elev > COMMON_ELEVATION_MIN_DEG) & (elev <= 90)
    out = []
    for i in (1, 2):
        names = {**{stat: f"common_delta_snr{i}_{stat}_dbhz" for stat in COMMON_STATS}, "n": f"n_common_delta_snr{i}"}
        valid = elevation_ok & np.isfinite(delta[:, i - 1])
        rows = pd.DataFrame({"epoch": epoch[valid], "delta": delta[valid, i - 1],
                             "minute": time_ns[valid] // MINUTE_NS * MINUTE_NS})
        per_epoch = rows.groupby("epoch", sort=False).agg(common=("delta", "median"), n=("delta", "count"),
                                                         minute=("minute", "first"))
        per_epoch = per_epoch[per_epoch["n"] >= COMMON_MIN_SATELLITES]
        if per_epoch.empty:
            out.append(pd.DataFrame(columns=list(names.values()), dtype=float))
            continue
        stats = per_epoch.groupby("minute", sort=True)["common"].apply(_common_stats).unstack()
        out.append(stats.rename(columns=names))
    return pd.concat(out, axis=1)


def _satellite_positions(frame, time_ns):
    """Row positions of every satellite, sorted by time."""
    return {prn: pos[np.argsort(time_ns[pos], kind="stable")]
            for prn, pos in frame.groupby("prn", sort=False, observed=True).indices.items()}


def compute_features(frame):
    """FEATURES and counts for every selected satellite-minute of a 20 Hz full-rate frame (after compute_products).

    Uses ``datetime``, ``prn``, ``epoch``, ``elev``, ``snr1``, ``snr2`` and
    ``tec_cph12``; one row per selected satellite-minute (``datetime`` = minute start).
    The frame is read, never copied or modified.
    """
    time_ns = frame["datetime"].to_numpy("datetime64[ns]").view(np.int64)
    snr = np.column_stack([frame["snr1"].to_numpy(dtype=float), frame["snr2"].to_numpy(dtype=float)])
    tec = frame["tec_cph12"].to_numpy(dtype=float)
    satellites = _satellite_positions(frame, time_ns)
    gap = pd.Timedelta(COMMON_PASS_GAP).value
    passes = [p for pos in satellites.values()
              for p in np.split(pos, np.flatnonzero(np.diff(time_ns[pos]) > gap) + 1) if len(p)]
    common = common_mode(time_ns, frame["epoch"].to_numpy(), frame["elev"].to_numpy(dtype=float), snr, passes)
    del passes

    records = []
    for prn, pos in satellites.items():
        minutes = time_ns[pos] // MINUTE_NS
        bounds = np.r_[0, np.flatnonzero(np.diff(minutes)) + 1, len(pos)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            rows = pos[start:stop]
            values = snr[rows]
            counts = np.isfinite(values).sum(axis=0)
            if counts.min() <= FEATURE_MIN_SAMPLES:
                continue
            linear = 10 ** (values / 10)  # float64 as in training; lvl3 s4_1 uses float32 C/N0
            s4 = np.nanstd(linear, axis=0) / np.nanmean(linear, axis=0)
            if not s4[0] > FEATURE_MIN_S4:
                continue
            minute_ns = int(minutes[start]) * MINUTE_NS
            snr1, n1 = minute_grid(time_ns[rows], values[:, 0], minute_ns)
            snr2, n2 = minute_grid(time_ns[rows], values[:, 1], minute_ns)
            if snr1 is None or snr2 is None:
                continue
            tec12, n_tec = minute_grid(time_ns[rows], tec[rows], minute_ns)
            tec_powers = band_powers_db(tec12, "linear") if tec12 is not None else [np.nan] * len(BANDS)
            records.append([prn, minute_ns, *band_powers_db(snr1, "median"), *band_powers_db(snr2, "median"),
                            *tec_powers, *s4, n1, n2, n_tec])

    spectral = [f for f in FEATURES if "power" in f]
    table = pd.DataFrame(records, columns=["prn", "minute_ns", *spectral, "s4_1", "s4_2", *COUNTS[:3]])
    table = table.join(common, on="minute_ns")
    table.insert(1, "datetime", pd.to_datetime(table.pop("minute_ns")))
    for column in COUNTS:
        table[column] = table[column].astype("Int64")
    return table[["prn", "datetime", *FEATURES, *COUNTS]].sort_values(["datetime", "prn"], kind="stable") \
        .reset_index(drop=True)


# --- Export for retraining ------------------------------------------------

KEY_COLUMNS = ["cons", "svid", "sig_1", "sig_2", "freq_1", "freq_2", "elev", "azim"]


def feature_table(path):
    """Feature table of one raw or lvl0 file: selected satellite-minutes with their lvl3 keys and ``source_file``.

    Any .pq/.parquet input is read as lvl0 (old ones without ``_lvl0`` are
    shifted to UTC). Raises ValueError when the data are not 20 Hz.
    """
    path = os.fspath(path)
    if path.lower().endswith((".pq", ".parquet")):
        df, _ = read_lvl0(path)
    else:
        df, _ = read_raw(path)
    frame, _, fs = compute_products(df)
    del df
    if fs != FS_HZ:
        raise ValueError(f"features need {FS_HZ} Hz data, got {fs:g} Hz")
    table = compute_features(frame)
    keys = first_samples(frame, "1min")[["prn", "datetime", *KEY_COLUMNS]]
    table = table.merge(keys, on=["prn", "datetime"], how="left")
    table["source_file"] = os.path.basename(path)
    return table


def _export(task):
    path, output_dir, overwrite = task
    name = os.path.basename(path)
    if name.lower().endswith((".pq", ".parquet")) and "_lvl0." not in name.lower():
        name = os.path.splitext(name)[0]  # old lvl0 names carry no _lvl0
    out = output_path(name, output_dir, "ml_features")
    if os.path.exists(out) and not overwrite:
        return path, "skipped"
    try:
        table = feature_table(path)
    except Exception as error:  # one bad file must not stop the others
        logger.exception("failed: %s", path)
        return path, f"failed: {type(error).__name__}: {error}"
    os.makedirs(output_dir, exist_ok=True)
    table.to_parquet(f"{out}.part", index=False)
    os.replace(f"{out}.part", out)
    logger.info("wrote %d rows to %s", len(table), out)
    return path, "ok"


def main(argv=None):
    """Entry point for ``scintkit-ml-features``: one ``<stem>_ml_features.pq`` per input file."""
    parser = argparse.ArgumentParser(prog="scintkit-ml-features", description=__doc__)
    parser.add_argument("files", nargs="+", help="raw or lvl0 files or globs (20 Hz)")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    files = [f for pattern in args.files for f in (sorted(glob.glob(pattern)) or [pattern])]
    tasks = [(f, args.output, args.overwrite) for f in files]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(_export, tasks))
    failed = [r for r in results if r[1].startswith("failed")]
    logger.info("%d files, %d failed", len(results), len(failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
