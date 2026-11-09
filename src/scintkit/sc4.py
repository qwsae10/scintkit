"""ScintPi 4 (Septentrio mosaic-X5, SBF) raw files -> lvl0 Parquet.

Chain: SBF --(RXTools bin2asc --extractGenMeas)--> text --> one row per
satellite and epoch with channels 1-3, ``datetime`` in UTC, and elevation/
azimuth from the SatVisibility1 block when the receiver logged it.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scintkit import schema
from scintkit.constants import (
    CONSTELLATIONS,
    DEFAULT_LEAP_SECONDS,
    GLONASS_L1_MHZ,
    GLONASS_L1_STEP_MHZ,
    GLONASS_L2_MHZ,
    GLONASS_L2_STEP_MHZ,
    GPS_EPOCH,
    RINEX_PREFIX,
    RINEX_SVID_OFFSET,
)
from scintkit.filenames import output_path
from scintkit.stations import load_station_codes

logger = logging.getLogger(__name__)

# SBF file name: <4-char station prefix><day of year><hour a-x><minute>.<yy>_
SBF_NAME = re.compile(r"^([a-z]{2}[0-9]{2})([0-9]{3})([a-x])([0-9]{2})\.([0-9]{2})_$", re.IGNORECASE)

BIN2ASC_ENV = "SCINTKIT_BIN2ASC"  # command, e.g. "bin2asc" or "rxtools_exec.sh bin2asc"
# Blocks exported next to the measurements; missing ones are simply not written.
SBF_BLOCKS = ("SatVisibility1", "GPSUtc", "GLONav", "ReceiverSetup1")

WEEK_MS = 7 * 86400 * 1000

# SBF satellite letter -> ScintPi 3 constellation code; svid numbering as in ScintPi 3
# lvl0 (SBF "S31" -> 131, "J02" -> 194).
CONSTELLATION_CODE = {RINEX_PREFIX[name]: code for code, name in CONSTELLATIONS.items()}
SVID_OFFSET = {RINEX_PREFIX[name]: offset for name, offset in RINEX_SVID_OFFSET.items()}

# Receiver channel of each SBF signal. Unlisted signals (GPS_L1PY/L2PY, GLO_L2P,
# GAL_E5 AltBOC, GAL_E6BC, BDS_B1C/B2b) are dropped.
CHANNEL = {
    "GPS_L1CA": 1, "GPS_L2C": 2, "GPS_L5": 3,
    "GLO_L1CA": 1, "GLO_L2CA": 2, "GLO_L3": 3,
    "GAL_E1BC": 1, "GAL_E5a": 2, "GAL_E5b": 3,
    "BDS_B1I": 1, "BDS_B3I": 2, "BDS_B2a": 3,
    "GEO_L1": 1, "GEO_L5": 2,
    "QZS_L1CA": 1, "QZS_L2C": 2, "QZS_L5": 3,
}
CHANNELS = (1, 2, 3)

# Carrier frequency (MHz); GLONASS L1/L2 FDMA use base + k * step instead.
SIGNAL_MHZ = {
    "GPS_L1CA": 1575.42, "GPS_L2C": 1227.60, "GPS_L5": 1176.45,
    "GLO_L3": 1202.025,
    "GAL_E1BC": 1575.42, "GAL_E5a": 1176.45, "GAL_E5b": 1207.14,
    "BDS_B1I": 1561.098, "BDS_B3I": 1268.52, "BDS_B2a": 1176.45,
    "GEO_L1": 1575.42, "GEO_L5": 1176.45,
    "QZS_L1CA": 1575.42, "QZS_L2C": 1227.60, "QZS_L5": 1176.45,
}
GLONASS_FDMA_MHZ = {"GLO_L1CA": (GLONASS_L1_MHZ, GLONASS_L1_STEP_MHZ),
                    "GLO_L2CA": (GLONASS_L2_MHZ, GLONASS_L2_STEP_MHZ)}
GLONASS_FREQNR_OFFSET = 8  # SBF FreqNr = k + 8


def parse_sc4_name(path):
    """``(station prefix, start time)`` from an SBF file name, or None. Time is GPS (receiver) time."""
    match = SBF_NAME.match(os.path.basename(os.fspath(path)))
    if match is None:
        return None
    prefix, doy, hour, minute, yy = match.groups()
    start = datetime(2000 + int(yy), 1, 1) + timedelta(
        days=int(doy) - 1, hours=ord(hour.lower()) - ord("a"), minutes=int(minute))
    return prefix.lower(), start


def sc4_prefixes():
    """SC4 station prefixes from the bundled station registry."""
    prefixes = load_station_codes()["SC4 Prefix"].dropna()
    return {str(p).strip().lower() for p in prefixes}


def is_sc4_file(path):
    """True for a raw SC4 SBF file: registered station prefix and ``<prefix><doy><h><mm>.<yy>_`` name."""
    parsed = parse_sc4_name(path)
    return parsed is not None and parsed[0] in sc4_prefixes()


def find_bin2asc(executable=None):
    """RXTools bin2asc command as an argv list: argument, then $SCINTKIT_BIN2ASC, then PATH.

    On PATH, ``bin2asc`` is tried first, then the cluster wrapper ``rxtools_exec.sh bin2asc``.
    """
    for command in (executable, os.environ.get(BIN2ASC_ENV)):
        if command:
            argv = shlex.split(os.fspath(command)) if isinstance(command, (str, os.PathLike)) else list(command)
            if shutil.which(argv[0]) is None:
                raise FileNotFoundError(f"bin2asc command not found: {argv[0]}")
            return argv
    if shutil.which("bin2asc"):
        return ["bin2asc"]
    if shutil.which("rxtools_exec.sh"):
        return ["rxtools_exec.sh", "bin2asc"]
    raise FileNotFoundError(
        "Septentrio RXTools bin2asc is required to read SC4 (SBF) files. Install RXTools "
        f"(or `module load rxtools`) and put bin2asc on PATH, or set {BIN2ASC_ENV}."
    )


def run_bin2asc(sbf_path, work_dir, bin2asc=None):
    """Run bin2asc on ``sbf_path`` inside ``work_dir``; return {"measurements" or block name: text path}."""
    work_dir = Path(work_dir)
    local = work_dir / Path(sbf_path).name
    if not local.exists():
        shutil.copy2(sbf_path, local)
    # Run in work_dir with a relative name: the cluster's Apptainer wrapper only binds $PWD.
    # -t -x: column titles (and a preamble for SBF blocks), so columns are read by name.
    argv = find_bin2asc(bin2asc) + ["-f", local.name, "-m", *SBF_BLOCKS, "--extractGenMeas", "-t", "-x"]
    result = subprocess.run(argv, cwd=work_dir, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"bin2asc failed on {sbf_path}: {result.stderr[-2000:]}")
    found = {"measurements": next(work_dir.glob(f"{local.name}*measurements.txt"), None)}
    found.update({b: next(work_dir.glob(f"{local.name}*_SBF_{b}.txt"), None) for b in SBF_BLOCKS})
    if found["measurements"] is None:
        raise RuntimeError(f"bin2asc wrote no measurements for {sbf_path}")
    return {k: v for k, v in found.items() if v is not None}


def read_measurements(path):
    """bin2asc ``--extractGenMeas`` text -> long frame: gps_ms, svid, sig, pr (m), cph (cycles), cn0 (dB-Hz)."""
    usecols = ["TOW [s]", "WNc [w]", "SVID", "SignalType", "Antenna ID", "PR [m]", "L [cyc]", "C/N0 [dB-Hz]"]
    df = pd.read_csv(path, skiprows=[1], usecols=usecols, low_memory=False,
                     dtype={"SVID": "string", "SignalType": "string", "Antenna ID": "string"})
    df = df[df["Antenna ID"] == "Main"]
    tow_ms = np.round(pd.to_numeric(df["TOW [s]"], errors="coerce") * 1000)
    out = pd.DataFrame({
        "gps_ms": pd.to_numeric(df["WNc [w]"], errors="coerce") * WEEK_MS + tow_ms,
        "svid": df["SVID"], "sig": df["SignalType"],
        "pr": pd.to_numeric(df["PR [m]"], errors="coerce"),
        "cph": pd.to_numeric(df["L [cyc]"], errors="coerce"),
        "cn0": pd.to_numeric(df["C/N0 [dB-Hz]"], errors="coerce"),
    }).dropna(subset=["gps_ms"])
    out["gps_ms"] = out["gps_ms"].astype(np.int64)
    return out.reset_index(drop=True)


def read_block(path):
    """One bin2asc SBF block text file (4 preamble lines, header, dashes, rows) as a frame."""
    return pd.read_csv(path, skiprows=[0, 1, 2, 3, 5], low_memory=False)


def read_visibility(path):
    """SatVisibility1 text -> gps_s, svid, elev, azim (degrees, from the receiver's ephemeris)."""
    df = read_block(path)
    out = pd.DataFrame({
        "gps_s": (df["WNc [w]"] * WEEK_MS + np.round(df["TOW [s]"] * 1000)).astype(np.int64) // 1000,
        "svid": df["SVID"].astype("string"),
        "elev": df["Elevation [°]"].astype(np.float32),
        "azim": df["Azimuth [°]"].astype(np.float32),
    })
    return out.drop_duplicates(["gps_s", "svid"])


def leap_seconds(gpsutc_path=None):
    """GPS - UTC (s) from a GPSUtc block text, else DEFAULT_LEAP_SECONDS."""
    if gpsutc_path is not None:
        values = read_block(gpsutc_path)["DEL_t_LS [s]"].dropna()
        if len(values):
            return int(values.mode()[0])
    logger.warning("no GPSUtc block, assuming GPS - UTC = %d s", DEFAULT_LEAP_SECONDS)
    return DEFAULT_LEAP_SECONDS


def glonass_k(glonav_path=None, visibility_path=None):
    """GLONASS frequency channel k per SBF svid (e.g. {"R09": -2}) from GLONav/SatVisibility1 FreqNr."""
    k = {}
    for path in (visibility_path, glonav_path):
        if path is None:
            continue
        df = read_block(path)[["SVID", "FreqNr"]].dropna()
        df = df[df["SVID"].astype(str).str.startswith("R")]
        k.update({sv: int(nr) - GLONASS_FREQNR_OFFSET for sv, nr in zip(df["SVID"], df["FreqNr"])})
    return k


def frequency_mhz(sig, svid, k):
    """Carrier frequency (MHz) for signal names and SBF svids; GLONASS FDMA from ``k``; NaN if unknown."""
    freq = sig.map(SIGNAL_MHZ).astype(float)
    for name, (base, step) in GLONASS_FDMA_MHZ.items():
        rows = (sig == name).fillna(False).astype(bool)
        freq[rows] = base + step * svid[rows].map(k).astype(float)
    return freq


def to_lvl0_frame(meas, visibility=None, k=None, leap=DEFAULT_LEAP_SECONDS):
    """Long measurements -> lvl0 rows (one per satellite and epoch) with UTC ``datetime``.

    Columns: datetime, cons, svid, elev, azim, snr/cph/rng 1-3, sig_1-3, freq_1-3 (MHz).
    A channel without a measurement (missing or zero) is NaN.
    """
    k = k or {}
    meas = meas.assign(ch=meas["sig"].map(CHANNEL)).dropna(subset=["ch"])
    meas = meas.drop_duplicates(["gps_ms", "svid", "ch"])
    meas["ch"] = meas["ch"].astype(int)
    wide = meas.set_index(["gps_ms", "svid", "ch"])[["sig", "cn0", "cph", "pr"]].unstack("ch")
    wide.columns = [f"{name}{ch}" for name, ch in wide.columns]
    wide = wide.reset_index()

    df = pd.DataFrame({"gps_ms": wide["gps_ms"], "svid_sbf": wide["svid"].astype(str)})
    for ch in CHANNELS:
        get = lambda name: wide[f"{name}{ch}"] if f"{name}{ch}" in wide else pd.Series(np.nan, index=wide.index)
        snr, cph, rng = get("cn0").astype(np.float32), get("cph").astype(float), get("pr").astype(float)
        absent = snr.isna() | cph.isna() | rng.isna() | snr.eq(0) | cph.eq(0) | rng.eq(0)
        df[f"snr{ch}"] = snr.mask(absent)
        df[f"cph{ch}"] = cph.mask(absent)
        df[f"rng{ch}"] = rng.mask(absent)
        df[f"sig_{ch}"] = get("sig").astype("string").mask(absent)
        df[f"freq_{ch}"] = frequency_mhz(df[f"sig_{ch}"], df["svid_sbf"], k)
    has_data = df[[f"cph{ch}" for ch in CHANNELS]].notna().any(axis=1)
    df = df[has_data]

    letter = df["svid_sbf"].str[0]
    df = df[letter.isin(list(CONSTELLATION_CODE))].copy()
    letter = df["svid_sbf"].str[0]
    df["cons"] = letter.map(CONSTELLATION_CODE).astype(np.uint8)
    df["svid"] = (df["svid_sbf"].str[1:].astype(int) + letter.map(SVID_OFFSET).fillna(0)).astype(np.uint8)

    if visibility is not None and len(visibility):
        df["gps_s"] = df["gps_ms"] // 1000
        df = df.merge(visibility.rename(columns={"svid": "svid_sbf"}), on=["gps_s", "svid_sbf"], how="left")
    else:
        df["elev"] = np.float32(np.nan)
        df["azim"] = np.float32(np.nan)
    df["datetime"] = (pd.Timestamp(GPS_EPOCH) + pd.to_timedelta(df["gps_ms"], unit="ms")
                      - pd.Timedelta(seconds=leap)).astype("datetime64[ns]")

    columns = ["datetime", "cons", "svid", "elev", "azim"]
    columns += [f"{n}{ch}" for n in ("snr", "cph", "rng") for ch in CHANNELS]
    columns += [f"sig_{ch}" for ch in CHANNELS] + [f"freq_{ch}" for ch in CHANNELS]
    return df.sort_values(["datetime", "cons", "svid"])[columns].reset_index(drop=True)


def sampling_rate(df):
    """Sampling rate (Hz) from the modal spacing of epochs."""
    step = df["datetime"].drop_duplicates().diff().dt.total_seconds().round(6).mode()
    return float(round(1 / step[0], 3)) if len(step) else float("nan")


def receiver_info(setup_path):
    """``"<product> <firmware>"`` from a ReceiverSetup1 block, e.g. ``"mosaic-X5 4.14.0"``; None if absent."""
    if setup_path is None:
        return None
    row = read_block(setup_path).iloc[0]
    return f"{row.get('ProductName', '')} {row.get('RxVersion', '')}".strip()


def convert_sc4_to_lvl0(path, out_dir, bin2asc=None, work_dir=None, overwrite=False):
    """Convert one raw SC4 SBF file to lvl0 Parquet in ``out_dir`` (name from filenames.output_path); return its path.

    Needs RXTools bin2asc (see find_bin2asc). Text output goes to a temporary
    directory under ``work_dir`` (system default if None) and is removed.
    """
    path = Path(path)
    if parse_sc4_name(path) is None:
        raise ValueError(f"not an SC4 SBF file name: {path.name}")
    out = output_path(path, out_dir, "lvl0")
    if os.path.exists(out) and not overwrite:
        logger.info("lvl0 exists: %s", out)
        return out

    with tempfile.TemporaryDirectory(prefix="sc4_", dir=work_dir) as tmp:
        logger.info("bin2asc: %s", path)
        texts = run_bin2asc(path, tmp, bin2asc)
        leap = leap_seconds(texts.get("GPSUtc"))
        vis_path = texts.get("SatVisibility1")
        visibility = read_visibility(vis_path) if vis_path else None
        if visibility is None:
            logger.warning("%s: no SatVisibility1 block, elev/azim left NaN", path.name)
        k = glonass_k(texts.get("GLONav"), vis_path)
        df = to_lvl0_frame(read_measurements(texts["measurements"]), visibility, k, leap)
        receiver = receiver_info(texts.get("ReceiverSetup1"))

    if df.empty:
        raise ValueError(f"no measurements in {path}")
    unknown_glo = df.loc[df["cons"] == CONSTELLATION_CODE["R"], "freq_1"].isna().mean()
    if unknown_glo > 0:
        logger.warning("%s: GLONASS k unknown for %.0f%% of GLONASS rows (freq NaN)", path.name, 100 * unknown_glo)

    firmware = f"SC4 {receiver}" if receiver else "SC4"
    schema.write(df, out, schema.file_metadata(path, firmware, sampling_rate(df), leap))
    logger.info("wrote: %s (%d rows)", out, len(df))
    return out
