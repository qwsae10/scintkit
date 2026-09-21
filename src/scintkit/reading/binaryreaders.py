import struct

import numpy as np
import pandas as pd

# %%


def readv324(path):
    cols2read = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]
    colnames = [
        "week",
        "towe",
        "cons",
        "svid",
        "elev",
        "azim",
        "snr1",
        "snr2",
        "cph1",
        "cph2",
        "rng1",
        "rng2",
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=cols2read)
    df.columns = colnames
    return df


def readv325(path):
    dt = np.dtype(
        [
            ("week", np.int32),
            ("towe", np.float32),
            ("leap", np.uint8),
            ("cons", np.uint8),
            ("sats", np.uint8),
            ("svid", np.uint8),
            ("elev", np.int8),
            ("azim", np.int32),
            ("snr1", np.uint8),
            ("snr2", np.uint8),
            ("snr3", np.uint8),
            ("pst1", np.uint8),
            ("pst2", np.uint8),
            ("pst3", np.uint8),
            ("rst1", np.uint8),
            ("rst2", np.uint8),
            ("rst3", np.uint8),
            ("cph1", np.float64),
            ("cph2", np.float64),
            ("cph3", np.float64),
            ("rng1", np.float64),
            ("rng2", np.float64),
            ("rng3", np.float64),
            ("lon", np.float32),
            ("lat", np.float32),
            ("hei", np.float32),
        ],
        align=True,
    )
    arr = np.fromfile(path, dtype=dt)
    return pd.DataFrame.from_records(arr)


def readv326(path):
    rec_dt = np.dtype(
        [
            ("towe", np.float32),
            ("cons", np.uint8),
            ("sats", np.uint8),
            ("svid", np.uint8),
            ("elev", np.int8),
            ("azim", np.int32),
            ("snr1", np.uint8),
            ("snr2", np.uint8),
            ("pst1", np.uint8),
            ("pst2", np.uint8),
            ("rst1", np.uint8),
            ("rst2", np.uint8),
            ("cph1", np.float64),
            ("cph2", np.float64),
            ("rng1", np.float64),
            ("rng2", np.float64),
            ("lck1", np.int32),
            ("lck2", np.int32),
        ],
        align=True,
    )
    hdr_size = 60
    with open(path, "rb") as f:
        buf = f.read(hdr_size)
    fmt = "@fBbiBBBBBBddddi"
    vals = struct.unpack(fmt, buf)
    week = vals[-1]
    rec = np.fromfile(path, dtype=rec_dt, offset=64 * 2)
    df = pd.DataFrame.from_records(rec)
    df["week"] = week
    return df


# %%
