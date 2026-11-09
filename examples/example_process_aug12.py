
#%%
import logging
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from scintkit.pipeline import process

logging.basicConfig(level=logging.INFO)

data_dir = Path("/mfs/io/groups/uars/scintpi/sc007/")
raw_files = sorted(data_dir.glob("scintpi3_20240812*_65*_18*.bin.zip"))
#These can also be .bin or .bin.zip files

#if taking too long, specify subset
raw_files = raw_files

if not raw_files:
    raise FileNotFoundError(
        f"No ScintPi files found in {data_dir.resolve()}. "
        "Start Jupyter from the repository root, or update data_dir."
    )

output_dir = Path("output")  # lvl0 and lvl3 files are written here
summary = process(raw_files, output_dir, mode="lvl3")
lvl3_files = [p for p in summary["written"] if p.endswith("_lvl3.pq")]

# %%

