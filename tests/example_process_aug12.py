
#%%
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import scintkit

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

lvl3_files = scintkit.pipelines.auto.process(raw_files, verbose=True,mode='lvl3')

# %%

