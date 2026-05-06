
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
from scintkit.services.plotting import individual_summary_plot  

fig, axs=individual_summary_plot(df,timecol='secbin',elmask=10)


# %%
