
import os
from scintkit.preprocessing.format import temp_formating,make_1min,make_1sec
from scintkit.services.compute import add_products
from scintkit.services.convert_to_parquet import process_one

import pandas as pd
import os
import pandas as pd
from pathlib import Path

def get_type(f):
    name = Path(f).name.lower()

    if name.endswith(".bin.zip"):
        return "binzip"
    if name.endswith(".bin"):
        return "bin"
    if name.endswith("_lvl0.pq") or name.endswith("_lvl0.parquet"):
        return "lvl0"
    if name.endswith("_lvl1.pq") or name.endswith("_lvl1.parquet"):
        return "lvl1"
    if name.endswith("_lvl2.pq") or name.endswith("_lvl2.parquet"):
        return "lvl2"
    if name.endswith("_lvl3.pq") or name.endswith("_lvl3.parquet"):
        return "lvl3"
    elif name.endswith(".pq") or name.endswith(".parquet"):
        return "pq"
    return None


def process(flist, verbose=False,mode='both'):
    """
    Wrapper to run full pipeline on list of files and make high level scintillation index product files (lvl3)
    Inputs:
    - flist: list of file paths to process. Can be .bin, .bin.zip, or .pq files. Output files will have same relative path but with _lvl3.pq suffix.
    - verbose: if True, print progress messages.
    - mode: 'lvl2' to make 1 second rate files. 'lvl3' to make 1 minute rate files. 'both' to make both. Default is 'both'.
    Outputs:
    - list of output file paths that were created.

    Each output file 'lvl3' will be 1 minute rate with then following added products:

    - tec12 and tec13: differences between detrended phases to estimate TEC (WIP)
    - sigma_phi_1, sigma_phi_2, sigma_phi_3: standard deviation of detrended phases with clock noise removed, for each frequency
    - n_1, n_2, n_3: number of valid samples for each frequency
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - quality_1, quality_2, quality_3: binary flags indicating potential quality issues (0 means no issue, 1 or more means issue) 
    - s4_1, s4_2, s4_3: S4 index computed from SNR values for each frequency
    - s4_corrected_1, s4_corrected_2, s4_corrected_3: S4 index corrected for bias based on Van Dierendonck (1993) method
    - clock_term: estimated common clock term (in units of radians/frequency) across all frequencies, used for detrending phases to compute sigma_phi 

    """

    if flist is None:
        flist = []
    elif isinstance(flist, (str, os.PathLike)):
        flist = [flist]

    if isinstance(flist, (list, tuple, set)):
        flist = list(flist)
    else:
        raise TypeError(
            "invalid file list type: expected path-like, list, tuple, or set, "
            f"got {type(flist)}"
        )

    converted_files = []

    allowed_types = ['bin', 'binzip', 'lvl0',]

    flist=[f for f in flist if get_type(f) in allowed_types]
    
    for fname in flist:
        
        try:

            if verbose:
                print(f"Processing {fname}...")

            ext = os.path.splitext(str(fname))[1].lower()

            # skip conversion if already parquet
            if ext in [".pq", ".parquet"]:
                pq_fname = fname
            else:
                pq_fname = process_one(fname)

            if verbose:
                print(f"Reading and formatting parquet file: {pq_fname}...")
            df = pd.read_parquet(pq_fname)
            df = add_products(df, verbose=verbose)
            

            if mode=='lvl2':
                df = make_1sec(df)
                outname = str(pq_fname).replace("_lvl0", "_lvl2")

                df.to_parquet(outname)
                converted_files.append(outname)


            if mode=='lvl3':
                df = make_1min(df)
                outname = str(pq_fname).replace("_lvl0", "_lvl3")
                df.to_parquet(outname)
                converted_files.append(outname)

            if mode =='both':
                df_1min = make_1min(df)
                outname_1min = str(pq_fname).replace("_lvl0", "_lvl3")
                df_1min.to_parquet(outname_1min)
                converted_files.append(outname_1min)
                df_1sec = make_1sec(df)
                outname_1sec = str(pq_fname).replace("_lvl0", "_lvl2")
                df_1sec.to_parquet(outname_1sec)
                converted_files.append(outname_1sec)



            if verbose:
                print(f"Finished processing {fname}.")


        except Exception as e:
            print(f"Error processing {fname}") 
            print(e)
            continue
    return converted_files

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import numpy as np


def split_list(items, n):
    items = list(items)
    if n <= 1:
        return [items]
    return [chunk.tolist() for chunk in np.array_split(items, n) if len(chunk) > 0]


def process_parallel(flist, n_workers=4, verbose=False, mode="both"):
    """
    Run process() in parallel by splitting flist across n workers.
    """

    if flist is None:
        flist = []
    elif isinstance(flist, (str, os.PathLike)):
        flist = [flist]
    else:
        flist = list(flist)

    chunks = split_list(flist, n_workers)

    all_outputs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process, chunk, verbose, mode)
            for chunk in chunks
        ]

        for future in as_completed(futures):
            try:
                outputs = future.result()
                all_outputs.extend(outputs)
            except Exception as e:
                print(f"Worker failed: {e}")

    return all_outputs

