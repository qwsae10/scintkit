import os
from pathlib import Path

import pandas as pd

from scintkit.preprocessing.format import make_1min, make_1sec
from scintkit.services.compute import add_products
from scintkit.services.convert_to_parquet import process_one


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


def process(flist, verbose=False, mode="both", fs=50):
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
    - n_sigphi_1, n_sigphi_2, n_sigphi_3: number of valid detrended phase samples used for sigma-phi
    - n_s4_1, n_s4_2, n_s4_3: number of valid SNR samples used for S4
    - n_cycleslip_1, n_cycleslip_2, n_cycleslip_3: number of detected cycle slips for each phase
    - sigma_phi_quality_flag_1/2/3: binary sigma-phi quality flags; 0 is good and 1 marks an edge/gap, too many dropped samples, or GLONASS
    - s4_quality_flag_1/2/3: binary S4 quality flags; 0 is good and 1 marks fewer than 80% of the expected samples
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

    allowed_types = [
        "bin",
        "binzip",
        "lvl0",
    ]

    flist = [f for f in flist if get_type(f) in allowed_types]

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

            # Pass merge=False to avoid massive memory duplication
            df, products = add_products(df, verbose=verbose, fs=fs, merge=False)

            if mode == "lvl2":
                df = make_1sec(df)
                if products is not None:
                    df = df.merge(products, on=["prn", "minbin"], how="left")
                outname = str(pq_fname).replace("_lvl0", "_lvl2")
                df.to_parquet(outname)
                converted_files.append(outname)

            elif mode == "lvl3":
                df = make_1min(df)
                if products is not None:
                    df = df.merge(products, on=["prn", "minbin"], how="left")
                outname = str(pq_fname).replace("_lvl0", "_lvl3")
                df.to_parquet(outname)
                converted_files.append(outname)

            elif mode == "both":
                df_1min = make_1min(df)
                if products is not None:
                    df_1min = df_1min.merge(products, on=["prn", "minbin"], how="left")
                outname_1min = str(pq_fname).replace("_lvl0", "_lvl3")
                df_1min.to_parquet(outname_1min)
                converted_files.append(outname_1min)

                del df_1min
                import gc

                gc.collect()

                df_1sec = make_1sec(df)
                if products is not None:
                    df_1sec = df_1sec.merge(products, on=["prn", "minbin"], how="left")
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
        futures = [executor.submit(process, chunk, verbose, mode) for chunk in chunks]

        for future in as_completed(futures):
            try:
                outputs = future.result()
                all_outputs.extend(outputs)
            except Exception as e:
                print(f"Worker failed: {e}")

    return all_outputs
