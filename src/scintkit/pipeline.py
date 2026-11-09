
import os
from scintkit.formatting import make_1min, make_1sec
from scintkit.products import add_products
from scintkit.reading import build_output_path, get_version, process_files, process_one

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


def process(flist, verbose=False,mode='both',fs=50):
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
            df = add_products(df, verbose=verbose,fs=fs)
            

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


import glob

def find_missing_inputs(
    input_pattern: str,
    input_root: str,
    output_root: str,
    allowed_versions: set[str] | None = None,
) -> list[str]:
    inputs = glob.glob(input_pattern)

    missing = []
    for f in inputs:
        version = get_version(f)
        if allowed_versions is not None and version not in allowed_versions:
            continue

        output_file = build_output_path(
            input_file=f,
            input_root=input_root,
            output_root=output_root,
            output_suffix=".pq",
        )

        if not os.path.exists(output_file):
            missing.append(f)

    return sorted(missing)


def chunk_for_slurm(
    flist: list[str],
    n_tasks: int,
    task_id: int | None = None,
) -> list[str]:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    if task_id < 0 or task_id >= n_tasks:
        raise ValueError(f"task_id must be in [0, {n_tasks - 1}], got {task_id}")

    chunks = np.array_split(flist, n_tasks)
    return list(chunks[task_id])
def resolve_flist(
    flist: list[str] | None = None,
    input_pattern: str | None = None,
    infer_missing: bool = False,
    input_root: str | None = None,
    output_root: str | None = None,
    allowed_versions: set[str] | None = None,
) -> list[str]:
    if flist is not None:
        return sorted(flist)

    if infer_missing:
        if input_pattern is None or input_root is None or output_root is None:
            raise ValueError(
                "input_pattern, input_root, and output_root are required when infer_missing=True"
            )

        return find_missing_inputs(
            input_pattern=input_pattern,
            input_root=input_root,
            output_root=output_root,
            allowed_versions=allowed_versions,
        )

    if input_pattern is not None:
        files = sorted(glob.glob(input_pattern))

        if allowed_versions is not None:
            files = [f for f in files if get_version(f) in allowed_versions]

        return files

    raise ValueError(
        "Must provide one of: flist, input_pattern, or infer_missing=True"
    )


def run_conversion(
    mode: str = "single",
    flist: list[str] | None = None,
    input_pattern: str | None = None,
    infer_missing: bool = False,
    input_root: str = "",
    output_root: str = "",
    allowed_versions: set[str] | None = None,
    n_workers: int = 1,
    n_tasks: int | None = None,
    task_id: int | None = None,
    temp_root: str = "/tmp",
    compression: str = "brotli",
    compression_level: int = 9,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    flist_resolved = resolve_flist(
        flist=flist,
        input_pattern=input_pattern,
        infer_missing=infer_missing,
        input_root=input_root,
        output_root=output_root,
        allowed_versions=allowed_versions,
    )

    if verbose:
        print(f"total_files_resolved={len(flist_resolved)}")

    if mode == "slurm":
        if n_tasks is None:
            raise ValueError("n_tasks is required for mode='slurm'")

        flist_resolved = chunk_for_slurm(
            flist=flist_resolved,
            n_tasks=n_tasks,
            task_id=task_id,
        )

        if verbose:
            resolved_task_id = task_id
            if resolved_task_id is None:
                resolved_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            print(f"mode=slurm task_id={resolved_task_id} files_in_chunk={len(flist_resolved)}")

    elif mode != "single":
        raise ValueError("mode must be 'single' or 'slurm'")

    return process_files(
        flist=flist_resolved,
        input_root=input_root,
        output_root=output_root,
        n_workers=n_workers,
        temp_root=temp_root,
        compression=compression,
        compression_level=compression_level,
        overwrite=overwrite,
        verbose=verbose,
    )



import argparse

def main():
    """
    Parallel ScintPi processing function for HPC SLURM array jobs

    This script distributes files across SLURM array tasks and then
    uses local multiprocessing within each task.

    Inputs
    ------
    --data-dir
        Directory containing input files.

    --pattern
        Glob pattern used to match input files.

    --mode
        Processing mode:
            'lvl2' -> create 1 second products
            'lvl3' -> create 1 minute products
            'both' -> create both products

            
    SLURM Environment Variables (set in slurm code)
    SLURM_ARRAY_TASK_ID
        Index of the current array task.

    SLURM_ARRAY_TASK_COUNT
        Total number of array tasks.

    SLURM_CPUS_PER_TASK
        Number of multiprocessing workers to use locally.

    Behavior: 
    1. Find all matching files.
    2. Split the file list evenly across SLURM array tasks.
    3. Assign one chunk to the current task.
    4. Process the assigned files using multiprocessing.

    Example
    -------
    python hpc_processing.py \
        --data-dir /data \
        --pattern "scintpi3_*.bin*" \
        --mode both
    """
    parser = argparse.ArgumentParser(
        description="Parallel ScintPi processing for SLURM array jobs"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing input files",
    )

    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for input files",
    )

    parser.add_argument(
        "--mode",
        default="both",
        choices=["lvl2", "lvl3", "both"],
        help="Processing mode",
    )

    args = parser.parse_args()

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    n_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    files = sorted(args.data_dir.glob(args.pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {args.data_dir} matching {args.pattern}"
        )

    chunks = np.array_split(files, n_tasks)
    my_files = [str(f) for f in chunks[task_id]]

    print(f"Task {task_id}/{n_tasks - 1}")
    print(f"Workers: {n_workers}")
    print(f"Assigned files: {len(my_files)}")

    outfiles = process_parallel(
        my_files,
        n_workers=n_workers,
        verbose=True,
        mode=args.mode,
    )

    print("Finished outputs:")
    for f in outfiles:
        print(f)


if __name__ == "__main__":
    main()