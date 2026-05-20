

import os
import sys
from pathlib import Path
import numpy as np

from scintkit.pipelines.auto import process_parallel

import argparse
import os
from pathlib import Path
import numpy as np

from scintkit.pipelines.auto import process_parallel


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