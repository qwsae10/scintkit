#!/usr/bin/env python3

import glob
import os

import numpy as np

from scintkit.services.convert_to_parquet import (
    build_output_path,
    get_version,
    process_files,
)


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

    raise ValueError("Must provide one of: flist, input_pattern, or infer_missing=True")


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
            print(
                f"mode=slurm task_id={resolved_task_id} files_in_chunk={len(flist_resolved)}"
            )

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


if __name__ == "__main__":
    INPUT_ROOT = "/mfs/io/groups/uars/scintpi"
    OUTPUT_ROOT = "/titan/frodrigues/scintpi_storage"
    INPUT_PATTERN = "/mfs/io/groups/uars/scintpi/*/*2024*.bin.zip"

    run_conversion(
        mode="single",
        input_pattern=INPUT_PATTERN,
        infer_missing=True,
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        allowed_versions={"v324", "v325", "v326"},
        n_workers=1,
        temp_root="/tmp",
        overwrite=False,
        verbose=True,
    )
