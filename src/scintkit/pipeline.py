"""End-to-end processing: raw/lvl0 files -> lvl2 (1 s) and lvl3 (1 min) products.

Also the ``scintkit-process`` command line, which splits work across Slurm
array tasks.
"""

import argparse
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from scintkit.filenames import firmware_version
from scintkit.formatting import make_1min, make_1sec
from scintkit.products import compute_products, merge_products
from scintkit.reading import build_output_path, convert_to_lvl0

logger = logging.getLogger(__name__)

RAW_SUFFIXES = (".bin", ".bin.zip")
LVL0_SUFFIXES = ("_lvl0.pq", "_lvl0.parquet")


def process(flist, mode="both", fs=50):
    """Write lvl3 (``mode="lvl3"``), lvl2 (``"lvl2"``) or both for each file; return their paths.

    Accepts ``.bin``, ``.bin.zip`` and ``*_lvl0.pq`` files (other names are
    skipped). Outputs are written next to the lvl0 file. A file that fails is
    logged and skipped. ``fs`` is the sampling rate in Hz (None = detect).
    """
    if isinstance(flist, (str, os.PathLike)):
        flist = [flist]
    flist = [f for f in flist if str(f).lower().endswith(RAW_SUFFIXES + LVL0_SUFFIXES)]

    outputs = []
    for fname in flist:
        try:
            logger.info("Processing %s", fname)
            if str(fname).lower().endswith(LVL0_SUFFIXES):
                pq_fname = fname
            else:
                pq_fname = convert_to_lvl0(fname)

            # Products are merged onto the reduced lvl2/lvl3 rows, not the full-rate frame.
            frame, minute = compute_products(pd.read_parquet(pq_fname), fs=fs)

            if mode in ("lvl3", "both"):
                outname = str(pq_fname).replace("_lvl0", "_lvl3")
                merge_products(make_1min(frame), minute).to_parquet(outname)
                outputs.append(outname)
            if mode in ("lvl2", "both"):
                outname = str(pq_fname).replace("_lvl0", "_lvl2")
                merge_products(make_1sec(frame), minute).to_parquet(outname)
                outputs.append(outname)
        except Exception:
            logger.exception("Error processing %s", fname)
    return outputs


def split_evenly(items, n):
    """Split ``items`` into ``n`` contiguous chunks whose sizes differ by at most one."""
    items = list(items)
    size, extra = divmod(len(items), n)
    chunks, start = [], 0
    for k in range(n):
        stop = start + size + (k < extra)
        chunks.append(items[start:stop])
        start = stop
    return chunks


def process_parallel(flist, n_workers=4, mode="both"):
    """Run :func:`process` over ``flist`` split across ``n_workers`` processes."""
    if isinstance(flist, (str, os.PathLike)):
        flist = [flist]
    chunks = [c for c in split_evenly(flist, max(n_workers, 1)) if c]

    outputs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process, chunk, mode) for chunk in chunks]
        for future in as_completed(futures):
            try:
                outputs.extend(future.result())
            except Exception:
                logger.exception("Worker failed")
    return outputs


def _convert_one(args):
    input_file, kwargs = args
    try:
        return convert_to_lvl0(input_file, **kwargs)
    except Exception:
        logger.exception("Error converting %s", input_file)
        return None


def convert_files(flist, input_root, output_root, n_workers=1, temp_root=None,
                  compression_level=6, overwrite=False):
    """Convert raw files to lvl0 Parquet, mirroring ``input_root`` into ``output_root``.

    Returns the paths written (failed files are logged and left out).
    """
    kwargs = dict(input_root=input_root, output_root=output_root, temp_root=temp_root,
                  compression_level=compression_level, overwrite=overwrite)
    jobs = [(f, kwargs) for f in flist]
    if n_workers == 1:
        results = [_convert_one(job) for job in jobs]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_convert_one, jobs)
    return [r for r in results if r is not None]


def find_missing_inputs(pattern, input_root, output_root, versions=None):
    """Raw files matching ``pattern`` that have no converted counterpart in ``output_root``."""
    missing = []
    for f in glob.glob(pattern):
        if versions is not None and firmware_version(f) not in versions:
            continue
        if not os.path.exists(build_output_path(f, input_root, output_root, output_suffix=".pq")):
            missing.append(f)
    return sorted(missing)


def _array_task_files(files, args):
    if not 0 <= args.task_id < args.n_tasks:
        raise ValueError(f"task id must be in [0, {args.n_tasks - 1}], got {args.task_id}")
    mine = split_evenly(files, args.n_tasks)[args.task_id]
    logger.info("task %d/%d: %d of %d files", args.task_id, args.n_tasks, len(mine), len(files))
    return mine


def _run_products(args):
    files = sorted(str(f) for f in args.data_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {args.data_dir} matching {args.pattern}")
    outputs = process_parallel(_array_task_files(files, args), n_workers=args.workers, mode=args.mode)
    for f in outputs:
        logger.info("wrote %s", f)


def _run_convert(args):
    versions = set(args.versions) if args.versions else None
    if args.only_missing:
        files = find_missing_inputs(args.pattern, args.input_root, args.output_root, versions)
    else:
        files = sorted(f for f in glob.glob(args.pattern)
                       if versions is None or firmware_version(f) in versions)
    convert_files(
        _array_task_files(files, args),
        input_root=args.input_root,
        output_root=args.output_root,
        n_workers=args.workers,
        temp_root=args.temp_root,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
    )


def main(argv=None):
    """Entry point for ``scintkit-process``."""
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="scintkit-process", description=__doc__)
    array = argparse.ArgumentParser(add_help=False)
    array.add_argument("--task-id", type=int, default=int(env("SLURM_ARRAY_TASK_ID", 0)),
                       help="array task index (default: $SLURM_ARRAY_TASK_ID or 0)")
    array.add_argument("--n-tasks", type=int, default=int(env("SLURM_ARRAY_TASK_COUNT", 1)),
                       help="number of array tasks (default: $SLURM_ARRAY_TASK_COUNT or 1)")
    array.add_argument("--workers", type=int, default=int(env("SLURM_CPUS_PER_TASK", 1)),
                       help="local processes per task (default: $SLURM_CPUS_PER_TASK or 1)")
    commands = parser.add_subparsers(dest="command", required=True)

    products = commands.add_parser("products", parents=[array],
                                   help="make lvl2/lvl3 products from raw or lvl0 files")
    products.add_argument("--data-dir", required=True, type=Path)
    products.add_argument("--pattern", required=True, help="glob pattern inside --data-dir")
    products.add_argument("--mode", default="both", choices=["lvl2", "lvl3", "both"])
    products.set_defaults(run=_run_products)

    convert = commands.add_parser("convert", parents=[array], help="convert raw files to lvl0 Parquet")
    convert.add_argument("--pattern", required=True, help="glob pattern for raw files")
    convert.add_argument("--input-root", required=True)
    convert.add_argument("--output-root", required=True)
    convert.add_argument("--versions", nargs="+", help="only these firmware versions, e.g. v326")
    convert.add_argument("--only-missing", action="store_true", help="skip files already converted")
    convert.add_argument("--temp-root", help="scratch directory (default: system temp)")
    convert.add_argument("--compression-level", type=int, default=9)
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(run=_run_convert)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.run(args)


if __name__ == "__main__":
    main()
