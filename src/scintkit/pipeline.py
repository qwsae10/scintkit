"""End-to-end processing: raw/lvl0 files -> lvl0, lvl2 (1 s) and lvl3 (1 min) Parquet files.

Also the ``scintkit-process`` command line (file list, Slurm array task, recent files).
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool

from scintkit import schema
from scintkit.filenames import output_path, start_time
from scintkit.machine_learning.classify import classify_minutes, null_classes
from scintkit.products import assemble, compute_products
from scintkit.reading import UnsupportedFirmware, read_lvl0, to_lvl0

logger = logging.getLogger(__name__)

RAW_SUFFIXES = (".bin", ".bin.zip", ".dat", ".dat.zip")
LVL0_SUFFIXES = ("_lvl0.pq", "_lvl0.parquet")
LEVELS = {"lvl0": [], "lvl2": ["lvl2"], "lvl3": ["lvl3"], "both": ["lvl2", "lvl3"]}


def _out_dir(path, output_dir, mirror):
    return os.path.join(output_dir, os.path.basename(os.path.dirname(os.path.abspath(path)))) if mirror else output_dir


def process_file(path, output_dir, mode="both", fs=None, overwrite=False, classify=True):
    """Write the outputs of one raw or lvl0 file into ``output_dir``; return the paths written.

    ``mode`` is "lvl0", "lvl2", "lvl3" or "both" (lvl2 and lvl3); raw input
    also gets its lvl0 file. Existing outputs are kept unless ``overwrite``.
    ``fs`` (Hz) is detected from the data when None. With ``classify`` lvl3
    gets the signal class of gated minutes (20 Hz data and the ``ml`` extra).
    """
    path = os.fspath(path)
    targets = {level: output_path(path, output_dir, level) for level in LEVELS[mode]}
    is_lvl0 = path.lower().endswith(LVL0_SUFFIXES)
    lvl0 = path if is_lvl0 else output_path(path, output_dir, "lvl0")
    wanted = list(targets.values()) + ([] if is_lvl0 else [lvl0])
    if not overwrite and all(os.path.exists(p) for p in wanted):
        logger.info("outputs exist, skipping %s", path)
        return []

    written = []
    if not is_lvl0 and (overwrite or not os.path.exists(lvl0)):
        written.append(to_lvl0(path, output_dir, overwrite=True))
    if not targets:
        return written

    df, meta = read_lvl0(lvl0)
    frame, minute, fs = compute_products(df, fs=fs)
    del df
    file_meta = schema.file_metadata(meta.get("source_file", path), meta.get("firmware"), fs,
                                     meta["leap_seconds"])
    ml_meta = {}
    if "lvl3" in targets:
        minute, ml_meta = (classify_minutes(frame, minute, fs) if classify
                           else (null_classes(minute), {"ml_classification": "off"}))
    for level, out in targets.items():
        level_meta = {**file_meta, **ml_meta} if level == "lvl3" else file_meta
        schema.write(assemble(frame, minute, "1min" if level == "lvl3" else "1s"), out, level_meta)
        logger.info("wrote %s", out)
        written.append(out)
    return written


def _run(task):
    """Process one file; return ``(path, status, written paths or reason)`` without raising."""
    path, output_dir, mode, fs, overwrite, classify = task
    try:
        written = process_file(path, output_dir, mode=mode, fs=fs, overwrite=overwrite, classify=classify)
        return path, "ok" if written else "skipped", written
    except UnsupportedFirmware as error:
        logger.warning("unsupported: %s", error)
        return path, "unsupported", str(error)
    except Exception as error:
        logger.exception("failed: %s", path)
        return path, "failed", f"{type(error).__name__}: {error}"


def _summarise(results):
    summary = {"written": [], "skipped": [], "failed": [], "unsupported": []}
    for path, status, detail in results:
        if status == "ok":
            summary["written"].extend(detail)
        elif status == "skipped":
            summary["skipped"].append(path)
        else:
            summary[status].append((path, detail))
    n_ok = len(results) - sum(len(summary[k]) for k in ("skipped", "failed", "unsupported"))
    logger.info("%d ok, %d skipped, %d failed, %d unsupported", n_ok, len(summary["skipped"]),
                len(summary["failed"]), len(summary["unsupported"]))
    for path, reason in summary["failed"] + summary["unsupported"]:
        logger.info("  %s: %s", os.path.basename(path), reason)
    return summary


def _tasks(files, output_dir, mode, fs, overwrite, mirror, classify):
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    return [(os.fspath(f), _out_dir(f, os.fspath(output_dir), mirror), mode, fs, overwrite, classify)
            for f in files]


def process(files, output_dir, mode="both", fs=None, overwrite=False, mirror=False, classify=True):
    """Process files one after another; return a summary dict.

    The summary has ``written`` (paths), ``skipped`` (inputs whose outputs
    exist), and ``failed``/``unsupported`` as ``(input, reason)`` pairs. With
    ``mirror`` outputs go to ``output_dir/<input's folder name>/``.
    """
    return _summarise([_run(t) for t in _tasks(files, output_dir, mode, fs, overwrite, mirror, classify)])


def process_parallel(files, output_dir, n_workers=4, mode="both", fs=None, overwrite=False, mirror=False,
                     classify=True):
    """:func:`process` with one file per task on ``n_workers`` processes; a failure never stops the pool.

    A 4 h 20 Hz file needs about 20 GB of memory, so size ``n_workers`` to the machine.
    """
    tasks = _tasks(files, output_dir, mode, fs, overwrite, mirror, classify)
    with Pool(n_workers, maxtasksperchild=1) as pool:
        return _summarise(list(pool.imap_unordered(_run, tasks)))


def find_recent_files(archive_root, days=1, now=None):
    """Raw files in ``archive_root/<station>/`` whose file-name start time is within the last ``days`` days.

    ``now`` is a naive UTC datetime (default: the current time).
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    since = now - timedelta(days=days)
    found = []
    for path in glob.glob(os.path.join(os.fspath(archive_root), "*", "*")):
        start = start_time(path)
        if path.lower().endswith(RAW_SUFFIXES) and start is not None and since <= start <= now:
            found.append(path)
    return sorted(found)


def process_recent(archive_root, output_root, days=1, n_workers=4, mode="both", classify=True):
    """Process the raw files of the last ``days`` days into ``output_root/<station>/``, skipping done ones.

    Meant for a daily cron job; returns the :func:`process` summary.
    """
    files = find_recent_files(archive_root, days=days)
    logger.info("%d raw files in the last %g days under %s", len(files), days, archive_root)
    return process_parallel(files, output_root, n_workers=n_workers, mode=mode, mirror=True, classify=classify)


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


def _expand(patterns):
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)) or [pattern])
    return files


def _run_files(args):
    return process_parallel(_expand(args.files), args.output, n_workers=args.workers, mode=args.mode,
                            fs=args.fs, overwrite=args.overwrite, mirror=True, classify=args.classify)


def _run_array(args):
    files = sorted(glob.glob(args.pattern))
    if not 0 <= args.task_id < args.n_tasks:
        raise SystemExit(f"task id must be in [0, {args.n_tasks - 1}], got {args.task_id}")
    mine = split_evenly(files, args.n_tasks)[args.task_id]
    logger.info("task %d/%d: %d of %d files", args.task_id, args.n_tasks, len(mine), len(files))
    return process_parallel(mine, args.output, n_workers=args.workers, mode=args.mode,
                            fs=args.fs, overwrite=args.overwrite, mirror=True, classify=args.classify)


def _run_recent(args):
    return process_recent(args.archive, args.output, days=args.days, n_workers=args.workers, mode=args.mode,
                          classify=args.classify)


def main(argv=None):
    """Entry point for ``scintkit-process``; exits 1 if any file failed."""
    env = os.environ.get
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True,
                        help="output root; files go to <output>/<input folder name>/")
    common.add_argument("--workers", type=int, default=int(env("SLURM_CPUS_PER_TASK", 1)),
                        help="parallel files (default: $SLURM_CPUS_PER_TASK or 1); ~20 GB each")
    common.add_argument("--mode", default="both", choices=list(LEVELS))
    common.add_argument("--fs", type=float, help="sampling rate in Hz (default: detect)")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--no-classify", dest="classify", action="store_false",
                        help="leave the lvl3 signal-class columns null")

    parser = argparse.ArgumentParser(prog="scintkit-process", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    files = commands.add_parser("files", parents=[common], help="process the given files or globs")
    files.add_argument("files", nargs="+")
    files.set_defaults(run=_run_files)

    array = commands.add_parser("array", parents=[common], help="process this Slurm array task's share")
    array.add_argument("--pattern", required=True, help="glob of raw or lvl0 files")
    array.add_argument("--task-id", type=int, default=int(env("SLURM_ARRAY_TASK_ID", 0)))
    array.add_argument("--n-tasks", type=int, default=int(env("SLURM_ARRAY_TASK_COUNT", 1)))
    array.set_defaults(run=_run_array)

    recent = commands.add_parser("recent", parents=[common], help="process recent files of an archive")
    recent.add_argument("--archive", required=True, help="archive root with one folder per station")
    recent.add_argument("--days", type=float, default=1)
    recent.set_defaults(run=_run_recent)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = args.run(args)
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
