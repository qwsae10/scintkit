"""Scan the archive, plot station availability, and email the plot."""

import argparse
import os

import pandas as pd

from scintkit.email_updates.core import (
    load_targets,
    scan_legacy_files,
    scan_sc4_files,
    scan_septentrio_files,
)
from scintkit.email_updates.mailer import send_status_email
from scintkit.email_updates.plotting import generate_availability_plot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", nargs="+",
                        help="recipient addresses (default: SCINTPI_STATUS_RECIPIENTS, comma-separated)")
    parser.add_argument("--archive", required=True, help="root of the ScintPi archive")
    parser.add_argument("--septentrio-dir", help="Septentrio file directory (scan skipped if omitted)")
    parser.add_argument("--months", type=int, default=3, help="months to look back (default 3)")
    parser.add_argument("--output", help="PNG path (default stations_availability_lvl3_YYYY_MM_DD.png)")
    args = parser.parse_args(argv)

    if not args.to:
        env = os.environ.get("SCINTPI_STATUS_RECIPIENTS", "")
        args.to = [a.strip() for a in env.split(",") if a.strip()]
    if not args.to:
        parser.error("give --to or set SCINTPI_STATUS_RECIPIENTS")
    return args


def main(argv=None):
    args = parse_args(argv)
    now = pd.Timestamp.now()
    cutoff = now - pd.DateOffset(months=args.months)
    output = args.output or f"stations_availability_lvl3_{now:%Y_%m_%d}.png"

    targets = load_targets()
    scan_legacy_files(targets, cutoff, args.archive)
    scan_sc4_files(targets, cutoff, args.archive)
    if args.septentrio_dir:
        scan_septentrio_files(targets, cutoff, args.septentrio_dir)

    generate_availability_plot(targets, cutoff, now, output)
    send_status_email(output, now, args.to)


if __name__ == "__main__":
    main()
