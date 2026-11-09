"""Assemble a labelled training set from per-file feature tables (``scintkit-ml-features`` output).

Each site has one label; a row is kept when its S4 and elevation pass that
label's rule (and, for scintillation, it is local night). Example with the
2024-2026 training sites:

    python ml_assemble_labeled_features.py FEATURE_DIR labeled.pq \\
        --site Scintillation -7.213 -35.907 --site RFI 43.271 -120.358 --site Multipath 46.907 -96.793
"""

import argparse
import glob
import logging
import os

import pandas as pd

from scintkit.filenames import coordinates

logger = logging.getLogger(__name__)

S4_MIN = 0.25
SITE_TOLERANCE_DEG = 0.0005
MIN_ELEVATION_DEG = {"Scintillation": 40, "RFI": 40, "Multipath": 20}
NIGHT_AFTER_HOUR, NIGHT_BEFORE_HOUR = 18, 4  # local hours kept for Scintillation


def label_rows(table, label, timezone):
    """Rows of one feature table that get ``label``."""
    keep = table["s4_1"].gt(S4_MIN) & table["elev"].gt(MIN_ELEVATION_DEG[label]) & table["elev"].le(90)
    if label == "Scintillation":
        hour = table["datetime"].dt.tz_localize("UTC").dt.tz_convert(timezone).dt.hour
        keep &= hour.gt(NIGHT_AFTER_HOUR) | hour.lt(NIGHT_BEFORE_HOUR)
    return table[keep].assign(label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feature_dir")
    parser.add_argument("output")
    parser.add_argument("--site", nargs=3, action="append", required=True, metavar=("LABEL", "LAT", "LON"),
                        help="label (Multipath, RFI or Scintillation) and site coordinates; repeat")
    parser.add_argument("--timezone", default="America/Fortaleza", help="local time of the scintillation site")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sites = [(label, float(lat), float(lon)) for label, lat, lon in args.site]
    parts = []
    for path in sorted(glob.glob(os.path.join(args.feature_dir, "*_ml_features.pq"))):
        position = coordinates(os.path.basename(path))
        for label, lat, lon in sites:
            if position and abs(position[0] - lat) <= SITE_TOLERANCE_DEG and abs(position[1] - lon) <= SITE_TOLERANCE_DEG:
                parts.append(label_rows(pd.read_parquet(path), label, args.timezone))
    labeled = pd.concat(parts, ignore_index=True)
    labeled.to_parquet(args.output, index=False)
    logger.info("wrote %d rows to %s: %s", len(labeled), args.output, labeled["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
