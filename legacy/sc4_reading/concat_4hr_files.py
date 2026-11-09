from pathlib import Path
import pandas as pd
import re


def merge_4hr_parquet(input_dir, output_dir):
    """
    Merge 15-minute parquet files into 4-hour parquet files.

    Expected filename format:
        xxNNDDDlMM....

    where
        xx   -> two letters
        NN   -> two digits (receiver number)
        DDD  -> day of year
        l    -> a-x
        MM   -> 00,15,30,45

    Examples
    --------
    mx01316a00...
    mx02316b15...
    sc04312m30...
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = [
        ["a", "b", "c", "d"],
        ["e", "f", "g", "h"],
        ["i", "j", "k", "l"],
        ["m", "n", "o", "p"],
        ["q", "r", "s", "t"],
        ["u", "v", "w", "x"],
    ]

    minute_order = ["00", "15", "30", "45"]

    pattern = re.compile(
        r"([A-Za-z]{2}\d{2})(\d{3})([a-x])(\d{2})",
        re.IGNORECASE,
    )

    grouped = {}

    for file in input_dir.glob("*.parquet"):

        m = pattern.match(file.stem)

        if m is None:
            continue

        station = m.group(1)      # mx01
        doy = m.group(2)          # 316
        letter = m.group(3).lower()
        minute = m.group(4)

        key = (station, doy)

        grouped.setdefault(key, {})
        grouped[key].setdefault(letter, {})
        grouped[key][letter][minute] = file

    for (station, doy) in sorted(grouped):

        print(f"\nProcessing {station} DOY {doy}")

        day_files = grouped[(station, doy)]

        for block_id, letters in enumerate(blocks):

            start_hour = block_id * 4
            end_hour = start_hour + 4

            outfile = (
                output_dir
                / f"{station}_DOY{doy}_{start_hour:02d}-{end_hour:02d}UTC_lvl0.parquet"
            )

            if outfile.exists():
                print(f"  {outfile.name} already exists. Skipping...")
                continue

            dfs = []

            print(f"  Block {block_id + 1}: {letters}")

            for letter in letters:

                for minute in minute_order:

                    file = day_files.get(letter, {}).get(minute)

                    if file is None:
                        print(f"    Missing {letter}{minute}")
                        continue

                    print(f"    {file.name}")

                    dfs.append(pd.read_parquet(file))

            if not dfs:
                continue

            merged = pd.concat(dfs, ignore_index=True)

            merged.to_parquet(outfile, index=False)


            print(f"Saved {outfile.name} ({len(merged):,} rows)")