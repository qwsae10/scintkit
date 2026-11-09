from pathlib import Path
import pandas as pd
import re

input_dir = Path("/home/dal674840/scratch/mx02_lvl0_welev_parquet")
output_dir = Path("/home/dal674840/scratch/mx02_4hr_parquet_lvl0")
output_dir.mkdir(exist_ok=True)

blocks = [
    ["a", "b", "c", "d"],   # 00-04
    ["e", "f", "g", "h"],   # 04-08
    ["i", "j", "k", "l"],   # 08-12
    ["m", "n", "o", "p"],   # 12-16
    ["q", "r", "s", "t"],   # 16-20
    ["u", "v", "w", "x"],   # 20-24
]


files = list(input_dir.glob("*.parquet"))

day_dict = {}

for f in files:

    # Example:
    # mx01316a15.25__mearem_welev_lvl0_lvl0.parquet

    m = re.match(r"mx02(\d{3})([a-x])(\d{2})", f.name)

    if m is None:
        continue

    doy = m.group(1)      # 316
    letter = m.group(2)   # a
    minute = m.group(3)   # 15

    day_dict.setdefault(doy, {})
    day_dict[doy].setdefault(letter, {})
    day_dict[doy][letter][minute] = f


minute_order = ["00", "15", "30", "45"]

for doy in sorted(day_dict.keys()):

    print(f"\nProcessing DOY {doy}")

    for block_id, letters in enumerate(blocks):

        dfs = []

        print(f"  Block {block_id+1}: {letters}")

        for letter in letters:
            for minute in minute_order:

                try:
                    file = day_dict[doy][letter][minute]
                except KeyError:
                    print(f"Missing {letter}{minute}")
                    continue

                print(f"    {file.name}")
                dfs.append(pd.read_parquet(file))

        if len(dfs) == 0:
            continue

        merged = pd.concat(dfs, ignore_index=True)

        start_hour = block_id * 4
        end_hour = start_hour + 4

        outfile = output_dir / f"DOY{doy}_{start_hour:02d}-{end_hour:02d}UTC_lvl0.parquet"

        merged.to_parquet(outfile, index=False)

        print(f"Saved {outfile.name} ({len(merged):,} rows)")