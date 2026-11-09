from pathlib import Path
import pandas as pd

# Folder containing parquet files
data_dir = Path("/home/dal674840/scratch/20240629/20240629_parquet")

# Read all files for the station
files = sorted(data_dir.glob("*19.2235E_34.4244S*_lvl0.pq"))

print(f"Found {len(files)} files")

dfs = []

for f in files:
    print(f"Reading {f.name}")

    df = pd.read_parquet(f)

    # Convert to datetime if needed
    df["datetime"] = pd.to_datetime(df["datetime"])

    dfs.append(df)

# Concatenate all files
day_df = pd.concat(dfs, ignore_index=True)

# Sort chronologically
day_df = day_df.sort_values("datetime").reset_index(drop=True)

# Optional: remove duplicate rows
day_df = day_df.drop_duplicates()

output_file = data_dir / "scintpi3_20240629_day_19.2235E_34.4244S_lvl0.pq"

day_df.to_parquet(
    output_file,
    index=False,
    compression="brotli",
    compression_level=6,
)

print(f"Saved: {output_file}")