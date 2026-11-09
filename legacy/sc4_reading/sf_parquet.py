from pathlib import Path
from scintkit.services.convert_to_parquet import process_files

# Input and output folders
input_root = "/home/dal674840/scratch/20241010"
output_root = "/home/dal674840/scratch/20241010/20241010_parquet"

# Get all .bin.zip files recursively
flist = [str(f) for f in Path(input_root).rglob("*.bin.zip")]

print(f"Found {len(flist)} files")

# Process all files
outputs = process_files(
    flist=flist,
    input_root=input_root,
    output_root=output_root,
    n_workers=4,      # Use 1 if you don't want multiprocessing
    overwrite=True,
    verbose=True,
)

print(f"Processed {len(outputs)} files")