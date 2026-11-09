from pathlib import Path
import pandas as pd
from bin_to_parquet import run_pipeline
from add_elevaz_pipeline import run_sp3_pipeline
from lvl3_pipeline import run_lvl3_pipeline
from concat_4hr_files import merge_4hr_parquet
from SP3_download_func import download_sp3_files

def get_input():
    """Get user input."""
    binary_dir = input("Enter the path to the binary files: ").strip()
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    return binary_dir, start_date, end_date


def has_elevation_azimuth(parquet_dir):
    """
    Check whether all Level-0 parquet files already contain
    elevation and azimuth columns.
    """
    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        return False

    elev_cols = {"elev"}
    azim_cols = {"azim"}

    for file in parquet_files:
        # Read only the schema (fast)
        cols = set(pd.read_parquet(file, engine="pyarrow").columns)

        has_elev = bool(elev_cols & cols)
        has_azim = bool(azim_cols & cols)

        if not (has_elev and has_azim):
            print(f"Missing elevation/azimuth columns in {file.name}")
            return False

    return True


def main():
    binary_dir, start_date, end_date = get_input()

    binary_dir = Path(binary_dir)

    txt_dir = binary_dir / "txt"
    mearem_dir = binary_dir / "mearem"
    lvl0_dir = binary_dir / "lvl0_noelev"
    lvl0_dir_welev= binary_dir / "lvl0_welev"
    concat_lvl0 = binary_dir / "concat_lvl0"
    lvl3_dir = binary_dir / "concat_lvl3"
    lvl2_dir = binary_dir / "concat_lvl2"
    sp3_dir = binary_dir

    print(f"\nBinary directory: {binary_dir}")

    # Binary -> TXT -> Mearem -> Level-0 
    run_pipeline( 
        binary_dir=binary_dir,
        txt_dir=txt_dir,
        mearem_dir=mearem_dir,
        lvl0_dir=lvl0_dir,
    )
    print("Binary to Parquet pipeline completed.")


    # Check if elevation/azimuth already exist
    if has_elevation_azimuth(lvl0_dir):
        print("\nElevation/Azimuth columns already found.")
        print("Skipping SP3 download and SP3 pipeline.")

        lvl0_input = None

    else:
        print("\nElevation/Azimuth columns not found.")
        print("Downloading SP3 files...")

        download_sp3_files(
            start_date=start_date,
            end_date=end_date,
            output_dir=sp3_dir,
        )

        print("SP3 files downloaded successfully.")

        run_sp3_pipeline(
            parquet_dir=lvl0_dir,
            sp3_dir=sp3_dir,
            output_dir=lvl0_dir_welev,
        )

        print("Elevation and Azimuth added to the parquet files.")

        lvl0_input = lvl0_dir_welev

    merge_4hr_parquet(
        lvl0_input, concat_lvl0,)
    # Level-3 pipeline
    run_lvl3_pipeline(
        lvl0_dir=concat_lvl0,
        lvl2_dir=lvl2_dir,
        lvl3_dir=lvl3_dir,
        mode="both",
        verbose=True,
    )

    print("Level-3 pipeline completed.")


if __name__ == "__main__":
    main()