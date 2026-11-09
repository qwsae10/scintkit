from pathlib import Path
import shutil
import traceback

from sc4_reading_func import binary_to_clean_txt
from sc4_s02_txtF2parquet import reading__measurements_file
from parquet_reshaping_lvl0 import pq_reshaping_lvl0
import time


def run_pipeline(
    binary_dir,
    txt_dir,
    mearem_dir,
    lvl0_dir,
):
    """
    Complete SC4 processing pipeline

    Binary
        ↓
    TXT
        ↓
    Measurement Parquet
        ↓
    Level-0 Parquet

    Parameters
    ----------
    binary_dir : str or Path
        Directory containing binary files.

    txt_dir : str or Path
        Directory for cleaned txt files.

    mearem_dir : str or Path
        Directory for measurement parquet files.

    lvl0_dir : str or Path
        Directory for level0 parquet files.
    """

    binary_dir = Path(binary_dir)
    txt_dir = Path(txt_dir)
    mearem_dir = Path(mearem_dir)
    lvl0_dir = Path(lvl0_dir)

    txt_dir.mkdir(parents=True, exist_ok=True)
    mearem_dir.mkdir(parents=True, exist_ok=True)
    lvl0_dir.mkdir(parents=True, exist_ok=True)

    binary_files = sorted(binary_dir.glob("*.??_"))

    print(f"\nFound {len(binary_files)} binary files.\n")

    successful = 0
    failed = []

    for i, binary_file in enumerate(binary_files, start=1):

        print("=" * 80)
        print(f"[{i}/{len(binary_files)}] {binary_file.name}")
        
        lvl0_file = lvl0_dir / f"{binary_file.name}_mearem_noelev_lvl0.parquet"

        if lvl0_file.exists():
            print(f"Level0 already exists:")
            print(lvl0_file)
            print("Skipping...")
            successful += 1
            continue

        try:


            txt_file = binary_to_clean_txt(
                binary_file,
                output_dir=txt_dir
            )

            txt_file = Path(txt_file)

            print(f"TXT created:")
            print(txt_file)


            mearem_df, mearem_file = reading__measurements_file(
                str(txt_file)
            )

            mearem_file = Path(mearem_file)

            destination = mearem_dir / mearem_file.name

            if mearem_file != destination:

                shutil.move(
                    str(mearem_file),
                    str(destination)
                )

                mearem_file = destination

            print(f"Measurement parquet:")
            print(mearem_file)

            lvl0_df, lvl0_file = pq_reshaping_lvl0(
                mearem_file,
                output_dir=lvl0_dir
            )

            print(f"Level0 parquet:")
            print(lvl0_file)

            successful += 1

            print("✓ Success")

        except Exception as e:

            failed.append(binary_file.name)

            print("✗ Failed")
            traceback.print_exc()
            print(e)

    print("\n" + "=" * 80)
    print("Pipeline completed")
    print("=" * 80)

    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed)}")

    if failed:

        print("\nFailed files:")

        for file in failed:

            print(file)
 
   