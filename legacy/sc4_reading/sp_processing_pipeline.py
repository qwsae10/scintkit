from pathlib import Path

from sc4_reading_func import binary_to_clean_txt
from sc4_s02_txtF2parquet import reading__measurements_file
from parquet_reshaping_lvl0 import pq_reshaping_lvl0


def run_pipeline(
    binary_dir,
    txt_dir,
    mearem_dir,
    lvl0_dir):
    """
    Complete SC4 processing pipeline.

    binary_dir
        Folder containing binary files.

    txt_dir
        Folder where txt files will be written.

    mearem_dir
        Folder where mearem parquet files will be written.

    lvl0_dir
        Folder where lvl0 parquet files will be written.
    """

    binary_dir = Path(binary_dir)
    txt_dir = Path(txt_dir)
    mearem_dir = Path(mearem_dir)
    lvl0_dir = Path(lvl0_dir)

    txt_dir.mkdir(exist_ok=True, parents=True)
    mearem_dir.mkdir(exist_ok=True, parents=True)
    lvl0_dir.mkdir(exist_ok=True, parents=True)

    binary_files = sorted(binary_dir.glob("*.??_"))

    print(f"Found {len(binary_files)} binary files\n")

    for i, binary_file in enumerate(binary_files, start=1):

        print("="*70)
        print(f"[{i}/{len(binary_files)}] {binary_file.name}")

        try:


            txt_file = binary_to_clean_txt(
                binary_file,
                output_dir=txt_dir
            )


            mearem_file = reading__measurements_file(
                txt_file,
                output_dir=mearem_dir
            )



            lvl0_df, lvl0_file = pq_reshaping_lvl0(
                mearem_file,
                output_dir=lvl0_dir
            )

            print("✓ Finished")

        except Exception as e:

            print("✗ Failed")
            print(e)

    print("\nPipeline completed.")

run_pipeline(
    binary_dir="/home/dal674840/scratch/mx02_data",
    txt_dir="/home/dal674840/scratch/mx02_txt_files",
    mearem_dir="/home/dal674840/scratch/mx02_mearem_parquet",
    lvl0_dir="/home/dal674840/scratch/mx02_lvl0_parquet"
)