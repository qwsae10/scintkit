from pathlib import Path
import pandas as pd

from add_elevaz import sp3_merge_lvl3


def run_sp3_pipeline(
    parquet_dir,
    sp3_dir,
    output_dir,
):
    """
    Apply SP3 merge to every Level-0 parquet file.
    """

    parquet_dir = Path(parquet_dir)
    sp3_dir = Path(sp3_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*_lvl0.parquet"))

    print(f"\nFound {len(parquet_files)} parquet files.\n")

    success = 0
    failed = []

    for i, parquet_file in enumerate(parquet_files, start=1):

        print("=" * 80)
        print(f"[{i}/{len(parquet_files)}] {parquet_file.name}")

        expected_name = parquet_file.name.replace("_noelev_", "_welev_")

        existing_files = list(output_dir.rglob(expected_name))

        print(f"Looking for: {expected_name}")
        print(f"Found: {existing_files}")

        if existing_files:
            print("Level0 welev file already exists:")
            print(existing_files[0])
            print("Skipping...")
            success += 1
            continue

        try:

            df = pd.read_parquet(
                parquet_file,
                columns=["timestamp"]
            )

            first_time = pd.to_datetime(df["timestamp"].iloc[0])

            year = first_time.year
            doy = first_time.dayofyear

            sp3_file = (
                sp3_dir /
                f"IAC0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3"
            )

            if not sp3_file.exists():
                raise FileNotFoundError(
                    f"SP3 file not found:\n{sp3_file}"
                )

            print(f"Using SP3 : {sp3_file.name}")

            unified_df, outfile = sp3_merge_lvl3(
                parquet_file=parquet_file,
                sp3_file=sp3_file,
                output_dir=output_dir,
            )

            print(f"✓ Saved : {outfile.name}")

            success += 1

        except Exception as e:

            print("✗ Failed")
            print(e)

            failed.append(parquet_file.name)

    print("\n" + "=" * 80)
    print("Pipeline completed")
    print("=" * 80)

    print(f"Successful : {success}")
    print(f"Failed     : {len(failed)}")

    if failed:
        print("\nFailed files:")

        for file in failed:
            print(file)

