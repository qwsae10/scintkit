from pathlib import Path
import shutil
import gc
import traceback

from scintkit.pipelines.auto import process


def run_lvl3_pipeline(
    lvl0_dir,
    lvl2_dir,
    lvl3_dir,
    mode="both",
    verbose=True,
):
    """
    Process all Level-0 parquet files using ScintKit.

    Parameters
    ----------
    lvl0_dir : str or Path
        Folder containing *_lvl0.parquet files.

    lvl2_dir : str or Path
        Folder where Level-2 files will be saved.

    lvl3_dir : str or Path
        Folder where Level-3 files will be saved.

    mode : str
        "lvl2", "lvl3", or "both"

    verbose : bool
        Print progress messages.
    """

    lvl0_dir = Path(lvl0_dir)
    lvl2_dir = Path(lvl2_dir)
    lvl3_dir = Path(lvl3_dir)

    lvl2_dir.mkdir(parents=True, exist_ok=True)
    lvl3_dir.mkdir(parents=True, exist_ok=True)

    lvl0_files = sorted(lvl0_dir.glob("*_lvl0.parquet"))

    print("=" * 80)
    print(f"Found {len(lvl0_files)} Level-0 files.")
    print("=" * 80)

    successful = 0
    failed = []

    for i, parquet_file in enumerate(lvl0_files, start=1):

        print("\n" + "=" * 80)
        print(f"[{i}/{len(lvl0_files)}] {parquet_file.name}")
        
        lvl3_file = lvl3_dir / parquet_file.name.replace(
        "_lvl0.parquet",
        "_lvl3.parquet")

        if lvl3_file.exists():
            print(f"Level-3 already exists:")
            print(lvl3_file)
            print("Skipping...")
            successful += 1
            continue

    

        try:

            outputs = process(
                str(parquet_file),
                verbose=verbose,
                mode=mode,
            )

            if outputs is None or len(outputs) == 0:
                print("No output files created.")
                failed.append(parquet_file.name)
                continue

            for outfile in outputs:

                outfile = Path(outfile)

                if not outfile.exists():
                    continue

                if "_lvl2" in outfile.name:

                    destination = lvl2_dir / outfile.name

                    shutil.move(
                        str(outfile),
                        str(destination)
                    )

                    print(f"Saved Level-2 : {destination}")

                elif "_lvl3" in outfile.name:

                    destination = lvl3_dir / outfile.name

                    shutil.move(
                        str(outfile),
                        str(destination)
                    )

                    print(f"Saved Level-3 : {destination}")

            successful += 1

            print("✓ Success")

        except Exception:

            print("✗ Failed")

            traceback.print_exc()

            failed.append(parquet_file.name)

        finally:

            gc.collect()

    print("\n" + "=" * 80)
    print("Pipeline completed")
    print("=" * 80)

    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed)}")

    if failed:

        print("\nFailed files:")

        for f in failed:

            print(f)

