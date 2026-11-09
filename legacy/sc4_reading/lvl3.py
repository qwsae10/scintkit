from lvl3_pipeline import run_lvl3_pipeline

file = "/home/dal674840/scratch/jul03_binary/concat_lvl0"
file1 = "/home/dal674840/scratch/jul03_binary/concat_lvl2"
file2 = "/home/dal674840/scratch/jul03_binary/concat_lvl3"

run_lvl3_pipeline(
        lvl0_dir=file,
        lvl2_dir=file1,
        lvl3_dir=file2,
        mode="both",
        verbose=True,
    )