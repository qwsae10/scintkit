#%%
from pathlib import Path
import pandas as pd
import scintkit

data_dir = Path("test_data")
raw_files = sorted(data_dir.glob("scintpi3_*lvl0.pq"))

# make sure files exist
if not raw_files:
    raise FileNotFoundError(
        f"No ScintPi files found in {data_dir.resolve()}"
    )

# quick test on first file
raw_file = raw_files[0]

# run processing
output_files = scintkit.pipelines.auto.process(
    raw_file,
    verbose=True,
    mode="lvl3"
)

# validate output object
assert isinstance(output_files, list), "output is not a list"
assert len(output_files) > 0, "no output files returned"

for outfile in output_files:

    # validate file exists
    path = Path(outfile)
    assert path.exists(), f"{outfile} does not exist"

    # validate parquet can open
    df = pd.read_parquet(path)

    # basic dataframe checks
    assert not df.empty, f"{outfile} is empty"

    print(f"successfully opened: {outfile}")
    print(df.head(2))

print("all tests passed")