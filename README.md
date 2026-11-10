# scintkit

scintkit converts raw files from ScintPi ionospheric scintillation monitors into
amplitude scintillation, phase scintillation, and total electron content (TEC)
products. It reads ScintPi 3 and ScintPi 4 files and writes Parquet files with
a fixed set of columns.

## Installation

scintkit needs Python 3.10 or newer.

```bash
git clone https://github.com/qwsae10/scintkit.git
cd scintkit
python -m pip install -e .
```

Use `".[dev]"` instead of `.` to also install pytest.

## Quick start

```python
import logging
import pandas as pd
from scintkit.pipeline import process

logging.basicConfig(level=logging.INFO)

process("scintpi3_20261101_0400_96.75784W_32.99184N_v326f.bin.zip", "products")
lvl3 = pd.read_parquet("products/scintpi3_20261101_0400_96.75784W_32.99184N_v326f_lvl3.pq")
```

Each raw file produces three files in the output folder.

| File | Sampling | Content |
|---|---|---|
| `<name>_lvl0.pq` | Receiver rate (10 to 50 Hz) | Raw measurements of every satellite, in UTC |
| `<name>_lvl2.pq` | 1 s | First sample of each second, with the products of its minute |
| `<name>_lvl3.pq` | 1 min | First sample of each minute, scintillation indices, quality flags, and signal class |

The columns, quality flags, and signal classes are defined in
[docs/data_levels.md](docs/data_levels.md). The processing steps are described
in [docs/processing.md](docs/processing.md).

## Processing many files

`process` takes one file or a list of files and returns a dictionary that
lists the files that were written, skipped, failed, or unsupported. Existing
outputs are skipped unless `overwrite=True`. An lvl0 file can be given in
place of a raw file. `process_parallel` does the same on several processes.

```python
from glob import glob
from scintkit.pipeline import process_parallel

summary = process_parallel(glob("/raw/sc000/scintpi3_2026*.bin.zip"), "products", n_workers=4)
```

A 4-hour ScintPi 3 file at 20 Hz contains about 10 million rows. It needs
about 7 GB of memory and 3 to 4 minutes on one core. Each worker needs that
amount of memory, so choose `n_workers` to fit the machine.

## Command line

The `scintkit-process` command writes its outputs to
`<output>/<station folder>/` and exits with status 1 if any file failed. It
has three subcommands.

`files` processes the given files or glob patterns.

```bash
scintkit-process files "/raw/sc000/scintpi3_2026*.bin.zip" --output /products --workers 4
```

`array` processes one share of a file list in a Slurm array job. The task
index, task count, and worker count are read from `SLURM_ARRAY_TASK_ID`,
`SLURM_ARRAY_TASK_COUNT`, and `SLURM_CPUS_PER_TASK`.

```bash
scintkit-process array --pattern "/raw/*/scintpi3_2026*.bin.zip" --output /products
```

`recent` processes the raw files that started within the last `--days` days
and skips files that were already processed. It is meant for a daily cron job.

```bash
scintkit-process recent --archive /raw --output /products --days 1 --workers 4
```

```
30 2 * * * scintkit-process recent --archive /raw --output /products --days 2 --workers 4 >> /products/daily.log 2>&1
```

## Supported receivers

| Receiver | Raw files |
|---|---|
| ScintPi 3 | `.bin` and `.bin.zip`, firmware v325, v326c, v326d, v326e, and v326f |
| ScintPi 4 | Septentrio Binary Format (SBF) files of registered stations |

Files from other firmware versions are skipped and listed as unsupported in
the processing summary. Readers for them are tracked in the
[GitHub issues](https://github.com/qwsae10/scintkit/issues).

ScintPi 4 files are converted with the Septentrio RxTools program `bin2asc`.
The program is looked up in the `SCINTKIT_BIN2ASC` environment variable and
then on `PATH`. A station needs an `SC4 Prefix` in
`src/scintkit/data/station_scintpi_codes.csv` for its files to be recognized.

## Other tools

- `scintkit.stations.identify_station` returns the station registry entry for
  a file name or for a latitude and longitude.
- `scintkit-ml-features` exports the classifier features of each file for
  retraining.
- `python -m scintkit.email_updates.run_pipeline` emails a data availability
  chart. See `src/scintkit/email_updates/readme.md`.

## Repository layout

```
src/scintkit/
  reading.py, sc4.py        raw files to lvl0
  timing.py                 sampling rate and receiver epochs
  formatting.py             constellation, satellite, signal, and frequency columns
  phase.py                  phase detrending and receiver clock removal
  tec.py                    TEC from carrier phase and pseudorange
  amplitude.py              S4 and decorrelation time
  products.py               one-minute products and quality flags
  machine_learning/         classifier features and signal classification
  schema.py                 output columns and file metadata
  pipeline.py               process functions and the command line
  constants.py              lookup tables and thresholds
  stations.py, filenames.py station registry and file name parsing
  email_updates/            data availability email
docs/                       data levels and processing description
examples/                   example scripts and Slurm files
legacy/                     unmaintained scripts kept for reference
tests/                      pytest suite with small raw and lvl0 fixtures
```

## Tests

```bash
python -m pytest tests
```
