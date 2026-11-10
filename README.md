# scintkit

scintkit converts raw files from ScintPi ionospheric scintillation monitors into
amplitude scintillation, phase scintillation, and total electron content (TEC)
products. The products are written as Parquet files with a fixed column schema.

## Supported receivers

| Receiver | Raw files | Status |
|---|---|---|
| ScintPi 3 | `.bin` and `.bin.zip`, firmware v325, v326c, v326d, v326e, and v326f | Supported |
| ScintPi 4 | Septentrio Binary Format (SBF) files of registered stations | Supported. The Septentrio RxTools program `bin2asc` is required. |

Files from other firmware versions are skipped and listed as unsupported in
the processing summary. Readers for them are tracked in the
[GitHub issues](https://github.com/qwsae10/scintkit/issues).

## Installation

```bash
git clone https://github.com/qwsae10/scintkit.git
cd scintkit
python -m pip install -e .
```

scikit-learn is pinned to 1.6.1, the version used to train the bundled signal
classifier. Use `".[dev]"` to also install pytest.

## Data levels

| Level | Sampling | Content |
|---|---|---|
| lvl0 | Receiver rate (10 to 50 Hz) | Raw measurements of every satellite, in UTC |
| lvl2 | 1 s | First sample of each second, with the products of its minute |
| lvl3 | 1 min | First sample of each minute, scintillation indices, quality flags, and signal class |

The columns, file metadata, and quality flags are defined in
[docs/data_levels.md](docs/data_levels.md). The processing steps are described
in [docs/processing.md](docs/processing.md).

## Processing files in Python

```python
import logging
from scintkit.pipeline import process, process_parallel

logging.basicConfig(level=logging.INFO)

summary = process(["scintpi3_20261101_0400_96.75784W_32.99184N_v326f.bin.zip"], "products")
summary = process_parallel(files, "products", n_workers=4)
```

Each raw file produces `<name>_lvl0.pq`, `<name>_lvl2.pq`, and `<name>_lvl3.pq`
in the output directory. Existing outputs are skipped unless `overwrite=True`.
The returned dictionary lists the files that were written, skipped, failed, or
unsupported. An lvl0 Parquet file can be given in place of a raw file.

## Command line

Outputs are written to `<output>/<station folder>/`. The command exits with
status 1 if any file failed.

```bash
scintkit-process files "/raw/sc000/scintpi3_2026*.bin.zip" --output /products --workers 4
```

```bash
scintkit-process array --pattern "/raw/*/scintpi3_2026*.bin.zip" --output /products
```

The `array` subcommand processes the share of the file list that belongs to
the current Slurm array task. The task index, task count, and worker count are
read from `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_COUNT`, and
`SLURM_CPUS_PER_TASK`.

## Daily processing

The `recent` subcommand processes the raw files that started within the last
`--days` days and skips the files that were already processed. It can be run
from cron.

```bash
scintkit-process recent --archive /raw --output /products --days 1 --workers 4
```

```
30 2 * * * scintkit-process recent --archive /raw --output /products --days 2 --workers 4 >> /products/daily.log 2>&1
```

The same operation is available in Python as
`scintkit.pipeline.process_recent(archive_root, output_root, days=1)`.

## Computing requirements

A 4-hour ScintPi 3 file at 20 Hz contains about 10 million rows. It requires
about 7 GB of memory and 3 to 4 minutes on one core. Each worker of
`process_parallel` needs that amount of memory.

## ScintPi 4

ScintPi 4 files are converted with `bin2asc`. The program is searched for in
the `SCINTKIT_BIN2ASC` environment variable and then on `PATH`. A station must
have an `SC4 Prefix` in `src/scintkit/data/station_scintpi_codes.csv` for its
files to be recognized.

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
