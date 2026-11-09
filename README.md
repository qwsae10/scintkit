# scintkit

Simple tools for working with ScintPi and GNSS scintillation data.

---

## Overview

This repository provides utilities and pipelines for:

- converting raw binary data to Parquet
- adding derived products (TEC, S4, sigma_phi, etc.)
- running processing workflows
- plotting ScintPi data

The Level 3 product schema and automatic quality-flag rules are documented in
[`docs/lvl3.md`](docs/lvl3.md).

---

## Project Layout


```
src/scintkit/
  constants.py     lookup tables and processing thresholds
  filenames.py     firmware version, start time and coordinates from file names
  stations.py      station registry lookup (CSV in data/)
  reading.py       binary readers and raw -> lvl0 Parquet conversion
  formatting.py    constellation, PRN, signal and frequency columns
  phase.py         phase detrending, cycle-slip repair, receiver clock
  tec.py           carrier-phase and pseudorange TEC
  amplitude.py     SNR detrending, S4, tau
  products.py      per-minute products and quality flags
  pipeline.py      process(), process_parallel() and the scintkit-process CLI
  plotting.py      summary plots
tests/             pytest suite and a 15-minute lvl0 fixture
```

---

## Installation

### 1. Get the repository

Clone with git:

```bash
git clone https://github.com/qwsae10/scintkit.git
cd scintkit
````

Or download as a ZIP from GitHub and extract it.

---

### 2. Install (recommended)

Editable install for development:

```bash
python -m pip install -e .
```

This makes the package importable as `scintkit`.

---

## Usage


You can import and run core processing functions directly.


```python
from scintkit.pipeline import process

summary = process(["example.bin.zip"], "output")
```

This will:

1. convert the raw file to `output/example_lvl0.pq` (UTC times)
2. decode signals, detect the sampling rate and fix receiver epochs
3. compute derived products
4. write `output/example_lvl2.pq` (1 s) and `output/example_lvl3.pq` (1 min)

`summary` lists the files written and the inputs that failed or have an
unsupported firmware (supported: v325, v326c, v326d, v326e, v326f). Column
definitions are in `scintkit.schema.COLUMNS`. Progress is reported through
`logging`; call `logging.basicConfig(level=logging.INFO)` to see it.

### Command line and Slurm arrays

Outputs go to `<output>/<input folder name>/`, e.g. `/products/sc000/`.
The command exits with status 1 if any file failed.

```bash
# explicit files or globs
scintkit-process files "/raw/sc000/scintpi3_2024*.bin.zip" --output /products --workers 2

# this Slurm array task's share of a glob
scintkit-process array --pattern "/raw/*/scintpi3_2024*.bin.zip" --output /products

# raw files from the last day, skipping those already processed (daily cron)
scintkit-process recent --archive /raw --output /products --days 1 --workers 2
```

`--task-id`, `--n-tasks` and `--workers` default to `SLURM_ARRAY_TASK_ID`,
`SLURM_ARRAY_TASK_COUNT` and `SLURM_CPUS_PER_TASK` (or 0, 1, 1 outside Slurm).
A 4 h file needs about 20 GB of memory per worker.

### Identifying a station

The maintained station registry can identify a receiver from coordinates:

```python
from scintkit.stations import identify_station

station = identify_station(latitude=32.9919, longitude=-96.7573)
print(station["Code"])  # US-TX1
```

It can also read legacy coordinate filenames or SC4 station prefixes:

```python
station = identify_station(
    filename="scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
)
```

The function returns the matching CSV row as a dictionary. It returns `None`
when no station is within 3 km. Pass `max_distance_km=` to change that limit.

---

## Tutorial

See:

```
tutorials/compare_oct11.ipynb
```

This notebook shows how to:

* load raw data
* run processing steps
* compare outputs

---
