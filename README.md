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
src/scintkit/reading/         # binary readers and data loading
src/scintkit/preprocessing/   # formatting and preprocessing
src/scintkit/services/        # core computations (TEC, S4, phase detrending)
src/scintkit/utils/           # helper utilities
src/scintkit/pipelines/       # end-to-end processing pipelines
tests/                        # test scripts and example notebooks

```

---

## Installation

### 1. Get the repository

Clone with git:

```bash
git clone https://github.com/qwsae10/scintkit.git
cd scintkit
```

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
from scintkit.pipelines.auto import process

process("example.bin.zip")
```

This will:

1. convert raw data to parquet
2. apply preprocessing
3. compute derived products
4. output lvl3 files

### Identifying a station

The maintained station registry can identify a receiver from coordinates:

```python
from scintkit.data import identify_station

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
examples/compare_oct11.ipynb
```

This notebook shows how to:

* load raw data
* run processing steps
* compare outputs

---
