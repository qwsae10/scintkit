# scintkit

Simple tools for working with ScintPi and GNSS scintillation data.

---

## Overview

This repository provides utilities and pipelines for:

- converting raw binary data to Parquet
- adding derived products (TEC, S4, sigma_phi, etc.)
- running processing workflows
- plotting ScintPi data

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
from scintkit.pipelines.auto import process

process("example.bin.zip")
```

This will:

1. convert raw data to parquet
2. apply preprocessing
3. compute derived products
4. output lvl3 files

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
