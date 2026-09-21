# ScintKit Email Updates Module 📡

`scintkit.email_updates` is a submodule of the `scintkit` package that automatically scans, plots, and reports the data availability of ScintPi (Scintillation Pi) ionospheric monitoring stations. It supports legacy ScintPi 2/3 formats, the newer ScintPi 4, and the UTD Septentrio receiver.

## Directory Structure

```text
src/scintkit/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── station_scintpi_codes.csv
├── email_updates/
│   ├── __init__.py
│   ├── core.py
│   ├── plotting.py
│   ├── mailer.py
│   └── run_pipeline.py
├── pipelines/
├── preprocessing/
├── reading/
└── services/
```

## Installation

This module is included when you install `scintkit`. From the project root (where `pyproject.toml` is located):

```bash
pip install -e .
```

This will install `scintkit` and all of its dependencies (`numpy`, `pandas`, `matplotlib`, `h5py`, etc.).

## Security & SMTP Credentials ⚠️

**DO NOT hard-code your email passwords into the code.** 

This module uses environment variables to handle SMTP credentials securely. Before running the pipeline, you must set the following environment variables on your system:

* `SMTP_USER`: The email address you are sending from (e.g., your Gmail).
* `SMTP_PASS`: Your email password (if using Gmail, this should be an **App Password**, not your main account password).
* `SMTP_SENDER`: The alias or email address you want to appear in the "From" field.

### Setting variables on Linux/macOS:

```bash
export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export SMTP_SENDER="ScintPi Bot <scintpi-bot@example.com>"
```

### Setting variables on Windows (Command Prompt):

```cmd
set SMTP_USER=your-email@gmail.com
set SMTP_PASS=your-app-password
set SMTP_SENDER=ScintPi Bot <scintpi-bot@example.com>
```

*(Alternatively, you can use a `.env` file and the `python-dotenv` library to load these variables automatically, but ensure `.env` is listed in your `.gitignore`!)*

## Usage

### Running the pipeline script

Once installed and your environment variables are set, run the pipeline:

```bash
python -m scintkit.email_updates.run_pipeline
```

### Importing in your own code

The maintained station registry lives at
`src/scintkit/data/station_scintpi_codes.csv`. It is bundled with the package
and loaded automatically, so installed code does not depend on a repository
path:

```python
from scintkit.email_updates import load_targets, scan_legacy_files, generate_availability_plot

# Loads the bundled CSV automatically
targets = load_targets()

# Or pass a custom CSV path
targets = load_targets("/path/to/custom_stations.csv")
```

Code outside the email updates module can load the shared station table directly:

```python
from scintkit.data import load_station_codes

stations = load_station_codes()
```

### Available functions

| Function | Description |
|---|---|
| `load_targets(csv_path=None)` | Load station targets from CSV (defaults to bundled CSV) |
| `scan_legacy_files(targets, cutoff)` | Scan ScintPi 2/3 data files |
| `scan_sc4_files(targets, cutoff)` | Scan ScintPi 4.0 data files using CSV prefixes |
| `scan_septentrio_files(targets, cutoff)` | Scan UTD Septentrio `CSS_DDDN.YY_` files |
| `generate_availability_plot(targets, cutoff, now, output_path)` | Generate availability PNG plot |
| `send_status_email(image_path, now_date, to_list)` | Email the plot to recipients |
| `checklvl3datamissing(lvl3file, thres=900)` | Check percent missing from Level-3 HDF5 file |

### Full pipeline example

```python
import pandas as pd
from scintkit.email_updates import (
    load_targets,
    scan_legacy_files,
    scan_sc4_files,
    scan_septentrio_files,
    generate_availability_plot,
    send_status_email,
)

now = pd.Timestamp.now()
cutoff = now - pd.DateOffset(months=3)

targets = load_targets()
scan_legacy_files(targets, cutoff)
scan_sc4_files(targets, cutoff)
scan_septentrio_files(targets, cutoff)

generate_availability_plot(targets, cutoff, now, "availability.png")
send_status_email("availability.png", now, ["recipient@example.com"])
```

This will:
1. Load the station targets from the bundled CSV.
2. Scan the file system for legacy, SC4, and UTD Septentrio data.
3. Generate an availability `.png` plot.
4. Email the plot to the configured recipients.
