# ScintKit 📡

`scintkit` is a Python library designed to automatically scan, plot, and report the data availability of ScintPi (Scintillation Pi) ionospheric monitoring stations. It supports legacy ScintPi 2/3 formats as well as the newer ScintPi 4.
## Directory Structure

\`\`\`text
scintkit_project/
├── .gitignore
├── README.md
├── run_pipeline.py
├── station_scintpi_codes_fsr.csv
├── setup.py
└── scintkit/
    ├── __init__.py
    ├── core.py
    ├── plotting.py
    └── mailer.py
\`\`\`

## Installation

You can install this package locally in editable mode. Navigate to the root directory (where `setup.py` is located) and run:

\`\`\bash
pip install -e .
\`\`\`

This will install `scintkit` and its dependencies (`numpy`, `pandas`, `matplotlib`, `h5py`).

## Security & SMTP Credentials ⚠️

For now, the email helper uses hardcoded example credentials at the top of the mailer module:

* `SMTP_USER = "use"`
* `SMTP_PASS = "pass"`
* `SMTP_SENDER = "use"`

Replace those values directly in the code before using this for a real mailbox.

## Usage

Once installed and your environment variables are set, simply run the pipeline script from the terminal:


python run_pipeline.py

This will:
1. Load the targets from the CSV.
2. Scan the file system for legacy and SC4 data.
3. Generate an availability `.png` plot.
4. Email the plot to the configured recipients.
