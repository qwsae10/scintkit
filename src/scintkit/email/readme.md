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

## Security & Environment Variables ⚠️

**DO NOT hard-code your email passwords into the code.** 

This project uses environment variables to handle SMTP credentials securely. Before running the pipeline, you must set the following environment variables on your system:

* `SMTP_USER`: The email address you are sending from (e.g., your Gmail).
* `SMTP_PASS`: Your email password (if using Gmail, this should be an **App Password**, not your main account password).
* `SMTP_SENDER`: The alias or email address you want to appear in the "From" field.

### Setting variables on Linux/macOS:

export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export SMTP_SENDER="ScintPi Bot <scintpi-bot@example.com>"


### Setting variables on Windows (Command Prompt):

set SMTP_USER=your-email@gmail.com
set SMTP_PASS=your-app-password
set SMTP_SENDER=ScintPi Bot <scintpi-bot@example.com>


*(Alternatively, you can use a `.env` file and the `python-dotenv` library to load these variables automatically, but ensure `.env` is listed in your `.gitignore`!)*

## Usage

Once installed and your environment variables are set, simply run the pipeline script from the terminal:


python run_pipeline.py

This will:
1. Load the targets from the CSV.
2. Scan the file system for legacy and SC4 data.
3. Generate an availability `.png` plot.
4. Email the plot to the configured recipients.
