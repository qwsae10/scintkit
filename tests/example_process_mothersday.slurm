#!/bin/bash
#SBATCH --job-name=scintkit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --partition=rodrigues
unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate scintkit

cd ~/scintkit/tests

python example_process_mothersday_process.py


