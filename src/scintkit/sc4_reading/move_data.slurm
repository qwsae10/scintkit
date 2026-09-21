#!/bin/bash
#SBATCH --job-name=movingdata
#SBATCH --output=sc4logs/%x_%j.out
#SBATCH --error=sc4logs/%x_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=rodrigues
unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate /groups/uars/jm-conda

#cd ~/scintkit/tests
cp /titan/frodrigues/scintpi_storage/sc000_pq_files/sept*_F.parquet ~/scratch/sc4_sc000_tmp/
#python sc4_s02_txtF2parquet_sc000.py
