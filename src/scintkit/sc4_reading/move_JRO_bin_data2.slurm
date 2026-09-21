#!/bin/bash
#SBATCH --job-name=movingdata1
#SBATCH --output=sc4logs/%x_%j.out
#SBATCH --error=sc4logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=rodrigues
unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate /groups/uars/jm-conda

#cd ~/scintkit/tests
mv /mfs/io/groups/uars/scintpi/sc002/scintpi*_202[4-6]*.zip /titan/frodrigues/scintpi_storage/sc002_bin_files/
#python sc4_s02_txtF2parquet_sc000.py
