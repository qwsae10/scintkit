#!/bin/bash
#SBATCH --job-name=pq2l3p4
#SBATCH --output=sc4logs/%x_%j.out
#SBATCH --error=sc4logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-preempt
unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate /groups/uars/jm-conda

#cd ~/scintkit/tests
cd ~/scintkit/src/scintkit/sc4_reading
python sc4_s03_pq2lvl3_noelev.py
#python sc4_s02_txtF2parquet_sc000.py
