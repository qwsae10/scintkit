#!/bin/bash
#SBATCH --job-name=add_elevaz
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

#SBATCH --output=add_elevaz_%j.out
#SBATCH --error=add_elevaz_%j.err

echo "========================================"
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scintkit_env

# Go to your working directory
cd /home/dal674840/scintkit/src/scintkit/sc4_reading

echo "Working directory:"
pwd

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "Running add_elevaz_pipeline.py..."

python add_elevaz_pipeline.py

echo "========================================"
echo "Finished at $(date)"
echo "========================================"