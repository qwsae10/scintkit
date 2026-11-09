#!/bin/bash
#SBATCH --job-name=lvl3_pipeline
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --output=lvl3_pipeline_%j.out
#SBATCH --error=lvl3_pipeline_%j.err

echo "========================================"
echo "Job started on $(hostname)"
echo "Start time : $(date)"
echo "========================================"

# Activate conda
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate scintkit_env

# Change to your project directory
cd /home/dal674840/scintkit/src/scintkit/sc4_reading

echo "Current directory:"
pwd

echo "Python executable:"
which python

echo "Python version:"
python --version

echo "Running Level-3 pipeline..."

python lvl3_pipeline.py

echo "========================================"
echo "Finished at $(date)"
echo "========================================"