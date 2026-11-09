#!/bin/bash
#SBATCH --job-name=sc4_pipeline
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=sc4_pipeline_%j.out
#SBATCH --error=sc4_pipeline_%j.err

# Activate your conda environment
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate scintkit_env

# Go to your project
cd /home/dal674840/scintkit/src/scintkit/sc4_reading

# Run the pipeline
python bin_to_parquet.py