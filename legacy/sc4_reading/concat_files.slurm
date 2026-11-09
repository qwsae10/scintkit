#!/bin/bash
#SBATCH --job-name=concat_4hr
#SBATCH --output=concat_4hr_%j.out
#SBATCH --error=concat_4hr_%j.err

#SBATCH --partition=cpu-preempt      # Change if your cluster uses a different partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "====================================="
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "====================================="

# Load modules if required
# module purge
# module load python

# Activate conda environment
source ~/.bashrc
conda activate scintkit_env

# Move to your working directory
cd /home/dal674840/scintkit/src/scintkit/sc4_reading

# Run the script
python concat_files.py

echo "====================================="
echo "Job finished at $(date)"
echo "====================================="