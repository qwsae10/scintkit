#!/bin/bash
#SBATCH --job-name=scintpi_conv
#SBATCH --output=logs/scintpi_%A_%a.out
#SBATCH --error=logs/scintpi_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu-preempt

mkdir -p logs

echo "job id: $SLURM_JOB_ID"
echo "array task: $SLURM_ARRAY_TASK_ID"
echo "array count: $SLURM_ARRAY_TASK_COUNT"
echo "node: $(hostname)"
echo "cpus per task: $SLURM_CPUS_PER_TASK"

module load miniconda
source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate myenv

python /home/igw180000/run_scintpi_array.py \
    --data-dir /path/to/data \
    --pattern "scintpi3_*.bin*" \
    --mode both