#!/bin/bash
# Export classifier feature tables (one <stem>_ml_features.pq per 20 Hz file) for retraining.
# Each array task takes every N-th file of INPUT_GLOB. Edit the variables, then: sbatch ml_features.slurm
#SBATCH --job-name=scintkit_ml_features
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=ml_features_%A_%a.out

INPUT_GLOB="/path/to/scintpi_storage/*/*2024*.pq"   # raw or lvl0 files
OUTPUT_DIR="/path/to/ml_features"
CONDA_ENV="scintkit"                                 # with scintkit installed

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t FILES < <(ls $INPUT_GLOB | awk -v n="$SLURM_ARRAY_TASK_COUNT" -v k="$SLURM_ARRAY_TASK_ID" 'NR % n == k')
scintkit-ml-features "${FILES[@]}" --output "$OUTPUT_DIR" --workers "$SLURM_CPUS_PER_TASK"
