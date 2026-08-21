#!/usr/bin/env bash

#SBATCH --job-name=refresh-stale-index
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/refresh_stale_%j.out
#SBATCH --error=logs/refresh_stale_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "🔄 Refreshing Fast-GRASS stale index (re-encoding corpus from InBatch checkpoint)..."

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/refresh_stale_index.py ${REFRESH_MODEL:+--model "$REFRESH_MODEL"}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Stale index refreshed"
else
    echo "❌ Stale index refresh failed with code $EXIT_CODE"
fi

exit $EXIT_CODE
