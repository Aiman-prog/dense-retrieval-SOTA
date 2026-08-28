#!/usr/bin/env bash

#SBATCH --job-name=inbatch-reasonir
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/inbatch_neg_%j.out
#SBATCH --error=logs/inbatch_neg_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory Management
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Experiment Knobs (override via env vars before sbatch) ---
# INBATCH_RESUME=1        # continue a run whose manifest fingerprint matches
# INBATCH_OVERWRITE=1     # discard an output dir built by a DIFFERENT config
# Default (both unset) starts FRESH: stale checkpoint-* are removed first, which is
# what stops Tevatron resuming them and reporting success after zero steps.

# --- Run Training in Container ---
echo "🚀 Starting In-Batch Negatives Training..."

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_inbatch.py \
        ${INBATCH_RESUME:+--resume} \
        ${INBATCH_OVERWRITE:+--overwrite}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ In-batch training completed successfully"
else
    echo "❌ In-batch training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
