#!/usr/bin/env bash

#SBATCH --job-name=grass_async_v2
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_async_v2_%j.out
#SBATCH --error=logs/grass_async_v2_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export CC=gcc

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Experiment Knobs (override via env vars before sbatch) ---
# GRASS_V2_M               target mining rounds (config default if unset)
# GRASS_V2_X               coverage fraction (e.g. 0.15, 0.25, 0.50)
# GRASS_V2_SELECTION       "bandit" or "random"
# GRASS_V2_LAMBDA          gap-index lambda (default 1.0)
# GRASS_V2_MODEL_SUFFIX    appended to model output dir (e.g. "M3_X25_bandit")
# GRASS_V2_DEBUG=1         restrict to 100 queries (smoke test)

mkdir -p logs

echo "🌿 Starting GRASS Async v2 Training (2-GPU)..."
echo "   M=${GRASS_V2_M:-cfg} | X=${GRASS_V2_X:-cfg} | SELECTION=${GRASS_V2_SELECTION:-bandit} | LAMBDA=${GRASS_V2_LAMBDA:-cfg} | SUFFIX=${GRASS_V2_MODEL_SUFFIX:-}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_grass_async_v2.py \
        --recipe grass \
        ${GRASS_V2_M:+--M $GRASS_V2_M} \
        ${GRASS_V2_X:+--X $GRASS_V2_X} \
        ${GRASS_V2_SELECTION:+--selection $GRASS_V2_SELECTION} \
        ${GRASS_V2_LAMBDA:+--lambda_val $GRASS_V2_LAMBDA} \
        ${GRASS_V2_MODEL_SUFFIX:+--model_suffix $GRASS_V2_MODEL_SUFFIX} \
        ${GRASS_V2_DEBUG:+--debug}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GRASS async v2 training completed successfully"
else
    echo "❌ GRASS async v2 training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
