#!/usr/bin/env bash

#SBATCH --job-name=grass
#SBATCH --partition=gpu-a100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_%j.out
#SBATCH --error=logs/grass_%j.err
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
# GRASS_UNCERTAINTY=mc_dropout    # Algorithm 2 σ estimator (mc_dropout | ema)
# GRASS_MODEL_SUFFIX=run1          # appended to model output dir
# GRASS_NUM_EPOCHS=3              # override config

mkdir -p logs

echo "🌿 Starting GRASS Training (1-GPU, Algorithm 1)..."
echo "   UNCERTAINTY=${GRASS_UNCERTAINTY:-mc_dropout} | SUFFIX=${GRASS_MODEL_SUFFIX:-} | EPOCHS=${GRASS_NUM_EPOCHS:-cfg}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_grass.py \
        ${GRASS_UNCERTAINTY:+--uncertainty $GRASS_UNCERTAINTY} \
        ${GRASS_MODEL_SUFFIX:+--model_suffix $GRASS_MODEL_SUFFIX} \
        ${GRASS_NUM_EPOCHS:+--num_epochs $GRASS_NUM_EPOCHS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GRASS training completed successfully"
else
    echo "❌ GRASS training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
