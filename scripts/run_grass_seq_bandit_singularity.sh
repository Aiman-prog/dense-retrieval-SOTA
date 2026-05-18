#!/usr/bin/env bash

#SBATCH --job-name=grass_seq_bandit
#SBATCH --partition=gpu-a100
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_seq_bandit_%j.out
#SBATCH --error=logs/grass_seq_bandit_%j.err
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
# GRASS_SELECTION=bandit         or "random" for baseline
# GRASS_COVERAGE=0.25            fraction of queries mined per epoch (X in Algorithm 1)
# GRASS_NUM_EPOCHS=3
# GRASS_LAMBDA=1.0               gap-index uncertainty weight (overrides config default)
# GRASS_MODEL_SUFFIX=bandit_c25  appended to model output dir
# GRASS_DEBUG=1                  enable train_grass.py --debug (100 queries, tiny config)
#
# Example invocations:
#   GRASS_SELECTION=bandit GRASS_COVERAGE=0.25 GRASS_MODEL_SUFFIX=bandit_c25 sbatch run_grass_seq_bandit_singularity.sh
#   GRASS_SELECTION=random GRASS_COVERAGE=0.25 GRASS_MODEL_SUFFIX=random_c25 sbatch run_grass_seq_bandit_singularity.sh

mkdir -p logs

echo "🌿 Starting GRASS Sequential Bandit Training (1-GPU)..."
echo "   SELECTION=${GRASS_SELECTION:-bandit} | COVERAGE=${GRASS_COVERAGE:-0.25} | EPOCHS=${GRASS_NUM_EPOCHS:-3}"
echo "   LAMBDA=${GRASS_LAMBDA:-cfg-default} | SUFFIX=${GRASS_MODEL_SUFFIX:-}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_grass.py \
        --mode seq_bandit \
        ${GRASS_SELECTION:+--selection $GRASS_SELECTION} \
        ${GRASS_COVERAGE:+--coverage $GRASS_COVERAGE} \
        ${GRASS_NUM_EPOCHS:+--num_epochs $GRASS_NUM_EPOCHS} \
        ${GRASS_LAMBDA:+--lambda_val $GRASS_LAMBDA} \
        ${GRASS_MODEL_SUFFIX:+--model_suffix $GRASS_MODEL_SUFFIX} \
        ${GRASS_DEBUG:+--debug}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GRASS seq-bandit training completed successfully"
else
    echo "❌ GRASS seq-bandit training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
