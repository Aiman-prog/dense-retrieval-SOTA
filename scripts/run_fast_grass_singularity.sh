#!/usr/bin/env bash

#SBATCH --job-name=fast_grass
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fast_grass_%j.out
#SBATCH --error=logs/fast_grass_%j.err
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
# FAST_GRASS_MODEL_SUFFIX=run1     # appended to model output dir
# FAST_GRASS_NUM_EPOCHS=1          # override config
# FAST_GRASS_B_DOC=100000          # global cache size (ablate 32k / 100k / 512k)
# FAST_GRASS_LAMBDA=1.0            # g = s_hat + lambda * sigma (baseline ablation = 0)
# FAST_GRASS_SELECTION_MODE=topk   # topk | softmax
# FAST_GRASS_M=1                   # negatives per query
# FAST_GRASS_NO_REGISTRY=1         # ablation: fully disable the retired registry R

mkdir -p logs

echo "🌿 Starting Fast-GRASS Training (1-GPU, Algorithm 1 over the global cache)..."
echo "   SUFFIX=${FAST_GRASS_MODEL_SUFFIX:-} | EPOCHS=${FAST_GRASS_NUM_EPOCHS:-cfg} | B_doc=${FAST_GRASS_B_DOC:-cfg} | LAMBDA=${FAST_GRASS_LAMBDA:-cfg} | SELECT=${FAST_GRASS_SELECTION_MODE:-cfg} | M=${FAST_GRASS_M:-cfg} | NO_R=${FAST_GRASS_NO_REGISTRY:-0}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_fast_grass.py \
        ${FAST_GRASS_MODEL_SUFFIX:+--model_suffix $FAST_GRASS_MODEL_SUFFIX} \
        ${FAST_GRASS_NUM_EPOCHS:+--num_epochs $FAST_GRASS_NUM_EPOCHS} \
        ${FAST_GRASS_B_DOC:+--B_doc $FAST_GRASS_B_DOC} \
        ${FAST_GRASS_LAMBDA:+--lambda_val $FAST_GRASS_LAMBDA} \
        ${FAST_GRASS_SELECTION_MODE:+--selection_mode $FAST_GRASS_SELECTION_MODE} \
        ${FAST_GRASS_M:+--m $FAST_GRASS_M} \
        ${FAST_GRASS_NO_REGISTRY:+--no_registry}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Fast-GRASS training completed successfully"
else
    echo "❌ Fast-GRASS training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
