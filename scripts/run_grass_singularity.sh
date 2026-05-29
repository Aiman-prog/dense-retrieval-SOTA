#!/usr/bin/env bash

#SBATCH --job-name=grass
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # Needed for FAISS operations
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

# Memory & Performance Tuning
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16            # Matches cpus-per-task for FAISS speed

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Pre-flight ---
mkdir -p logs

# --- Run GRASS Pipeline in Container ---
echo "🌿 Starting GRASS Training Loop..."
echo "📋 GRASS: mine hard negatives (stale ANN + MC-dropout) → train → evaluate"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_grass.py \
        ${GRASS_MODE:+--mode $GRASS_MODE} \
        ${GRASS_N_DAS:+--n_das $GRASS_N_DAS} \
        ${GRASS_MODEL_SUFFIX:+--model_suffix $GRASS_MODEL_SUFFIX} \
        ${GRASS_NUM_EPOCHS:+--num_epochs $GRASS_NUM_EPOCHS} \
        ${GRASS_P:+--P $GRASS_P} \
        ${GRASS_L:+--L $GRASS_L}

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
