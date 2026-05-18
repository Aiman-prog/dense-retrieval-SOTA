#!/usr/bin/env bash

#SBATCH --job-name=grass_seq
#SBATCH --partition=gpu-a100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_seq_%j.out
#SBATCH --error=logs/grass_seq_%j.err
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
# GRASS_N_DAS=30          number of queries mined per mining round
# GRASS_MINE_EVERY=2      mine every N training steps (coverage knob)
# GRASS_SELECTION=bandit  or "random" for ablation baseline
# GRASS_MODEL_SUFFIX=seq_ndas30_me2_bandit   appended to model output dir

mkdir -p logs

echo "🌿 Starting GRASS Sequential Training (1-GPU)..."
echo "   N_DAS=${GRASS_N_DAS:-5} | MINE_EVERY=${GRASS_MINE_EVERY:-2} | SELECTION=${GRASS_SELECTION:-bandit} | SUFFIX=${GRASS_MODEL_SUFFIX:-}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_grass_seq.py \
        ${GRASS_N_DAS:+--n_das $GRASS_N_DAS} \
        ${GRASS_MINE_EVERY:+--mine_every $GRASS_MINE_EVERY} \
        ${GRASS_SELECTION:+--selection $GRASS_SELECTION} \
        ${GRASS_MODEL_SUFFIX:+--model_suffix $GRASS_MODEL_SUFFIX} \
        ${GRASS_NUM_EPOCHS:+--num_epochs $GRASS_NUM_EPOCHS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GRASS sequential training completed successfully"
else
    echo "❌ GRASS sequential training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
