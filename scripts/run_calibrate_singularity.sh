#!/usr/bin/env bash

#SBATCH --job-name=grass_calibrate
#SBATCH --partition=gpu-a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_calibrate_%j.out
#SBATCH --error=logs/grass_calibrate_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Quick calibration: measures t_init, t_mine (both variants), t_train on this
# hardware and prints a config snippet. Runs single-GPU. ~25-35 min.

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
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=0

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Calibration knobs (override via env vars before sbatch) ---
# CAL_N_INIT          queries for t_init (default 5000)
# CAL_N_MINE          queries for each t_mine variant (default 1000)
# CAL_N_TRAIN_STEPS   trainer steps for t_train (default 200)
# CAL_SKIP_TRAIN=1    skip the t_train measurement (~15 min faster)

mkdir -p logs

echo "📏 Starting async v2 calibration..."
echo "   N_INIT=${CAL_N_INIT:-5000} | N_MINE=${CAL_N_MINE:-1000} | N_TRAIN_STEPS=${CAL_N_TRAIN_STEPS:-200} | SKIP_TRAIN=${CAL_SKIP_TRAIN:-0}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/calibrate_async_v2_times.py \
        --recipe grass \
        ${CAL_N_INIT:+--n_init $CAL_N_INIT} \
        ${CAL_N_MINE:+--n_mine $CAL_N_MINE} \
        ${CAL_N_TRAIN_STEPS:+--n_train_steps $CAL_N_TRAIN_STEPS} \
        ${CAL_SKIP_TRAIN:+--skip_train}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Calibration completed. Paste the printed snippet into config.yaml."
else
    echo "❌ Calibration failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
