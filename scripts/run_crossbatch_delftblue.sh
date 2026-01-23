#!/usr/bin/env bash

#SBATCH --job-name=crossbatch_train
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/crossbatch_%j.out
#SBATCH --error=logs/crossbatch_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- 1. Load Modules & Activate Env ---
module purge
module load 2025 cuda/12.9 miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- 2. Centralized Environment Setup ---
# This allows helpers.py to find your scratch space automatically
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA:${PYTHONPATH}"

# Hugging Face Offline Mode (Points to your scratch cache)
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 3. Memory & DDP Management ---
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export MASTER_ADDR=localhost
# Unique port based on Job ID to avoid collisions
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 55535))

# --- 4. Run Training ---
# We use torchrun to initialize the DDP environment required by GradCache.
# All hyperparameters (2048 batch size, 2e-5 LR) are pulled from config.yaml
torchrun --nproc_per_node=1 scripts/train_crossbatch.py

echo "=========================================="
echo "Cross-Batch Training Job %j Completed"
echo "=========================================="