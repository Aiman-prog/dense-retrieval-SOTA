#!/usr/bin/env bash

#SBATCH --job-name=crossbatch_train
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/crossbatch_%j.out
#SBATCH --error=logs/crossbatch_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Load modules ---
module purge
module load 2025
module load gcc          # This will load GCC 13.x (modern C++)
module load cuda/12.9
module load miniconda3/4.12.0

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- 2. Centralized Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA:${PYTHONPATH}"

# Hugging Face Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 3. Memory & DDP Management ---
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 55535))

# --- 4. Run Training ---
torchrun --nproc_per_node=1 scripts/train_crossbatch.py

echo "=========================================="
echo "Cross-Batch Training Job %j Completed"
echo "=========================================="