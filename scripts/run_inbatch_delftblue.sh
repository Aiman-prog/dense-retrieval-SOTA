#!/usr/bin/env bash

#SBATCH --job-name=inbatch_train
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/inbatch_%j.out
#SBATCH --error=logs/inbatch_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- 1. Load Modules & Activate Env ---
module purge
module load 2025 cuda/12.9 miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- 2. Centralized Environment Setup ---
# This is the "Nervous System" that connects DelftBlue to your Python code
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"

# Hugging Face Offline Mode (Points to your scratch cache)
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 3. Run Training ---
# Hyperparameters are automatically pulled from config.yaml 'inbatch' recipe
# Paths are automatically resolved to /scratch via helpers.py
python scripts/train_inbatch.py

echo "=========================================="
echo "Job %j Completed"
echo "=========================================="