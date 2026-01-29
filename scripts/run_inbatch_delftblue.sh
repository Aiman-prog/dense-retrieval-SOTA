#!/usr/bin/env bash

#SBATCH --job-name=inbatch_128
#SBATCH --partition=gpu-a100
#SBATCH --time=05:00:00               # Increased to 8h for HQ+VL mixture
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M                     # Simplified memory request
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/inbatch_%j.out
#SBATCH --error=logs/inbatch_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- 1. Load Modules ---
module purge
module load 2025 cuda/12.9 miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- 2. Environment Fixes ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Memory Management (Prevents fragmentation OOM at high limits)
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Hugging Face Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 3. Run ---
python scripts/train_inbatch.py

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="