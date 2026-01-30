#!/usr/bin/env bash

#SBATCH --job-name=ANCE_ReasonIR
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00               
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000             
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_%j.out
#SBATCH --error=logs/ance_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- 1. Load Modules ---
module purge
module load 2025 cuda/12.9 miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- 2. Environment Fixes ---
# Use scratch for high-speed I/O during iterative mining
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Memory & Performance Tuning
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16             # Matches cpus-per-task for FAISS speed

# Hugging Face Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 3. Run ANCE Pipeline ---
echo "🚀 Starting Iterative ANCE Loop..."
python scripts/train_ance.py

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="