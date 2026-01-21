#!/usr/bin/env bash

#SBATCH --job-name=inbatch_train
#SBATCH --partition=gpu-v100   # Switch to the small partition
#SBATCH --time=04:00:00  # 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # 4 CPUs: 1 main process + 4 dataloader workers
#SBATCH --gpus-per-task=1  # V100 GPU with 32 GB video RAM
#SBATCH --mem-per-cpu=5G  # gpu-v100: max 5G per CPU
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=inbatch_train_%j.out
#SBATCH --error=inbatch_train_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Load modules ---
module purge
module load 2025
module load cuda/12.9
module load miniconda3/4.12.0

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- Set up scratch space for outputs ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval-SOTA"
mkdir -p "${SCRATCH_DIR}/models"
mkdir -p "${SCRATCH_DIR}/data/processed"
mkdir -p "${SCRATCH_DIR}/data/bright"

# Note: Processed data directory is now auto-detected
# Code will automatically use /scratch/${USER}/dense-retrieval-SOTA/data/processed on DelftBlue

# --- CRITICAL: Redirect ALL caches to /scratch to save /home space ---
# This prevents future cache growth in ~/.cache (~9GB) and ~/.conda (~17GB)
# Existing caches remain in /home but won't grow further.
# To clean up existing caches manually:
#   rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
#   conda clean --all  # Cleans conda cache (but keeps environments)
# Note: Conda environments (~/.conda/envs) are NOT moved - only package cache is redirected
SCRATCH_CACHE_DIR="${SCRATCH_DIR}/cache"
mkdir -p "${SCRATCH_CACHE_DIR}"

# Conda package cache (saves ~17GB in ~/.conda)
export CONDA_PKGS_DIRS="${SCRATCH_CACHE_DIR}/conda-pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

# Pip cache (saves ~7.7GB in ~/.cache/pip)
export PIP_CACHE_DIR="${SCRATCH_CACHE_DIR}/pip"
mkdir -p "${PIP_CACHE_DIR}"

# PyTorch cache (saves ~759MB in ~/.cache/torch)
export TORCH_HOME="${SCRATCH_CACHE_DIR}/torch"
mkdir -p "${TORCH_HOME}"

# General cache directory (redirects ~/.cache to /scratch)
export XDG_CACHE_HOME="${SCRATCH_CACHE_DIR}/xdg"
mkdir -p "${XDG_CACHE_HOME}"

# Hugging Face caches (saves ~589MB in ~/.cache/huggingface)
HF_CACHE_DIR="${SCRATCH_DIR}/data/bright"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE_DIR}"

# --- CRITICAL: Set Hugging Face to OFFLINE mode ---
# Models and datasets MUST be pre-downloaded to cache before running
# Pre-download using: python scripts/prepare_models.py and load datasets once online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "=========================================="
echo "OFFLINE MODE ENABLED"
echo "  Models must be pre-downloaded to cache"
echo "  Datasets must be pre-downloaded to cache"
echo "  Cache location: ${HF_CACHE_DIR}"
echo "=========================================="

# Debug: show cache locations
echo "DEBUG: Cache directories redirected to /scratch:"
echo "  CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
echo "  PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "  HF_HOME=${HF_HOME}"
echo "DEBUG: Cache directory sizes:"
du -sh "${SCRATCH_CACHE_DIR}"/* 2>/dev/null | head -5 || echo "   Cache directories empty or not accessible"

# --- Set PYTHONPATH to project root ---
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-SOTA:${PYTHONPATH}

# Note: Dataset and processed data directories are now auto-detected
# Code automatically detects DelftBlue (via SLURM_JOB_ID and /scratch/${USER})
# and uses scratch space: /scratch/${USER}/dense-retrieval-SOTA/data/
# No need to set BRIGHT_CACHE_DIR or PROCESSED_DATA_DIR manually

# --- PyTorch CUDA memory management ---
# Reduce memory fragmentation (helps with OOM errors)
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"  # Reduce fragmentation, limit max split size
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"  # For compatibility

# --- Configuration (can be overridden via environment variables) ---
# Batch size for gpu-v100 (32GB GPU memory, FP32 mode)
# Using gradient accumulation to achieve effective batch size 64
BATCH_SIZE=${BATCH_SIZE:-64}  # Physical batch size 32, effective batch size 64 with gradient accumulation
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}

# Export for Python script
export BATCH_SIZE

echo "=========================================="
echo "In-Batch Negatives Training Configuration:"
echo "  GPU: V100 (32 GB video RAM, 4 CPUs)"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Output: ${SCRATCH_DIR}/models/inbatch_reasonir"
echo "  Dataset Cache: Auto-detected (scratch space on DelftBlue)"
echo "=========================================="

# --- Run In-Batch Negatives training ---
# Note: This script assumes data has been preprocessed and cached
# The script will:
# 1. Load dataset from cache (offline)
# 2. Load model from cache (offline)
# 3. Train In-Batch Negatives model
python scripts/train_inbatch.py

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${SCRATCH_DIR}/models/inbatch_reasonir"
echo "=========================================="
