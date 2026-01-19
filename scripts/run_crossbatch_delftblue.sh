#!/usr/bin/env bash

#SBATCH --job-name=crossbatch_train
#SBATCH --partition=gpu-v100  # V100: Phase 1 GPU nodes with 32 GB video RAM each
#SBATCH --time=04:00:00  # gpu-v100: max 4 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # gpu-v100: more CPUs available
#SBATCH --gpus-per-task=1  # V100 GPU with 32 GB video RAM
#SBATCH --mem-per-cpu=5G  # gpu-v100: max 5G per CPU (5333 MB limit)
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=crossbatch_train_%j.out
#SBATCH --error=crossbatch_train_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Load modules ---
module purge
module load 2025
module load cuda/12.9
module load miniconda3/4.12.0

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- CRITICAL: Ensure correct Tevatron and grad-cache versions ---
# Option 1: Install from GitHub (may have latest fixes)
# pip install git+https://github.com/texttron/tevatron.git
# Option 2: Install pinned versions (known to work with FP16 + grad_cache)
# pip install tevatron==0.2.0 grad-cache==0.0.1

# Verify Tevatron installation
echo "Verifying Tevatron installation..."
python -c "import tevatron; print(f'Tevatron version: {getattr(tevatron, \"__version__\", \"unknown\")}')" || {
    echo "ERROR: Tevatron not installed correctly!"
    exit 1
}
python -c "import tevatron.retriever.driver" || {
    echo "ERROR: tevatron.retriever.driver module not found!"
    echo "Try: pip install tevatron==0.2.0 grad-cache==0.0.1"
    exit 1
}
echo "✅ Tevatron installation verified"

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

# --- PyTorch CUDA memory management ---
# Reduce memory fragmentation (helps with OOM errors)
# PyTorch memory management - reduce fragmentation
# Note: Error message shows PYTORCH_CUDA_ALLOC_CONF but that's deprecated
# Using PYTORCH_ALLOC_CONF (new name) with both settings
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
# Also set deprecated name for compatibility (some versions may still check it)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Note: Dataset and processed data directories are now auto-detected
# Code automatically detects DelftBlue (via SLURM_JOB_ID and /scratch/${USER})
# and uses scratch space: /scratch/${USER}/dense-retrieval-SOTA/data/
# No need to set BRIGHT_CACHE_DIR or PROCESSED_DATA_DIR manually

# --- Initialize DDP for Single GPU (grad_cache requires DDP) ---
# grad_cache library requires models to be wrapped in DistributedDataParallel
# Even for single GPU, we need to initialize DDP (torchrun will do this automatically)
# DDP works fine with single GPU - it's just less efficient but required by grad_cache
# Note: torchrun will set WORLD_SIZE, RANK, LOCAL_RANK automatically
export MASTER_ADDR=localhost
# Use SLURM_JOB_ID to generate unique port (avoids EADDRINUSE errors)
# Port range: 10000-65535 (avoiding system ports)
if [ -n "$SLURM_JOB_ID" ]; then
    export MASTER_PORT=$((10000 + SLURM_JOB_ID % 55535))
else
    export MASTER_PORT=$((10000 + RANDOM % 55535))
fi
echo "Using MASTER_PORT=${MASTER_PORT} (SLURM_JOB_ID=${SLURM_JOB_ID:-N/A})"

# --- Configuration (can be overridden via environment variables) ---
# PRODUCTION MODE: Full batch sizes for gpu-v100 (32 GB video RAM)
# V100 does NOT support BF16 (only A100 does), so we use FP32 or FP16
# Target batch size 1024 for full training
export TARGET_BATCH_SIZE=${TARGET_BATCH_SIZE:-1024}  # Virtual batch size (target) - full training
export PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-64}  # Physical batch size (reduced for FP32 on 32GB GPU)
export CHUNK_SIZE=${CHUNK_SIZE:-64}  # Chunk size - grad_cache accumulates 16 chunks to reach 1024
# Note: V100 supports FP16 but not BF16. Using FP32 for stability (or FP16 if needed)
# Chunk size 128 with batch size 128 should use ~12-16 GB of the 32 GB GPU memory (safe)
export LEARNING_RATE=${LEARNING_RATE:-1e-5}
export NUM_EPOCHS=${NUM_EPOCHS:-2}  # Full training: 2 epochs

echo "=========================================="
echo "Cross-Batch (RocketQA) Training Configuration - PRODUCTION MODE:"
echo "  GPU: V100 (32 GB video RAM)"
echo "  Virtual Batch Size: ${TARGET_BATCH_SIZE} (target - full training)"
echo "  Physical Batch Size: ${PHYSICAL_BATCH_SIZE} (per GPU - for 32GB GPU)"
echo "  Gradient Cache Chunk Size: ${CHUNK_SIZE} (accumulates $((TARGET_BATCH_SIZE / CHUNK_SIZE)) chunks to reach ${TARGET_BATCH_SIZE} - uses ~12-16 GB)"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Precision: FP32 (no mixed precision - FP16 causes scaler errors)"
echo "  Output: ${SCRATCH_DIR}/models/crossbatch_reasonir"
echo "  Dataset Cache: Auto-detected (scratch space on DelftBlue)"
echo "=========================================="

# --- Run Cross-Batch (RocketQA) training ---
# The Python script now uses Tevatron's API directly and handles DDP wrapping.
# We use torchrun to initialize the distributed environment for 2 GPUs.
# Check if script exists in scratch, otherwise use home
SCRIPT_PATH="${SCRATCH_DIR}/scripts/train_crossbatch.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="/home/${USER}/dense-retrieval-SOTA/scripts/train_crossbatch.py"
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: train_crossbatch.py not found in /scratch or /home!"
    exit 1
fi

# Use 1 process for single GPU - DDP still required by grad_cache
# MASTER_PORT is already set above, torchrun will use it from environment
torchrun --nproc_per_node=1 "$SCRIPT_PATH"

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${SCRATCH_DIR}/models/crossbatch_reasonir"
echo "=========================================="
