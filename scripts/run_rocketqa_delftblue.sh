#!/usr/bin/env bash

#SBATCH --job-name=rocketqa_train
#SBATCH --partition=gpu-a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=rocketqa_train_%j.out
#SBATCH --error=rocketqa_train_%j.err
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

# --- Set Hugging Face cache location (use scratch space) ---
# Match bright_loader.py cache location: /scratch/${USER}/dense-retrieval-SOTA/data/bright
SCRATCH_DIR="/scratch/${USER}/dense-retrieval-SOTA"
HF_CACHE_DIR="${SCRATCH_DIR}/data/bright"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE_DIR}"

# --- CRITICAL: Set Hugging Face to OFFLINE mode ---
# Models and datasets MUST be pre-downloaded to cache before running
# Pre-download using: python scripts/prepare_offline_cache.py (or similar)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "=========================================="
echo "OFFLINE MODE ENABLED"
echo "  Models must be pre-downloaded to cache"
echo "  Datasets must be pre-downloaded to cache"
echo "  Cache location: ${HF_CACHE_DIR}"
echo "=========================================="

# Debug: show cache location and contents
echo "DEBUG: HF_HOME=${HF_HOME}"
echo "DEBUG: Cache contents:"
ls -la "${HF_CACHE_DIR}/" 2>/dev/null | head -10 || echo "   Cache directory empty or not accessible"

# --- Set PYTHONPATH to project root ---
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-SOTA:${PYTHONPATH}

# Note: Dataset and processed data directories are now auto-detected
# Code automatically detects DelftBlue (via SLURM_JOB_ID and /scratch/${USER})
# and uses scratch space: /scratch/${USER}/dense-retrieval-SOTA/data/
# No need to set BRIGHT_CACHE_DIR or PROCESSED_DATA_DIR manually

# --- Configuration (can be overridden via environment variables) ---
DOMAIN=${DOMAIN:-biology}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-128} 
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}

echo "=========================================="
echo "RocketQA Training Configuration:"
echo "  Domain: ${DOMAIN}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "    Output: ${SCRATCH_DIR}/models/rocketqa_${DOMAIN}"
  echo "  Dataset Cache: Auto-detected (scratch space on DelftBlue)"
echo "=========================================="

# --- Run RocketQA training ---
# Note: This script assumes data has been preprocessed and cached
# The script will:
# 1. Load dataset from cache (offline)
# 2. Load model from cache (offline)
# 3. Train RocketQA model
python scripts/train_rocketqa.py \
  --num_gpus ${NUM_GPUS}

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${SCRATCH_DIR}/models/rocketqa_${DOMAIN}"
echo "=========================================="

