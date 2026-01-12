#!/usr/bin/env bash

#SBATCH --job-name=evaluate_rocketqa
#SBATCH --partition=gpu-a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=evaluate_rocketqa_%j.out
#SBATCH --error=evaluate_rocketqa_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Load modules ---
module purge
module load 2025
module load cuda/12.9
module load miniconda3/4.12.0

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- Set up scratch space ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval-SOTA"
mkdir -p "${SCRATCH_DIR}/models"
mkdir -p "${SCRATCH_DIR}/data/processed"
mkdir -p "${SCRATCH_DIR}/data/bright"

# --- Set Hugging Face cache location (use scratch space) ---
HF_CACHE_DIR="${SCRATCH_DIR}/data/bright"
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE_DIR}"

# --- CRITICAL: Set Hugging Face to OFFLINE mode ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "=========================================="
echo "OFFLINE MODE ENABLED"
echo "  Datasets must be pre-downloaded to cache"
echo "  Cache location: ${HF_CACHE_DIR}"
echo "=========================================="

# --- Set PYTHONPATH to project root ---
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-SOTA:${PYTHONPATH}

# --- Configuration ---
MODEL_PATH=${MODEL_PATH:-/scratch/aimanabdulwaha/dense-retrieval-SOTA/models/inbatch_reasonir}
DOMAIN=${DOMAIN:-biology}
BATCH_SIZE=${BATCH_SIZE:-512}
K=${K:-10}

echo "=========================================="
echo "Evaluation Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Domain: ${DOMAIN}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Top-K: ${K}"
echo "=========================================="

# --- Run evaluation ---
python -m src.evaluation.evaluate \
  --model_path "${MODEL_PATH}" \
  --domain "${DOMAIN}" \
  --batch_size "${BATCH_SIZE}" \
  --k "${K}"

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

