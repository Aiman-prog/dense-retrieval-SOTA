#!/usr/bin/env bash

#SBATCH --job-name=eval_reasonir
#SBATCH --partition=gpu-a100-small   # Max 10GB VRAM, Max 2 CPUs
#SBATCH --time=01:00:00              # Fast job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # <--- CHANGED: Max allowed on this partition
#SBATCH --gpus-per-task=1            # 1 Slice of A100 (10GB)
#SBATCH --mem-per-cpu=5G             # 10GB System RAM total (5G * 2)
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=eval_reasonir_%j.out
#SBATCH --error=eval_reasonir_%j.err
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
mkdir -p "${SCRATCH_DIR}/data/evaluation"

# --- CACHE REDIRECTION (Standard) ---
SCRATCH_CACHE_DIR="${SCRATCH_DIR}/cache"
mkdir -p "${SCRATCH_CACHE_DIR}"
export CONDA_PKGS_DIRS="${SCRATCH_CACHE_DIR}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_CACHE_DIR}/pip"
export TORCH_HOME="${SCRATCH_CACHE_DIR}/torch"
export XDG_CACHE_HOME="${SCRATCH_CACHE_DIR}/xdg"
HF_CACHE_DIR="${SCRATCH_DIR}/data/bright"
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-SOTA:${PYTHONPATH}

# --- CONFIGURATION ---
MODEL_PATH="/scratch/aimanabdulwaha/dense-retrieval-SOTA/models/crossbatch_reasonir"
DOMAIN="biology"
K=10

# --- SAFE BATCH SIZE ---
# 128 is the "Safe Zone" for a 10GB GPU slice.
# 256 is risky (might hit 10.1GB and crash).
BATCH_SIZE=128

echo "=========================================="
echo "Evaluation Configuration (Small Partition):"
echo "  CPUs: 2 (Partition Limit)"
echo "  Batch Size: ${BATCH_SIZE} (Safe for 10GB)"
echo "  Model: ${MODEL_PATH}"
echo "=========================================="

# --- RUN EVALUATION ---
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

python src/evaluation/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --domain "${DOMAIN}" \
    --k "${K}" \
    --batch_size "${BATCH_SIZE}"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✅ Evaluation completed successfully!"