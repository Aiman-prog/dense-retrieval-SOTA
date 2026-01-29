#!/usr/bin/env bash

#SBATCH --job-name=eval_reasonir
#SBATCH --partition=gpu-a100-small   
#SBATCH --time=04:00:00              
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            
#SBATCH --gpus-per-task=1            
#SBATCH --mem-per-cpu=8000M           
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/eval_crossbatch_%j.out
#SBATCH --error=logs/eval_crossbatch_%j.err
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

# --- Set up scratch space ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval-SOTA"
mkdir -p "${SCRATCH_DIR}/models"
mkdir -p "${SCRATCH_DIR}/data/processed"
mkdir -p "${SCRATCH_DIR}/data/evaluation"

# --- CACHE REDIRECTION ---
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
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# --- CONFIGURATION ---
MODEL_PATH="/scratch/aimanabdulwaha/dense-retrieval-SOTA/models/inbatch_reasonir"
DOMAIN="biology"
K=10
BATCH_SIZE=128

# --- RUN EVALUATION ---
# python -u src/evaluation/evaluate.py \
#     --model_path "${MODEL_PATH}" \
#     --domain "${DOMAIN}" \
#     --k "${K}" \
#     --batch_size "${BATCH_SIZE}"

python -u scripts/run_all_evals.py \
    --model_path "${MODEL_PATH}" \
    --k "${K}" \
    --batch_size "${BATCH_SIZE}"

EXIT_CODE=$?
echo "✅ Evaluation process finished with code $EXIT_CODE"
exit $EXIT_CODE