#!/usr/bin/env bash

#SBATCH --job-name=eval-reasonir
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- CONFIGURATION ---
MODEL_PATH="${EVAL_MODEL_PATH:?Error: EVAL_MODEL_PATH must be set}"

# --- Create output directories ---
mkdir -p logs

# --- Run Evaluation in Container ---
# K and batch_size are read from config/config.yaml by the Python scripts
echo "🔍 Starting evaluation for model: ${MODEL_PATH}"
echo "📊 Evaluating on all BRIGHT domains (settings from config.yaml)"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_all_evals.py \
        --model_path "${MODEL_PATH}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
else
    echo "❌ Evaluation failed with code $EXIT_CODE"
fi

exit $EXIT_CODE