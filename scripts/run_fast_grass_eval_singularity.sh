#!/usr/bin/env bash

#SBATCH --job-name=fg_eval
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fg_eval_%j.out
#SBATCH --error=logs/fg_eval_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Sequential BRIGHT eval of a Fast-GRASS sweep (trained with --no_eval), using the
# canonical utils.helpers.evaluate_bright. ONE job, ONE GPU (within the account's GPU
# quota), models evaluated one at a time — evaluate_bright shares a single scratch dir
# so it must NOT run concurrently. Full gpu-a100 (40/80GB): the config eval batch /
# query_max_len=1024 fit here (gpu-a100-small's ~9.5GB MIG slice OOMs on long queries).
#
#   FG_EVAL_GLOB="/scratch/$USER/dense-retrieval-SOTA/models/*fg_*_ema" \
#     sbatch scripts/run_fast_grass_eval_singularity.sh

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export CC=gcc
export TORCHDYNAMO_DISABLE=1   # eval does no compile; keep logs clean

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# Glob of trained model dirs to evaluate (override before sbatch).
FG_EVAL_GLOB="${FG_EVAL_GLOB:-${DATA_BASE_DIR}/models/*fg_*_ema}"

mkdir -p logs

echo "📊 Sequential Fast-GRASS BRIGHT eval over: ${FG_EVAL_GLOB}"

OVERALL=0
for MODEL_DIR in ${FG_EVAL_GLOB}; do
    [ -d "${MODEL_DIR}" ] || continue
    echo "──────────────────────────────────────────────"
    echo "Evaluating: ${MODEL_DIR}"
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/run_fast_grass_eval.py --model_dir "${MODEL_DIR}"
    RC=$?
    [ $RC -ne 0 ] && { echo "❌ eval failed for ${MODEL_DIR} (code $RC)"; OVERALL=$RC; }
done

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed (overall=$OVERALL)"
echo "=========================================="
exit $OVERALL
