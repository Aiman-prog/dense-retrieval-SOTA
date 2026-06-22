#!/usr/bin/env bash

#SBATCH --job-name=fg_eval
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fg_eval_%j.out
#SBATCH --error=logs/fg_eval_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# BRIGHT eval for a Fast-GRASS sweep (trained with --no_eval).
#
# Each eval writes scratch into a PER-MODEL dir (<model_dir>/eval_scratch), so runs
# never collide. Two ways to drive it:
#
#   PARALLEL (one job per model, different GPUs — fastest):
#     for d in /scratch/$USER/dense-retrieval-SOTA/models/*fg_*_ema; do
#       FG_EVAL_MODEL_DIR="$d" sbatch scripts/run_fast_grass_eval_singularity.sh
#     done
#
#   SEQUENTIAL (one job, all models in a loop):
#     FG_EVAL_GLOB="/scratch/$USER/dense-retrieval-SOTA/models/*fg_*_ema" \
#       sbatch scripts/run_fast_grass_eval_singularity.sh

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
export OMP_NUM_THREADS=2   # match cpus-per-task=2 on gpu-a100-small (avoid oversubscription)
export CC=gcc
export TORCHDYNAMO_DISABLE=1   # eval does no compile; keep logs clean

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

run_one() {
    local MODEL_DIR="$1"
    echo "──────────────────────────────────────────────"
    echo "Evaluating: ${MODEL_DIR}"
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/run_fast_grass_eval.py --model_dir "${MODEL_DIR}"
    return $?
}

OVERALL=0
if [ -n "${FG_EVAL_MODEL_DIR}" ]; then
    # one model per job — submit several of these in parallel
    echo "📊 Fast-GRASS BRIGHT eval (single model, parallel-safe per-model scratch)"
    run_one "${FG_EVAL_MODEL_DIR}"; OVERALL=$?
else
    # sequential fallback: loop a glob in one job
    FG_EVAL_GLOB="${FG_EVAL_GLOB:-${DATA_BASE_DIR}/models/*fg_*_ema}"
    echo "📊 Fast-GRASS BRIGHT eval (sequential) over: ${FG_EVAL_GLOB}"
    for MODEL_DIR in ${FG_EVAL_GLOB}; do
        [ -d "${MODEL_DIR}" ] || continue
        run_one "${MODEL_DIR}"
        RC=$?
        [ $RC -ne 0 ] && { echo "❌ eval failed for ${MODEL_DIR} (code $RC)"; OVERALL=$RC; }
    done
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed (overall=$OVERALL)"
echo "=========================================="
exit $OVERALL
