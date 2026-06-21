#!/usr/bin/env bash

#SBATCH --job-name=fg_feas
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fg_feas_%j.out
#SBATCH --error=logs/fg_feas_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Standalone Phase-1 gate: baseline GRASS mining vs Fast-GRASS mining on the SAME
# batches, emitting encoder rows + wall-time + the SPEEDUP RATIO. No training, no
# checkpoints, no data/config mutation. sbatch this independently of a training run.
#
# Faster queue: swap --partition=gpu-a100 -> gpu-a100-small (MIG slice) above for a
# shorter wait when B_doc / batch counts are small.

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

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Benchmark Knobs (override via env vars before sbatch) ---
# FG_FEAS_BATCHES=10              # number of query batches to time (--batches)
# FG_FEAS_B_DOC=100000           # cache size override (--B_doc)
# FG_FEAS_UNCERTAINTY=ema        # estimator (--uncertainty; v0: ema only)
# FG_FEAS_MAINTAIN_EVERY=100     # reserved knob (--maintain-every)
# FG_FEAS_STEPS_PER_EPOCH=5000   # override derived steps_per_epoch (--steps-per-epoch)

mkdir -p logs

echo "📐 Fast-GRASS feasibility benchmark (baseline vs Fast-GRASS mining)..."
echo "   BATCHES=${FG_FEAS_BATCHES:-10} | B_doc=${FG_FEAS_B_DOC:-cfg} | UNC=${FG_FEAS_UNCERTAINTY:-ema} | MAINT_EVERY=${FG_FEAS_MAINTAIN_EVERY:-amort} | SPE=${FG_FEAS_STEPS_PER_EPOCH:-derived}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/fast_grass_feasibility.py \
        --batches ${FG_FEAS_BATCHES:-10} \
        ${FG_FEAS_B_DOC:+--B_doc $FG_FEAS_B_DOC} \
        ${FG_FEAS_UNCERTAINTY:+--uncertainty $FG_FEAS_UNCERTAINTY} \
        ${FG_FEAS_MAINTAIN_EVERY:+--maintain-every $FG_FEAS_MAINTAIN_EVERY} \
        ${FG_FEAS_STEPS_PER_EPOCH:+--steps-per-epoch $FG_FEAS_STEPS_PER_EPOCH}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Feasibility benchmark completed"
else
    echo "❌ Feasibility benchmark failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
