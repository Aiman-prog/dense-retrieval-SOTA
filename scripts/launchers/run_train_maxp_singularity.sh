#!/usr/bin/env bash

#SBATCH --job-name=maxp-reasonir
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/maxp_%j.out
#SBATCH --error=logs/maxp_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
# The submitting checkout (sbatch --chdir), never a hardcoded one.
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory Management
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Experiment Knobs (override via env vars before sbatch) ---
# MAXP_RECIPE=maxp             in-batch (default), 1 GPU
# MAXP_RECIPE=crossbatch_maxp  cross-batch: submit with --gpus-per-task=2; runs torchrun x2
# MAXP_BENCHMARK=1             end-to-end timed steps; GO/NO-GO, writes nothing
# There is no overwrite: an occupied output dir is refused, never cleared.
# Preflight is CPU-only, so it is a compute-p1 `sbatch --wrap`, not this launcher.
MAXP_RECIPE="${MAXP_RECIPE:-maxp}"
NPROC=1
[ "$MAXP_RECIPE" = "crossbatch_maxp" ] && NPROC=2

echo "🚀 Starting MaxP training${MAXP_BENCHMARK:+ (benchmark)}..."

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    torchrun --standalone --nproc_per_node "$NPROC" scripts/train_maxp.py \
        --recipe "$MAXP_RECIPE" ${MAXP_BENCHMARK:+--benchmark}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ MaxP ${MAXP_BENCHMARK:+benchmark }completed successfully"
else
    echo "❌ MaxP ${MAXP_BENCHMARK:+benchmark }failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
