#!/usr/bin/env bash

# Validate the BM25 warm-up end to end WITHOUT a GPU, before spending a GPU slot.
#
# Runs the same code on the same data as the real job: loads the real 5.2 GB mixture
# and reports its ragged-negative counts, builds the model from real roberta-base and
# checks exactly the projection head is fresh, builds the collate/LAMB/schedule, takes
# a few optimization steps on CPU, saves a checkpoint, reloads it through
# load_ance_encoder and runs assert_training_succeeded. Writes nothing to the model dir.
#
# CPU partition, so it clears the queue in minutes: this preflight ran in 2:26 (75676).
# Memory is sized for the eager mixture load: 400,782 records of parsed JSON with full
# passage text peak at 28.4 GB RSS (measured, 75676) -- roughly 5x the 5.2 GB file, and
# far above job 70494's 10.9 GB, which was a mid-load sample rather than the peak.
# compute-p1 caps mem-per-cpu at 3996M, so the CPU count is really a memory request.

#SBATCH --job-name=ance-warmup-preflight
#SBATCH --partition=compute-p1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12          # = 46.8 GB; 28.4 GB peak needs real headroom
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_warmup_preflight_%j.out
#SBATCH --error=logs/ance_warmup_preflight_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=12

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "🔎 BM25 warm-up preflight (recipe: ance_paper_warmup) — no GPU, no writes"

# No --nv: this must not bind or initialize a GPU.
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance_warmup.py --preflight

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ preflight passed — safe to submit:"
    echo "     sbatch scripts/launchers/run_ance_warmup_singularity.sh"
else
    echo "❌ preflight failed with code $EXIT_CODE — do NOT submit the GPU job"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
