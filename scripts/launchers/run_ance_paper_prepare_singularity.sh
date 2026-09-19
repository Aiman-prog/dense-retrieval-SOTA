#!/usr/bin/env bash

# Mine ance_paper's round 0 ONCE, on one GPU, outside the training allocation.
#
# Job 204931 spent 6h38m on it inside a 24h two-GPU job, which is why 300K steps (20.5h)
# could not fit. Round 0 depends only on the warm-up weights and the corpus, so it is the
# same every run; upstream splits the stage out too (run_train.sh --end_output_num 0).
# Stage 1 is the CPU preflight, so an input regression costs minutes, not six GPU hours.
#
# Train against the result with:
#   sbatch --export=ALL,ANCE_INITIAL_ROUND=$PREPARE_DIR scripts/launchers/run_ance_paper_singularity.sh

#SBATCH --job-name=ance-paper-prepare
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00              # measured 6h38m; headroom for a slower node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1            # ONE gpu: there is no trainer beside this
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_paper_prepare_%j.out
#SBATCH --error=logs/ance_paper_prepare_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH:-}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

source scripts/launchers/runtime_provenance.sh || exit 1

export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
PREPARE_DIR="${ANCE_PREPARE_DIR:-${DATA_BASE_DIR}/prepared_rounds/ance_paper_initial}"

mkdir -p logs "$(dirname "${PREPARE_DIR}")"

echo "🧱 ance_paper initial-round preparation → ${PREPARE_DIR}"
echo "   Refuses to overwrite a committed artifact; point ANCE_PREPARE_DIR elsewhere"
echo "   if you need to re-mine."

# ---- stage 1: CPU preflight, no GPU bound ----
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance.py --recipe ance_paper --preflight
PREFLIGHT=$?
if [ $PREFLIGHT -ne 0 ]; then
    echo "❌ preflight failed with code ${PREFLIGHT} — not mining"
    exit $PREFLIGHT
fi

# ---- stage 2: mine round 0 on one GPU ----
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance.py --recipe ance_paper --prepare-initial "${PREPARE_DIR}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ prepared. Train with:"
    echo "   sbatch --export=ALL,ANCE_INITIAL_ROUND=${PREPARE_DIR} \\"
    echo "     scripts/launchers/run_ance_paper_singularity.sh"
else
    echo "❌ preparation failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="
exit $EXIT_CODE
