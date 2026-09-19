#!/usr/bin/env bash

# Paper-fidelity ANCE: the MS MARCO Passage reproduction (RoBERTa + LAMB + pairwise
# NLL). Not a BRIGHT arm -- `run_ance_singularity.sh` is that one.
#
# The recipe is written out here rather than read from an environment variable: this
# job is only ever the reproduction, and an unset variable silently selecting a
# different model is exactly the failure this launcher exists to prevent.

#SBATCH --job-name=ance-paper
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00              # train_stop_steps is chosen to fit this
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2            # 1:1 Trainer:Inferencer GPU split (paper Appendix A.3)
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_paper_%j.out
#SBATCH --error=logs/ance_paper_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
source scripts/launchers/runtime_provenance.sh || exit 1
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance Tuning
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "🚀 ANCE MS MARCO Passage reproduction (recipe: ance_paper)"
echo "   Refuses to start unless training.ance_paper.expected_init_sha256 matches"
echo "   the released 60K BM25 warm-up on disk."

# Adopt a round 0 mined by run_ance_paper_prepare_singularity.sh when one is named.
# Never discovered automatically: a leftover artifact must not reach a run by accident,
# and the adopted round is refused unless it was mined from this exact initialization,
# corpus, queries, qrels, mixture, mining settings and seed.
# ance_paper_roberta may still hold checkpoints from a run that died (204931 left
# checkpoint-10000). The orchestrator refuses to delete those silently, so discarding
# them is a decision taken HERE, explicitly, by whoever submits.
OVERWRITE_ARG=""
if [ -n "${ANCE_OVERWRITE:-}" ]; then
    OVERWRITE_ARG="--overwrite"
    echo "   ANCE_OVERWRITE set — existing checkpoints in the output dir will be cleared"
fi

INITIAL_ARG=""
if [ -n "${ANCE_INITIAL_ROUND:-}" ]; then
    INITIAL_ARG="--initial-round ${ANCE_INITIAL_ROUND}"
    echo "   Adopting prepared initial round: ${ANCE_INITIAL_ROUND}"
else
    echo "   No ANCE_INITIAL_ROUND set — round 0 will be mined inside this allocation"
    echo "   (measured 6h38m in job 204931, against a 24h wall)."
fi

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance.py --recipe ance_paper ${INITIAL_ARG} ${OVERWRITE_ARG}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ance_paper completed. Evaluate with:"
    echo "   EVAL_MODEL_PATH=<checkpoint> \\"
    echo "     sbatch scripts/launchers/eval_msmarco_singularity.sh"
else
    echo "❌ ance_paper failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
