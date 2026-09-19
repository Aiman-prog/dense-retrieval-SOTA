#!/usr/bin/env bash

# The BM25 warm-up `ance_paper` initializes from. Microsoft's released 60K warm-up is
# gone (both blob URLs 409; microsoft/ANCE #23/#24/#26 open since 2022; the only mirror
# is bit-identical to the 600K FINAL, which assert_permitted_init refuses as an init),
# so it is trained here from roberta-base on the BM25 negatives already in the mixture.
#
# ONE GPU, not two: there is no ANN index and no Inferencer. Static negatives only.

#SBATCH --job-name=ance-warmup
#SBATCH --partition=gpu-a100
#SBATCH --time=05:00:00              # 60K steps at seq 128 / batch 32 on RoBERTa-base
                                     # is ~2-3h; the rest is margin for the eager read
                                     # of the 5.2 GB mixture at startup. There is no
                                     # resume here, so a wall-clock kill loses the run.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_warmup_%j.out
#SBATCH --error=logs/ance_warmup_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "🚀 BM25 warm-up (recipe: ance_paper_warmup)"
echo "   roberta-base + a FRESH projection head, 60K steps at seq 128."
echo "   Refuses to start unless exactly the head is newly initialized."

# --- 1/2 preflight, NO GPU bound (no --nv) ---
# The same code on the same data: real mixture, real roberta-base, real collate and
# optimizer, a few CPU steps, save and reload. Job 70494 burned a GPU allocation to
# die on the mixture 2:30 in; this catches that class of failure for free.
echo "--- [1/2] preflight (no GPU) ---"
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance_warmup.py --preflight
PRE_EXIT=$?
if [ $PRE_EXIT -ne 0 ]; then
    echo "❌ preflight failed with code $PRE_EXIT — aborting before GPU work"
    exit $PRE_EXIT
fi

echo "--- [2/2] training ---"
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance_warmup.py ${ANCE_WARMUP_OVERWRITE:+--overwrite}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ warm-up completed. Next, IN ORDER:"
    echo "   1. Evaluate it — expect MRR@10 ~0.311 (EVAL_ALLOW_DRIFT is required:"
    echo "      the warm-up trains q128/p128 and is consumed at q64/p512):"
    echo "        EVAL_ALLOW_DRIFT=1 \\"
    echo "          EVAL_MODEL_PATH=\$DATA_BASE_DIR/models/ance_bm25_warmup_60k \\"
    echo "          sbatch scripts/launchers/eval_msmarco_singularity.sh"
    echo "   2. Record the sha256 printed above as"
    echo "      training.ance_paper.expected_init_sha256."
    echo "   3. sbatch scripts/launchers/run_ance_paper_singularity.sh"
else
    echo "❌ warm-up failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
