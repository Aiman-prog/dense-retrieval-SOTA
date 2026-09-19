#!/usr/bin/env bash

#SBATCH --job-name=eval-msmarco
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00   # the 8.8M-passage encode is 6.3h at the miner's measured
                          # 392 rows/s (p512, bs256 -- the same settings). 4h would be killed mid-encode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/eval_msmarco_%j.out
#SBATCH --error=logs/eval_msmarco_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
EVAL_RECIPE="${EVAL_RECIPE:-ance_paper}"
# The BM25 warm-up trains at q128/p128 but is USED by ance_paper at q64/p512, so
# encoding_contract_drift fires and eval_msmarco.py exits before encoding anything.
# Evaluating at q64/p512 is deliberate: it is the contract the warm-up is consumed
# under, and the one upstream's 0.311 refers to. Set EVAL_ALLOW_DRIFT=1 for it.
DRIFT_FLAG=""
if [ -n "${EVAL_ALLOW_DRIFT:-}" ]; then
    DRIFT_FLAG="--allow-config-drift"
fi
DEFAULT_MODEL_DIR="/scratch/${USER}/dense-retrieval-SOTA/models/ance_paper_roberta"

# An explicit path may be a released/final model without checkpoint-* children.
# With no override, preserve the old behavior and select the latest BGE checkpoint.
if [ -n "${EVAL_MODEL_PATH:-}" ]; then
    MODEL="${EVAL_MODEL_PATH}"
else
    MODEL=$(singularity exec --bind /scratch/${USER}:/scratch/${USER} "${CONTAINER}" \
        python -c 'from transformers.trainer_utils import get_last_checkpoint; import sys; print(get_last_checkpoint(sys.argv[1]))' \
        "${DEFAULT_MODEL_DIR}")
fi
echo "Evaluating checkpoint: ${MODEL}"

# get_last_checkpoint prints the string "None" when MODEL_DIR holds no checkpoint;
# without this guard the eval runs with --model_path None and fails obscurely.
if [ -z "${MODEL}" ] || [ "${MODEL}" = "None" ]; then
    echo "❌ no checkpoint found in ${MODEL_DIR} — nothing to evaluate"
    exit 2
fi

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    "${CONTAINER}" \
    python -u scripts/eval_msmarco.py \
        --model_path "${MODEL}" \
        --recipe "${EVAL_RECIPE}" ${DRIFT_FLAG}

EXIT_CODE=$?

echo "Done: $EXIT_CODE"

exit $EXIT_CODE
