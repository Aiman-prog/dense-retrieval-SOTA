#!/usr/bin/env bash

#SBATCH --job-name=eval-msmarco
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
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
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
MODEL_DIR="/scratch/${USER}/dense-retrieval-SOTA/models/ance_msmarco_bge_m3"
MODEL=$(singularity exec --bind /scratch/${USER}:/scratch/${USER} ${CONTAINER} \
    python -c "from transformers.trainer_utils import get_last_checkpoint; print(get_last_checkpoint('${MODEL_DIR}'))")
echo "Evaluating checkpoint: ${MODEL}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/eval_msmarco.py \
        --model_path ${MODEL} \
        --recipe ance_msmarco

echo "Done: $?"
