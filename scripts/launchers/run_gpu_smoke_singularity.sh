#!/usr/bin/env bash

# Memory/throughput smoke for the p1024 transition. Verifies the configured BGE
# q1024/p1024 training and encoding batches fit in 80 GiB.
# Nothing here is a result; no checkpoint is written.

#SBATCH --job-name=gpu-smoke
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/gpu_smoke_%j.out
#SBATCH --error=logs/gpu_smoke_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/dev/gpu_memory_smoke.py ${SMOKE_ARMS:+--arms $SMOKE_ARMS}

SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
    echo "GPU smoke failed with code $SMOKE_RC"
    exit $SMOKE_RC
fi

echo "Done"
