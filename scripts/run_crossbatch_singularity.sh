#!/usr/bin/env bash

#SBATCH --job-name=rocketqa-a100-2048
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00             # bge-m3 from raw, 2 epochs, 2 GPUs — 12h ceiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # 2 GPUs (like V100 test)
#SBATCH --cpus-per-task=4          # 2 CPUs per GPU (for 4 workers)
#SBATCH --gpus-per-task=1           # 2 GPUs total
#SBATCH --mem-per-gpu=16GB           # 16GB RAM per GPU
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/crossbatch_bge_%j.out
#SBATCH --error=logs/crossbatch_bge_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA Configuration for A100
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Run Training in Container ---
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    torchrun --nproc_per_node=2 scripts/train_crossbatch.py

echo "Job Completed"
