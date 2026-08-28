#!/usr/bin/env bash

#SBATCH --job-name=rocketqa-a100-2048
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00             # bge-m3 from raw, 2 epochs, 2 GPUs, 2048 pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # 2 GPUs (like V100 test)
#SBATCH --cpus-per-task=8           # 2 tasks x 8 = 16 CPUs, matching every other GPU launcher
#SBATCH --gpus-per-task=1           # 2 GPUs total
#SBATCH --mem-per-cpu=8000M         # 16 x 8000M = 125GB. --mem-per-gpu=16GB gave 2 ranks 32GB
                                    # total while in-batch gets 125GB for one. (15039 died of
                                    # SIGBUS; see the note below -- not a memory term.)
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
# --- Experiment Knobs (override via env vars before sbatch) ---
# CROSSBATCH_RESUME=1     # continue a run whose manifest fingerprint matches
# CROSSBATCH_OVERWRITE=1  # discard an output dir built by a DIFFERENT config
# nproc_per_node MUST stay 2: train_crossbatch.py refuses any other world size,
# because a single process drops the all-gather and halves the negative pool.

# What the allocation actually granted. Kept for the record, but note the SIGBUS in
# 15039/18995 was NOT host memory and NOT the DataLoader: 18995 ran with
# dataloader_num_workers=0 and still took SIGBUS on both ranks at once, while in-batch
# 18996 ran 4 workers for 13h and completed. Cause was the Singularity image on
# /scratch going unreadable (EREMOTEIO, 0 bytes); singularity mmaps the SquashFS, so
# every rank faults together. See CONSOLIDATION_STATUS.md P14.
echo "[alloc] cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null \
    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown)"
echo "[alloc] /dev/shm: $(df -h /dev/shm | tail -1)"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    torchrun --nproc_per_node=2 scripts/train_crossbatch.py \
        ${CROSSBATCH_RESUME:+--resume} \
        ${CROSSBATCH_OVERWRITE:+--overwrite}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Cross-batch training completed successfully"
else
    echo "❌ Cross-batch training failed with code $EXIT_CODE"
fi

echo "Job Completed"

exit $EXIT_CODE
