#!/usr/bin/env bash

#SBATCH --job-name=ance-reasonir
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00              # Async ANCE: encoding passes add significant time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # More CPUs for FAISS operations
#SBATCH --gpus-per-task=2            # 1:1 Trainer:Inferencer GPU split (paper Appendix A.3)
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_%j.out
#SBATCH --error=logs/ance_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance Tuning
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16            # Matches cpus-per-task for FAISS speed

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Run ANCE Pipeline in Container ---
echo "🚀 Starting Iterative ANCE Loop..."
echo "📋 ANCE will iteratively: train → encode → mine → repeat"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ANCE training completed successfully"
else
    echo "❌ ANCE training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
