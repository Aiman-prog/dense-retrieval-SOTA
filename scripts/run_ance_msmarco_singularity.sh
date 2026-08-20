#!/usr/bin/env bash

#SBATCH --job-name=ance-msmarco
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00              # ~16-18h estimated (BGE-M3 on A100 is ~5x faster than paper's RoBERTa on 4xV100)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2            # 1:1 Trainer:Inferencer GPU split (paper Appendix A.3)
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_msmarco_%j.out
#SBATCH --error=logs/ance_msmarco_%j.err
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
export OMP_NUM_THREADS=16

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Run ANCE MS MARCO Pipeline ---
echo "🚀 Starting ANCE MS MARCO Validation..."

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_ance.py --recipe ance_msmarco

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ANCE MS MARCO completed successfully"
else
    echo "❌ ANCE MS MARCO failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
