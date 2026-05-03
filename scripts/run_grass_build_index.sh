#!/usr/bin/env bash

#SBATCH --job-name=grass_index
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_index_%j.out
#SBATCH --error=logs/grass_index_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=2

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "📦 Building stale ANN index from base model..."

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_grass.py --build_index_only

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Stale index built successfully"
else
    echo "❌ Index build failed with code $EXIT_CODE"
fi

exit $EXIT_CODE
