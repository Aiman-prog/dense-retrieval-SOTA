#!/usr/bin/env bash
# Two-Set Cached GRASS feasibility (NO training) on the small-GPU partition.
# gpu-a100-small policy: <=4h, 1 GPU, <=10GB VRAM, <=2 CPU cores.

#SBATCH --job-name=grass_feas
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_feas_%j.out
#SBATCH --error=logs/grass_feas_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance (2 cores → cap thread pools so FAISS/torch don't oversubscribe)
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Knobs (override via env before sbatch) ---
# FEAS_RACTIVE="5 10 20"     # active-set sweep
# FEAS_P0=200                # C_q size
# FEAS_KREFRESH=5            # k_refresh for reserve test
# FEAS_MAXQ=0               # subsample queries (0=all); use small for a dry run
# FEAS_BATCH=256            # encode batch size (keep <=10GB VRAM)
# FEAS_CACHE_GB=10          # T6 budget: max Z_H cache size (bf16 GB)
# FEAS_FUNCTIONAL=1         # also run real-MCDP T8b
# FEAS_FAISS_CPU=1          # force CPU faiss search (default: GPU torch matmul)

RACTIVE="${FEAS_RACTIVE:-5 10 20}"
P0="${FEAS_P0:-200}"
KREFRESH="${FEAS_KREFRESH:-5}"
MAXQ="${FEAS_MAXQ:-0}"
BATCH="${FEAS_BATCH:-256}"
CACHE_GB="${FEAS_CACHE_GB:-10}"

mkdir -p logs

echo "🌿 Two-Set GRASS feasibility | R_active=[${RACTIVE}] P0=${P0} k_refresh=${KREFRESH} maxq=${MAXQ} batch=${BATCH} cache_gb=${CACHE_GB}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/grass_twoset_feasibility.py \
        --r_active ${RACTIVE} \
        --P0 ${P0} \
        --k_refresh ${KREFRESH} \
        --mc_batch_size ${BATCH} \
        --max_cache_gb ${CACHE_GB} \
        ${MAXQ:+--max_queries ${MAXQ}} \
        ${FEAS_FUNCTIONAL:+--functional_test} \
        ${FEAS_FAISS_CPU:+--faiss_cpu}

EXIT_CODE=$?
echo "=========================================="
echo "Feasibility job $SLURM_JOB_ID exit=${EXIT_CODE}"
echo "=========================================="
exit $EXIT_CODE
