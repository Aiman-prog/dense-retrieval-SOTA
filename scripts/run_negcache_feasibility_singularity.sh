#!/usr/bin/env bash
# Negative-Cache Fast-GRASS feasibility (NO training) on the FULL-A100 partition.
# Run on gpu-a100 (same hardware as run_grass_singularity.sh) so the Fast-GRASS
# vs current-GRASS speedup ratio is measured on the target GPU + CPU allocation.

#SBATCH --job-name=negcache_feas
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/negcache_feas_%j.out
#SBATCH --error=logs/negcache_feas_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance (16 cores → match run_grass.py so CPU-bound parts
# don't skew the speedup ratio vs the full-A100 training run)
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Knobs (override via env before sbatch) ---
# NC_BUDGET_FRACS="0.05 0.10 0.20"  # cache-budget sweep (T1)
# NC_BDOC_FRAC=0.10                 # operating cache fraction (T2/T3/T5/T7/T8)
# NC_P0=200                         # top stale hits per query (frequency basis)
# NC_BATCHSIZES="64 128"            # scoring-throughput batch sizes (T3)
# NC_MINEQ=3000                     # mining sample (T8)
# NC_MAXQ=0                         # subsample queries (0=all); small for a dry run
# NC_BATCH=256                      # encode batch size (keep <=10GB VRAM)
# NC_CACHE_GB=10                    # T1 budget: max Z_H size (bf16 GB)
# NC_SCORE_MIN=15                   # T3 budget: max full-epoch scoring minutes
# NC_ENCODE_PROBE=512               # T2 real doc-encode probe size
# NC_SKIP_ENCODE=1                  # T2: analytic only (skip real doc encoding)
# NC_MINIBATCH=1                    # also run real one-minibatch e2e (T7)
# NC_FAISS_CPU=1                    # force CPU faiss search (default: GPU torch matmul)

BUDGET_FRACS="${NC_BUDGET_FRACS:-0.05 0.10 0.20}"
BDOC_FRAC="${NC_BDOC_FRAC:-0.10}"
P0="${NC_P0:-200}"
BATCHSIZES="${NC_BATCHSIZES:-64 128}"
MINEQ="${NC_MINEQ:-3000}"
MAXQ="${NC_MAXQ:-0}"
BATCH="${NC_BATCH:-256}"
CACHE_GB="${NC_CACHE_GB:-10}"
SCORE_MIN="${NC_SCORE_MIN:-15}"
ENCODE_PROBE="${NC_ENCODE_PROBE:-512}"

mkdir -p logs

echo "🌿 NegCache Fast-GRASS feasibility | budget_fracs=[${BUDGET_FRACS}] b_doc=${BDOC_FRAC} P0=${P0} maxq=${MAXQ} batch=${BATCH} cache_gb=${CACHE_GB}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/grass_negcache_feasibility.py \
        --budget_fracs ${BUDGET_FRACS} \
        --b_doc_frac ${BDOC_FRAC} \
        --P0 ${P0} \
        --batch_sizes ${BATCHSIZES} \
        --mine_queries ${MINEQ} \
        --mc_batch_size ${BATCH} \
        --max_cache_gb ${CACHE_GB} \
        --max_epoch_scoring_min ${SCORE_MIN} \
        --encode_probe ${ENCODE_PROBE} \
        ${MAXQ:+--max_queries ${MAXQ}} \
        ${NC_SKIP_ENCODE:+--skip_encode_test} \
        ${NC_MINIBATCH:+--minibatch_test} \
        ${NC_FAISS_CPU:+--faiss_cpu}

EXIT_CODE=$?
echo "=========================================="
echo "NegCache feasibility job $SLURM_JOB_ID exit=${EXIT_CODE}"
echo "=========================================="
exit $EXIT_CODE
