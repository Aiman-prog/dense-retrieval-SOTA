#!/usr/bin/env bash

#SBATCH --job-name=fg_timing
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fg_timing_%j.out
#SBATCH --error=logs/fg_timing_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Async Fast-GRASS — Phase 0 timing calibration (trainer-only + miner-only).
# Measures t_train_step and t_mine_round to CHOOSE async_mine_every_steps. No async
# trainer/miner, no online mining during training timing, no cache maintenance during
# training timing, no eval. Reuses the stale corpus pickle as the cache-init source
# only (no full-corpus ANN rebuild, no per-query FAISS top-P).
#
# Run one (B_doc, L, T) mine setting per job via env vars. The three required cluster
# settings (submit as three jobs):
#     FG_B_DOC=32000  FG_L=64  FG_T=3
#     FG_B_DOC=32000  FG_L=128 FG_T=3
#     FG_B_DOC=100000 FG_L=64  FG_T=3
# The trainer timing (t_train_step) is independent of L/T, so it is run ONCE here and
# its seconds_per_train_step is fed into the miner timing to emit async_mine_every_steps.
#
# Outputs (JSON + console): analysis/async_fast_grass_timing/

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Memory & Performance
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8
export CC=gcc

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Timing Knobs (override via env vars before sbatch) ---
FG_B_DOC="${FG_B_DOC:-32000}"                 # cache size H (--B_doc)
FG_L="${FG_L:-64}"                            # MCDP lazy top-L (--L)
FG_T="${FG_T:-3}"                             # MCDP dropout passes (--T)
FG_UNCERTAINTY="${FG_UNCERTAINTY:-mcdp}"      # mcdp (default) | ema
FG_TRAIN_STEPS="${FG_TRAIN_STEPS:-500}"       # timed trainer steps (doc: 500-1000)
FG_TRAIN_WARMUP="${FG_TRAIN_WARMUP:-20}"      # untimed trainer warmup steps
FG_MAX_QUERIES="${FG_MAX_QUERIES:-}"          # subset miner mixture (blank = full)
FG_SAFETY_MARGIN="${FG_SAFETY_MARGIN:-1.2}"   # async cadence margin (doc: 1.1-1.25)
FG_SKIP_TRAIN="${FG_SKIP_TRAIN:-}"            # set to skip trainer timing (mine only)
FG_TRAIN_TIMING_JSON="${FG_TRAIN_TIMING_JSON:-}"  # explicit train_timing_*.json for the
                                              # miner's seconds_per_train_step. Set this
                                              # with FG_SKIP_TRAIN=1 so the miner reads the
                                              # RIGHT trainer run, not the newest unrelated one.
FG_ROUND_SPAN="${FG_ROUND_SPAN:-}"            # override round_training_span_steps (--round_training_span_steps)
FG_WRITE_JSONL="${FG_WRITE_JSONL:-}"          # set to fold JSONL round-write into t_mine_round

mkdir -p logs analysis/async_fast_grass_timing

echo "⏱️  Fast-GRASS Phase-0 timing calibration"
echo "   B_doc=${FG_B_DOC} | L=${FG_L} | T=${FG_T} | UNC=${FG_UNCERTAINTY} | "\
"train_steps=${FG_TRAIN_STEPS} | max_queries=${FG_MAX_QUERIES:-full} | margin=${FG_SAFETY_MARGIN}"

# --- 1. Trainer-only timing (t_train_step); independent of L/T ---
if [ -z "${FG_SKIP_TRAIN}" ]; then
    echo "--- [1/2] trainer-only timing (t_train_step) ---"
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/fast_grass_train_timing.py \
            --steps ${FG_TRAIN_STEPS} \
            --warmup_steps ${FG_TRAIN_WARMUP} \
            ${FG_MAX_QUERIES:+--max_queries $FG_MAX_QUERIES}
    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ trainer timing failed with code $TRAIN_EXIT"; exit $TRAIN_EXIT
    fi
else
    echo "--- [1/2] trainer timing SKIPPED (FG_SKIP_TRAIN set) ---"
fi

# --- 2. Miner-only timing (t_mine_round) + async_mine_every_steps ---
# Reads the newest train_timing_*.json for seconds_per_train_step automatically.
echo "--- [2/2] miner-only timing (t_mine_round) ---"
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/fast_grass_mine_timing.py \
        --B_doc ${FG_B_DOC} \
        --L ${FG_L} \
        --T ${FG_T} \
        --uncertainty ${FG_UNCERTAINTY} \
        --safety_margin ${FG_SAFETY_MARGIN} \
        ${FG_MAX_QUERIES:+--max_queries $FG_MAX_QUERIES} \
        ${FG_TRAIN_TIMING_JSON:+--train_timing_json $FG_TRAIN_TIMING_JSON} \
        ${FG_ROUND_SPAN:+--round_training_span_steps $FG_ROUND_SPAN} \
        ${FG_WRITE_JSONL:+--write_jsonl_timing}
MINE_EXIT=$?

if [ $MINE_EXIT -eq 0 ]; then
    echo "✅ Timing calibration completed"
else
    echo "❌ miner timing failed with code $MINE_EXIT"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $MINE_EXIT
