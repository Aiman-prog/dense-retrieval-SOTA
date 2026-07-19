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

# Async Fast-GRASS — Phase 0 feasibility gate (run BEFORE building the async trainer/miner).
# One job answers "is async worth it, and at what cadence?" in five steps:
#   [1/5] CPU correctness gate  : handoff + cache-semantics tests (+ synthetic smokes).
#                                 Fail-fast so a logic bug never burns GPU time.
#   [2/5] trainer-only timing   : t_train_step (seconds_per_train_step). No mining/cache/eval.
#   [3/5] miner-only timing     : t_mine_round over the mixture with periodic in-round
#                                 maintenance. No full-corpus ANN, no per-query FAISS top-P.
#   [4/5] speed estimate        : expected overlap speedup + recommended async_mine_every_steps
#                                 + staleness warning, from the two JSONs above.
#   [5/5] quality probe (opt)   : real hardness/diversity of frozen-checkpoint mining vs
#                                 current-model mining (needs checkpoints; off by default).
#
# Run one (B_doc, L, T) mine setting per job via env vars. The three required cluster
# settings (submit as three jobs):
#     FG_B_DOC=32000  FG_L=64  FG_T=3
#     FG_B_DOC=32000  FG_L=128 FG_T=3
#     FG_B_DOC=100000 FG_L=64  FG_T=3
# The trainer timing (t_train_step) is independent of L/T, so it is run ONCE here and
# its seconds_per_train_step is fed into the miner timing and the speed estimate.
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

# Speed-estimate + quality-probe knobs
FG_RUN_TESTS="${FG_RUN_TESTS:-1}"             # [1/5] CPU correctness gate (blank = skip)
FG_NUM_EPOCHS="${FG_NUM_EPOCHS:-3}"           # planned training epochs (speed estimate)
FG_CKPT_WRITE="${FG_CKPT_WRITE:-}"            # checkpoint write seconds (async handoff I/O); blank = excluded
FG_MIN_SPEEDUP="${FG_MIN_SPEEDUP:-1.3}"       # acceptance threshold for expected speedup
FG_QUALITY_PROBE="${FG_QUALITY_PROBE:-}"      # set to run [5/5] real quality probe (needs a checkpoint)
FG_SEQ_CKPT="${FG_SEQ_CKPT:-}"                # quality probe: current-model dir (blank = base model)
FG_ASYNC_CKPT="${FG_ASYNC_CKPT:-}"            # quality probe: frozen/stale model dir (blank = base+noise)
FG_STALENESS_NOISE="${FG_STALENESS_NOISE:-0.0}"  # quality probe: weight noise if no async ckpt

mkdir -p logs analysis/async_fast_grass_timing

echo "⏱️  Fast-GRASS Phase-0 timing calibration"
echo "   B_doc=${FG_B_DOC} | L=${FG_L} | T=${FG_T} | UNC=${FG_UNCERTAINTY} | "\
"train_steps=${FG_TRAIN_STEPS} | max_queries=${FG_MAX_QUERIES:-full} | margin=${FG_SAFETY_MARGIN}"

# --- 1. CPU correctness gate (fail-fast before any GPU work) ---
if [ -n "${FG_RUN_TESTS}" ]; then
    echo "--- [1/5] CPU correctness gate (handoff + cache semantics + synthetic smokes) ---"
    singularity exec \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} bash -c '
            set -e
            python -u scripts/async_fast_grass_handoff_test.py
            python -u scripts/async_fast_grass_cache_semantics_test.py
            python -u scripts/async_fast_grass_quality_probe.py --synthetic
        '
    TEST_EXIT=$?
    if [ $TEST_EXIT -ne 0 ]; then
        echo "❌ correctness gate failed with code $TEST_EXIT — aborting before GPU work"
        exit $TEST_EXIT
    fi
else
    echo "--- [1/5] correctness gate SKIPPED (FG_RUN_TESTS blank) ---"
fi

# --- 2. Trainer-only timing (t_train_step); independent of L/T ---
if [ -z "${FG_SKIP_TRAIN}" ]; then
    echo "--- [2/5] trainer-only timing (t_train_step) ---"
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
    echo "--- [2/5] trainer timing SKIPPED (FG_SKIP_TRAIN set) ---"
fi

# --- 3. Miner-only timing (t_mine_round) + async_mine_every_steps ---
# Reads the newest train_timing_*.json for seconds_per_train_step automatically.
echo "--- [3/5] miner-only timing (t_mine_round) ---"
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

if [ $MINE_EXIT -ne 0 ]; then
    echo "❌ miner timing failed with code $MINE_EXIT"
    exit $MINE_EXIT
fi

# --- 4. Speed estimate (CPU): expected speedup + recommended async_mine_every_steps ---
# Reads the newest train_timing_*.json + mine_timing_*.json this job just produced.
echo "--- [4/5] speed estimate (expected speedup + cadence) ---"
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/async_fast_grass_speed_estimate.py \
        --num_epochs ${FG_NUM_EPOCHS} \
        --safety_margin ${FG_SAFETY_MARGIN} \
        --min_speedup ${FG_MIN_SPEEDUP} \
        ${FG_TRAIN_TIMING_JSON:+--train_timing_json $FG_TRAIN_TIMING_JSON} \
        ${FG_CKPT_WRITE:+--checkpoint_write_time $FG_CKPT_WRITE}
EST_EXIT=$?
[ $EST_EXIT -ne 0 ] && echo "⚠️  speed estimate exited $EST_EXIT (analysis step; timing JSONs are still valid)"

# --- 5. Real quality probe (optional; needs a checkpoint) ---
if [ -n "${FG_QUALITY_PROBE}" ]; then
    echo "--- [5/5] real quality probe (frozen vs current mining) ---"
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/async_fast_grass_quality_probe.py --real \
            --B_doc ${FG_B_DOC} --L ${FG_L} --T ${FG_T} \
            --staleness_noise ${FG_STALENESS_NOISE} \
            ${FG_MAX_QUERIES:+--max_queries $FG_MAX_QUERIES} \
            ${FG_SEQ_CKPT:+--seq_checkpoint $FG_SEQ_CKPT} \
            ${FG_ASYNC_CKPT:+--async_checkpoint $FG_ASYNC_CKPT}
    echo "   quality probe exited $?"
else
    echo "--- [5/5] quality probe SKIPPED (set FG_QUALITY_PROBE=1 to enable) ---"
fi

echo "✅ Async feasibility gate completed"
echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $MINE_EXIT
