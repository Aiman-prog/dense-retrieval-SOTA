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
# Times CACHED-MCDP: Z_mc[T, B_doc, D] built once, then T query passes + T matmuls over
# all of H per batch with ZERO document encodes during mining. The old lazy top-L path
# is oracle-only and has no launcher path here (there is no FG_L knob any more).
#
# One job answers "is async worth it, and at what cadence?" in five steps:
#   [1/5] CPU correctness gate  : handoff + cache-semantics tests (+ synthetic smokes).
#                                 Fail-fast so a logic bug never burns GPU time.
#   [2/5] trainer-only timing   : t_train_step + checkpoint_write_time + peak memory.
#                                 No mining/cache/eval.
#   [3/5] miner-only timing     : t_mine_round over the mixture with PERIODIC IN-ROUND
#                                 maintenance every cache_update_interval*batch_size
#                                 mined queries. No full-corpus ANN, no per-query FAISS.
#   [4/5] speed estimate        : expected speedup, recommended async_mine_every_steps,
#                                 memory gate, and the go/no-go table.
#   [5/5] signal probe (opt)    : cached-MCDP lambda=0 vs lambda>0 non-degeneracy
#                                 diagnostic. REPORT-ONLY, never gates the build, and
#                                 supports NO Recall/NDCG claim. Off by default.
#
# Run one (B_doc, T) mine setting per job via env vars. The GATE setting is 32k/T=3;
# only run the others if it clears:
#     FG_B_DOC=32000  FG_T=3     <- gate
#     FG_B_DOC=100000 FG_T=3
#     FG_B_DOC=32000  FG_T=5
# The trainer timing (t_train_step) is independent of B_doc/T, so it is run ONCE here
# and its seconds_per_train_step is fed into the miner timing and the speed estimate.
#
# NOTE: FG_MAX_QUERIES must cross at least TWO maintenance thresholds (ordinarily
# >= 12800 = 2 * 100 * 64) or per-interval maintenance cost is a single sample and
# t_mine_round extrapolation is flagged unreliable. Blank = full mixture (best).
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
FG_B_DOC="${FG_B_DOC:-32000}"                 # cache size H (--B_doc); gate = 32000
FG_T="${FG_T:-3}"                             # cached MC passes, doc AND query (--T)
FG_LAMBDA="${FG_LAMBDA:-}"                    # g = s_hat + lambda*sigma (async default 0.5)
FG_SOURCE_CKPT_STEP="${FG_SOURCE_CKPT_STEP:-}"  # model time for the round: cache age, rho and
                                              # last_refreshed_step all use it. Blank =
                                              # steps_per_epoch. NEVER 0 (every slot would
                                              # have age 0 and refresh work is under-measured).
FG_CHUNK_SIZE="${FG_CHUNK_SIZE:-}"            # chunk Q x Z_mc over cache slots (blank = no chunking)
# config.yaml holds the SEQUENTIAL maintenance defaults (rho_end 0.10, max_age_epochs 4).
# The miner timing applies the ASYNC doc defaults (0.25 / 2) unless overridden here —
# max_age_epochs 4->2 halves the age threshold and materially raises refresh work.
FG_RHO_START="${FG_RHO_START:-}"              # blank = async default 0.50
FG_RHO_END="${FG_RHO_END:-}"                  # blank = async default 0.25
FG_MAX_AGE_EPOCHS="${FG_MAX_AGE_EPOCHS:-}"    # blank = async default 2
FG_TRAIN_STEPS="${FG_TRAIN_STEPS:-500}"       # timed trainer steps (doc: 500-1000)
FG_TRAIN_WARMUP="${FG_TRAIN_WARMUP:-20}"      # untimed trainer warmup steps
FG_MAX_QUERIES="${FG_MAX_QUERIES:-}"          # subset miner mixture (blank = full; if set,
                                              # use >= 12800 to cross 2 maintenance intervals)
FG_SAFETY_MARGIN="${FG_SAFETY_MARGIN:-1.2}"   # async cadence margin (doc: 1.1-1.25)
FG_SKIP_TRAIN="${FG_SKIP_TRAIN:-}"            # set to skip trainer timing (mine only)
FG_TRAIN_TIMING_JSON="${FG_TRAIN_TIMING_JSON:-}"  # explicit train_timing_*.json for the
                                              # miner's seconds_per_train_step. Set this
                                              # with FG_SKIP_TRAIN=1 so the miner reads the
                                              # RIGHT trainer run, not the newest unrelated one.
FG_WRITE_JSONL="${FG_WRITE_JSONL:-}"          # set to fold JSONL round-write into t_mine_round

# Speed-estimate + signal-probe knobs
FG_RUN_TESTS="${FG_RUN_TESTS:-1}"             # [1/5] CPU correctness gate (blank = skip)
FG_NUM_EPOCHS="${FG_NUM_EPOCHS:-3}"           # planned training epochs (speed estimate)
FG_CKPT_WRITE="${FG_CKPT_WRITE:-}"            # override checkpoint write seconds; blank =
                                              # use the value [2/5] measured
FG_MIN_SPEEDUP="${FG_MIN_SPEEDUP:-1.3}"       # acceptance threshold for expected speedup
FG_GPU_CAPACITY="${FG_GPU_CAPACITY:-80e9}"    # miner GPU bytes for the memory gate (A100 80GB)
FG_MAX_MEM_FRAC="${FG_MAX_MEM_FRAC:-0.85}"    # memory gate: peak reserved <= this fraction
FG_SIGNAL_PROBE="${FG_SIGNAL_PROBE:-}"        # set to run [5/5] real signal probe (REPORT-ONLY)
FG_PROBE_SEEDS="${FG_PROBE_SEEDS:-5}"         # signal probe: MC draws (signal vs noise)
FG_PROBE_QUERIES="${FG_PROBE_QUERIES:-256}"   # signal probe: query sample size

mkdir -p logs analysis/async_fast_grass_timing

echo "⏱️  Fast-GRASS Phase-0 timing calibration (cached-MCDP)"
echo "   B_doc=${FG_B_DOC} | T=${FG_T} | lambda=${FG_LAMBDA:-config} | "\
"train_steps=${FG_TRAIN_STEPS} | max_queries=${FG_MAX_QUERIES:-full} | margin=${FG_SAFETY_MARGIN}"

# --- 1. CPU correctness gate (fail-fast before any GPU work) ---
if [ -n "${FG_RUN_TESTS}" ]; then
    echo "--- [1/5] CPU correctness gate (handoff + cache semantics + synthetic smokes) ---"
    singularity exec \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} bash -c '
            set -e
            python -u tests/async_fast_grass_handoff_test.py
            python -u tests/async_fast_grass_cache_semantics_test.py
            python -u tests/fast_grass_test.py
            python -u scripts/dev/fast_grass_mine_timing.py --synthetic
            python -u scripts/dev/async_fast_grass_quality_probe.py --synthetic
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
        python -u scripts/dev/fast_grass_train_timing.py \
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
    python -u scripts/dev/fast_grass_mine_timing.py \
        --B_doc ${FG_B_DOC} \
        --T ${FG_T} \
        --safety_margin ${FG_SAFETY_MARGIN} \
        ${FG_LAMBDA:+--lambda_val $FG_LAMBDA} \
        ${FG_SOURCE_CKPT_STEP:+--source_checkpoint_step $FG_SOURCE_CKPT_STEP} \
        ${FG_CHUNK_SIZE:+--chunk_size $FG_CHUNK_SIZE} \
        ${FG_RHO_START:+--rho_start $FG_RHO_START} \
        ${FG_RHO_END:+--rho_end $FG_RHO_END} \
        ${FG_MAX_AGE_EPOCHS:+--max_age_epochs $FG_MAX_AGE_EPOCHS} \
        ${FG_MAX_QUERIES:+--max_queries $FG_MAX_QUERIES} \
        ${FG_TRAIN_TIMING_JSON:+--train_timing_json $FG_TRAIN_TIMING_JSON} \
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
    python -u scripts/dev/async_fast_grass_speed_estimate.py \
        --num_epochs ${FG_NUM_EPOCHS} \
        --safety_margin ${FG_SAFETY_MARGIN} \
        --min_speedup ${FG_MIN_SPEEDUP} \
        --gpu_capacity_bytes ${FG_GPU_CAPACITY} \
        --max_mem_fraction ${FG_MAX_MEM_FRAC} \
        ${FG_TRAIN_TIMING_JSON:+--train_timing_json $FG_TRAIN_TIMING_JSON} \
        ${FG_CKPT_WRITE:+--checkpoint_write_time $FG_CKPT_WRITE}
EST_EXIT=$?
[ $EST_EXIT -ne 0 ] && echo "⚠️  speed estimate exited $EST_EXIT (analysis step; timing JSONs are still valid)"

# --- 5. Cached-MCDP signal probe (optional; REPORT-ONLY, never gates the build) ---
if [ -n "${FG_SIGNAL_PROBE}" ]; then
    echo "--- [5/5] cached-MCDP signal probe (lambda=0 vs lambda>0) ---"
    echo "    REPORT-ONLY: non-degeneracy diagnostic on a frozen base model."
    echo "    It supports NO Recall/NDCG claim and does not gate the build."
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/dev/async_fast_grass_quality_probe.py --real \
            --B_doc ${FG_B_DOC} --T ${FG_T} \
            --seeds ${FG_PROBE_SEEDS} \
            --max_queries ${FG_PROBE_QUERIES} \
            ${FG_LAMBDA:+--lambda_val $FG_LAMBDA}
    echo "   signal probe exited $? (non-gating)"
else
    echo "--- [5/5] signal probe SKIPPED (set FG_SIGNAL_PROBE=1 to enable) ---"
fi

echo "✅ Async feasibility gate completed"
echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $MINE_EXIT
