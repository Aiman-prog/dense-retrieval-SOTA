#!/usr/bin/env bash

#SBATCH --job-name=async_fg
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/async_fg_%j.out
#SBATCH --error=logs/async_fg_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Async Fast-GRASS (cached-MCDP), ANCE-style 2-process split:
#   trainer on GPU 0 — fresh-loss optimizer steps over mined rounds
#   miner   on GPU 1 — owns H/Z_mc/Z_mean, mines the next round from a recent
#                      checkpoint with periodic in-round cache maintenance
# TWO GPUs are required (--gpus-per-task=2). With one visible GPU both processes
# share it, which serialises them and defeats the point.
#
# The trainer never reads miner state; the miner never touches gradients. The only
# coupling is checkpoint-in / ready_N-out under temp_fast_grass_workdir/async_mining/.
#
# NOT ANCE mining: no full-corpus ANN rebuild, no per-query stale FAISS top-P.
# The stale index pickle is read ONCE to sample the initial H docids.
#
# R_doc (retired-document registry) is DEFERRED: replacement candidates are drawn
# uniformly from the corpus excluding H, then recertified against the query-MC
# reservoir. mining_meta reports the R counters as zero.
#
# PREREQUISITES:
#   - stale index at $DATA_BASE_DIR/temp_grass_workdir/stale_index/corpus.pkl
#   - processed training mixture (run_setup builds it if absent)

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

# --- Knobs (override via env vars before sbatch) ---
ASYNC_FG_RUN_TESTS="${ASYNC_FG_RUN_TESTS:-1}"   # CPU gate before any GPU work
ASYNC_FG_DEBUG="${ASYNC_FG_DEBUG:-}"            # set for a 512-item smoke run
ASYNC_FG_MAX_ROUNDS="${ASYNC_FG_MAX_ROUNDS:-}"  # stop the miner after N rounds
ASYNC_FG_NO_EVAL="${ASYNC_FG_NO_EVAL:-}"        # skip the BRIGHT eval at the end
ASYNC_FG_NO_COMPILE="${ASYNC_FG_NO_COMPILE:-}"  # disable torch.compile
ASYNC_FG_LAMBDA="${ASYNC_FG_LAMBDA:-}"          # override lambda_val (sweep arm)
ASYNC_FG_SUFFIX="${ASYNC_FG_SUFFIX:-}"          # isolate model dir + handoff root
ASYNC_FG_FRESH="${ASYNC_FG_FRESH:-}"            # wipe the handoff root before starting
ASYNC_FG_BOOTSTRAP_CKPT="${ASYNC_FG_BOOTSTRAP_CKPT:-}"  # extra early checkpoint (ABLATION)
ASYNC_FG_RECIPE="${ASYNC_FG_RECIPE:-}"          # training.<recipe> block; see below
ASYNC_FG_MANIFEST="${ASYNC_FG_MANIFEST:-}"      # pilot manifest JSONL

# RECIPES (config/config.yaml -> training.*), selected with ASYNC_FG_RECIPE:
#   async_fast_grass        full corrected run   (max_age_steps 1000, no gate)
#   async_fast_grass_pilot  10% lambda pilot     (1032 steps, gate >= 128 steps)
#   async_fast_grass_smoke  GPU wiring smoke     (64 steps, gate >= 1 step)
# The pilot/smoke recipes carry `pilot_gate_min_steps`, so the orchestrator evaluates
# the run validity gate and EXITS NONZERO if the run never consumed a refreshed mined
# round. The full recipe has no such key and its exit behaviour is unchanged.
#
# SLURM wall-clock is set per stage on the sbatch line rather than by editing this file:
#   sbatch --time=01:00:00 ...   # GPU smoke
#   sbatch --time=04:00:00 ...   # one pilot arm

# ASYNC_FG_BOOTSTRAP_CKPT: trainer saves ONE extra checkpoint at this step (e.g. 200)
# so the miner stops idling ~37 min at startup. Must be 0 < N < async_mine_every_steps
# or the trainer RAISES. This is NOT a free speedup: the extra mined round runs ~51
# maintenance intervals that mutate the PERSISTED cache and shift the weights, so the
# run trajectory forks permanently. Hold it constant across every arm of a sweep.

# LAMBDA SWEEP: every arm MUST set both ASYNC_FG_LAMBDA and a distinct
# ASYNC_FG_SUFFIX. Without the suffix all arms share one model dir and one handoff
# root, so they overwrite each other's checkpoints and mined rounds. The lambda is
# read here at SUBMIT-EXPANSION time and pinned onto the command line, so editing
# config.yaml while jobs sit in the queue cannot change what a queued arm runs.
#   ASYNC_FG_LAMBDA=0   ASYNC_FG_SUFFIX=lam0   sbatch scripts/run_async_fast_grass_singularity.sh
#   ASYNC_FG_LAMBDA=0.5 ASYNC_FG_SUFFIX=lam05  sbatch scripts/run_async_fast_grass_singularity.sh

mkdir -p logs

# An UNSET ASYNC_FG_MANIFEST makes `${ASYNC_FG_MANIFEST:+--manifest $X}` expand to
# nothing, so a pilot/smoke job would run against the full 330k mixture instead of its
# manifest — a different experiment that still looks healthy. Catch the typo here,
# before the queue, instead of relying on the Python check inside the job.
case "${ASYNC_FG_RECIPE}" in
  *_pilot|*_smoke)
    if [ -z "${ASYNC_FG_MANIFEST}" ]; then
        echo "❌ ASYNC_FG_RECIPE=${ASYNC_FG_RECIPE} requires ASYNC_FG_MANIFEST, which is empty."
        echo "   Did the shell variable get lost? Use an absolute path:"
        echo "   ASYNC_FG_MANIFEST=/scratch/\$USER/dense-retrieval-SOTA/data/processed/pilot_manifests/<name>.jsonl"
        exit 2
    fi
    if [ ! -f "${ASYNC_FG_MANIFEST}" ]; then
        echo "❌ ASYNC_FG_MANIFEST does not exist: ${ASYNC_FG_MANIFEST}"
        exit 2
    fi
    ;;
esac

echo "🚀 Async Fast-GRASS (cached-MCDP) — trainer GPU 0 / miner GPU 1"
echo "   debug=${ASYNC_FG_DEBUG:-off} | max_rounds=${ASYNC_FG_MAX_ROUNDS:-unbounded}"
echo "   lambda=${ASYNC_FG_LAMBDA:-<config>} | suffix=${ASYNC_FG_SUFFIX:-<none>} | bootstrap_ckpt=${ASYNC_FG_BOOTSTRAP_CKPT:-off}"
echo "   recipe=${ASYNC_FG_RECIPE:-async_fast_grass} | manifest=${ASYNC_FG_MANIFEST:-<full mixture>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- 1. CPU correctness gate (fail fast before burning 2 GPUs) ---
if [ -n "${ASYNC_FG_RUN_TESTS}" ]; then
    echo "--- [1/2] CPU correctness gate ---"
    singularity exec \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} bash -c '
            set -e
            python -u tests/async_fast_grass_handoff_test.py
            python -u tests/async_fast_grass_cache_semantics_test.py
            python -u tests/async_fast_grass_persistence_test.py
            python -u tests/async_fast_grass_pilot_test.py
            python -u tests/async_fast_grass_integration_smoke.py
            python -u tests/fast_grass_test.py
        '
    TEST_EXIT=$?
    if [ $TEST_EXIT -ne 0 ]; then
        echo "❌ correctness gate failed with code $TEST_EXIT — aborting before GPU work"
        exit $TEST_EXIT
    fi
else
    echo "--- [1/2] correctness gate SKIPPED (ASYNC_FG_RUN_TESTS blank) ---"
fi

# --- 1b. Preflight against the REAL processed corpus (no GPU) ---
# run_setup MD5-dedupes passages and remaps the corpus + qrels but NOT the mixture,
# so a positive whose text was a duplicate names a docid the strict trainer rejects.
# This is a step-0 failure, so catch it before the long job starts.
echo "--- [1b/2] preflight (mixture/corpus/qrels consistency) ---"
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_async_fast_grass.py --preflight \
        ${ASYNC_FG_DEBUG:+--debug} \
        ${ASYNC_FG_RECIPE:+--recipe $ASYNC_FG_RECIPE} \
        ${ASYNC_FG_MANIFEST:+--manifest $ASYNC_FG_MANIFEST}
PRE_EXIT=$?
if [ $PRE_EXIT -ne 0 ]; then
    echo "❌ preflight failed with code $PRE_EXIT — aborting before GPU work"
    exit $PRE_EXIT
fi

# --- 2. Async training (orchestrator spawns miner, runs trainer in foreground) ---
echo "--- [2/2] async training ---"
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/train_async_fast_grass.py \
        ${ASYNC_FG_RECIPE:+--recipe $ASYNC_FG_RECIPE} \
        ${ASYNC_FG_MANIFEST:+--manifest $ASYNC_FG_MANIFEST} \
        ${ASYNC_FG_DEBUG:+--debug} \
        ${ASYNC_FG_MAX_ROUNDS:+--max_rounds $ASYNC_FG_MAX_ROUNDS} \
        ${ASYNC_FG_NO_EVAL:+--no_eval} \
        ${ASYNC_FG_NO_COMPILE:+--no_compile} \
        ${ASYNC_FG_LAMBDA:+--lambda_val $ASYNC_FG_LAMBDA} \
        ${ASYNC_FG_SUFFIX:+--run_suffix $ASYNC_FG_SUFFIX} \
        ${ASYNC_FG_BOOTSTRAP_CKPT:+--bootstrap_checkpoint_step $ASYNC_FG_BOOTSTRAP_CKPT} \
        ${ASYNC_FG_FRESH:+--fresh}
RUN_EXIT=$?

echo "=========================================="
echo "Job $SLURM_JOB_ID completed with code $RUN_EXIT"
echo "Handoff artifacts: \$DATA_BASE_DIR/temp_fast_grass_workdir/async_mining${ASYNC_FG_SUFFIX:+_$ASYNC_FG_SUFFIX}/"
echo "Run summary:       \$DATA_BASE_DIR/models/<model_name>/async_run_summary.json"
echo "RUN_EXIT=1 with a PASS/FAIL gate block above means the run completed but is"
echo "  INVALID evidence about lambda (no refreshed mined round was trained on long"
echo "  enough). Do NOT submit the nonzero arms on a failed lambda=0 run."
echo "Tuning signals to read from the log:"
echo "  async_gap_steps / data_age_steps rising + miner_idle_time ~0"
echo "     => miner is the bottleneck: raise async_mine_every_steps or lower B_doc/T"
echo "  large rounds_skipped or miner_idle_time at a near-zero gap"
echo "     => trainer over-checkpoints: raise async_mine_every_steps"
echo "=========================================="

exit $RUN_EXIT
