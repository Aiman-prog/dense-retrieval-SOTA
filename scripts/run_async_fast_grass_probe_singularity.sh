#!/usr/bin/env bash

#SBATCH --job-name=fg_lambda_probe
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fg_lambda_probe_%j.out
#SBATCH --error=logs/fg_lambda_probe_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Cached-MCDP LAMBDA DOSAGE probe. ONE GPU: this measures selection, not training.
#
# Every lambda on the grid is scored from the SAME s_hat/sigma draw, so the whole grid
# costs one set of MC query encodes per seed. The dominant cost is building Z_mc once
# (B_doc docs x T dropout passes at passage_max_len), which is why 1 hour is adequate
# but not generous.
#
# REGIME CAVEAT: this runs on the BASE checkpoint with a freshly built cache, i.e. zero
# staleness, so sigma here is pure dropout noise. The flip-rate bands calibrate lambda
# DOSAGE only. Whether uncertainty improves retrieval is decided by the pilot arms and
# their BRIGHT evaluation, never by this script.
#
# PREREQUISITES:
#   - stale index at $DATA_BASE_DIR/temp_grass_workdir/stale_index/corpus.pkl
#   - pilot manifest built by scripts/async_fast_grass_pilot.py build-manifest

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

ASYNC_FG_RECIPE="${ASYNC_FG_RECIPE:-async_fast_grass_pilot}"
ASYNC_FG_MANIFEST="${ASYNC_FG_MANIFEST:-}"
PROBE_LAMBDA_GRID="${PROBE_LAMBDA_GRID:-0,0.1,0.2,0.3,0.5,0.7,1.0}"
PROBE_SEEDS="${PROBE_SEEDS:-3}"
PROBE_MAX_QUERIES="${PROBE_MAX_QUERIES:-2048}"
PROBE_QUERY_BATCH="${PROBE_QUERY_BATCH:-128}"

mkdir -p logs

echo "🔬 Cached-MCDP lambda dosage probe"
echo "   recipe=${ASYNC_FG_RECIPE} | manifest=${ASYNC_FG_MANIFEST:-<full mixture>}"
echo "   grid=${PROBE_LAMBDA_GRID} | seeds=${PROBE_SEEDS} | queries=${PROBE_MAX_QUERIES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# CPU smoke of the probe itself before touching the GPU
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/async_fast_grass_quality_probe.py --synthetic
SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo "❌ probe CPU smoke failed with code $SMOKE_EXIT — aborting before GPU work"
    exit $SMOKE_EXIT
fi

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/async_fast_grass_quality_probe.py --real \
        --recipe "${ASYNC_FG_RECIPE}" \
        ${ASYNC_FG_MANIFEST:+--manifest $ASYNC_FG_MANIFEST} \
        --lambda_grid "${PROBE_LAMBDA_GRID}" \
        --seeds "${PROBE_SEEDS}" \
        --max_queries "${PROBE_MAX_QUERIES}" \
        --query_batch_size "${PROBE_QUERY_BATCH}"
RUN_EXIT=$?

echo "=========================================="
echo "Job $SLURM_JOB_ID completed with code $RUN_EXIT"
echo "Report: analysis/async_fast_grass_timing/lambda_probe_<timestamp>.json"
echo "Read selected_low / selected_medium / band_satisfied / n_arms."
echo "  n_arms == 1 -> submit ONE nonzero pilot arm, not two."
echo "  band_satisfied false -> the grid never reached the intended dosage; the"
echo "  selection is a nearest-neighbour fallback, so say so when reporting it."
echo "=========================================="

exit $RUN_EXIT
