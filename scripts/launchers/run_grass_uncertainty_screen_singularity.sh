#!/usr/bin/env bash

# Phase B: the GRASS uncertainty estimator screen.
#
# Asks whether MC-dropout representation dispersion predicts future embedding movement
# BEYOND staleness and momentum, using saved ance_mixed_bge_m3 checkpoints. No training,
# no writes into any model directory, nothing deleted -- it reads three checkpoints and the
# processed corpus and writes one analysis directory.
#
# Everything downstream in docs/grass_uncertainty_refresh_architecture.md is gated on this.
#
#   sbatch scripts/launchers/run_grass_uncertainty_screen_singularity.sh
#   OPEN_TEST=1 sbatch scripts/launchers/run_grass_uncertainty_screen_singularity.sh
#
# Run it WITHOUT OPEN_TEST first. The held-out split is read once, after T is chosen on
# validation; opening it early spends the only unbiased read this screen gets.

#SBATCH --job-name=grass-screen
#SBATCH --partition=gpu-a100-small   # schedules in minutes where gpu-a100 queues. NOTE: this
                                     # is a ~9.5 GiB MIG SLICE, not a whole A100, and job
                                     # 284893 OOMed at batch 128 -- see SCREEN_BATCH below.
#SBATCH --time=04:00:00              # the partition cap. Job 284893 measured 463 s per
                                     # 20k-doc pass, so 11 passes is ~1.5 h before overhead.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # partition cap
#SBATCH --gpus-per-task=1            # partition cap
#SBATCH --mem-per-cpu=8000M          # 2 x 8000M = 16 GB, the partition ceiling. The screen
                                     # streams the corpus twice and keeps only 20k texts,
                                     # so it never builds the full 648,942-doc lookup.
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/grass_screen_%j.out
#SBATCH --error=logs/grass_screen_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

set -u

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH:-}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=2             # matches --cpus-per-task

# This screen produces a REPORTED result, so its revision is recorded on the host, where
# git exists -- the container has no git binary.
source scripts/launchers/runtime_provenance.sh || exit 1

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
MODEL_DIR="${DATA_BASE_DIR}/models/ance_mixed_bge_m3"

mkdir -p logs

# The three checkpoints ARE the experiment: they supply the momentum control and the drift
# label. Fail here rather than after loading the corpus.
for STEP in 3000 4000 5000; do
    if [ ! -d "${MODEL_DIR}/checkpoint-${STEP}" ]; then
        echo "❌ ${MODEL_DIR}/checkpoint-${STEP} is missing — the screen cannot label drift."
        echo "   Present checkpoints:"
        ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null || echo "   (none)"
        exit 2
    fi
done

EXTRA=""
if [ -n "${OPEN_TEST:-}" ]; then
    EXTRA="--open-test"
    echo "🔓 opening the HELD-OUT split — this is the single unbiased read."
else
    echo "🔒 calibration + validation only; held-out split stays closed."
fi

echo "📍 model dir : ${MODEL_DIR}"
echo "📍 output    : analysis/grass_estimator_screen/"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/grass_uncertainty_screen.py \
        --model_dir  "${MODEL_DIR}" \
        --batch_size "${SCREEN_BATCH:-32}" ${EXTRA}

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ screen complete — read analysis/grass_estimator_screen/*/screen.json"
    echo "   gate: incremental_spearman with a 95% lower bound above zero"
else
    echo "❌ screen failed with exit code ${EXIT_CODE} (traceback is in the .err)"
fi
exit $EXIT_CODE
