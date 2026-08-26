#!/usr/bin/env bash

#SBATCH --job-name=eval-reasonir
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment Setup ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline Mode
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- CONFIGURATION ---
MODEL_PATH="${EVAL_MODEL_PATH:?Error: EVAL_MODEL_PATH must be set}"
# Comma-separated subset, DEFAULTING to the four lambda-pilot development domains so a
# pilot evaluation never has to spell them out (and can never quietly disagree with the
# set the decision rule reads). Evaluating all twelve costs ~3x more GPU hours and is not
# what the pilot compares, so it must be asked for explicitly:
#   EVAL_DOMAINS=all                      -> all twelve (config.yaml evaluation.eval_domains)
#   EVAL_DOMAINS=biology,economics        -> that subset
PILOT_DOMAINS="biology,economics,stackoverflow,theoremqa_questions"
EVAL_DOMAINS="${EVAL_DOMAINS:-$PILOT_DOMAINS}"
# `all` is the escape hatch: an empty --domains makes run_all_evals use every domain.
if [ "${EVAL_DOMAINS}" = "all" ]; then
    EVAL_DOMAINS=""
fi
# Refuse to build missing BRIGHT domain files. Set for the pilot so an evaluation job
# can never regenerate processed data as a side effect mid-experiment.
EVAL_REQUIRE_EXISTING="${EVAL_REQUIRE_EXISTING:-}"

# --- Create output directories ---
mkdir -p logs

# --- Run Evaluation in Container ---
# K and batch_size are read from config/config.yaml by the Python scripts
echo "🔍 Starting evaluation for model: ${MODEL_PATH}"
echo "📊 Domains: ${EVAL_DOMAINS:-all (config.yaml evaluation.eval_domains)}"
echo "📊 require_existing: ${EVAL_REQUIRE_EXISTING:-off}"

# The summary path is NOT computed here: two models can share a basename, and this
# line used to point both of them at the same summary.json. run_all_evals.py writes
# it under the hashed run tag and prints the path.
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_all_evals.py \
        --model_path "${MODEL_PATH}" \
        ${EVAL_DOMAINS:+--domains $EVAL_DOMAINS} \
        ${EVAL_REQUIRE_EXISTING:+--require_existing}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
else
    echo "❌ Evaluation failed with code $EXIT_CODE"
fi

exit $EXIT_CODE