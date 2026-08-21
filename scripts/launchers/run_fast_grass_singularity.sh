#!/usr/bin/env bash

#SBATCH --job-name=fast_grass
#SBATCH --partition=gpu-a100
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/fast_grass_%j.out
#SBATCH --error=logs/fast_grass_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

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

# Container path
CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

# --- Experiment Knobs (override via env vars before sbatch) ---
# FAST_GRASS_DEBUG=1      # 512-item mixture smoke run
# FAST_GRASS_MODEL_SUFFIX=run1     # appended to model output dir
# FAST_GRASS_NUM_EPOCHS=1          # override config
# FAST_GRASS_B_DOC=100000          # global cache size (ablate 32k / 100k / 512k)
# FAST_GRASS_LAMBDA=1.0            # g = s_hat + lambda * sigma (baseline ablation = 0)
# FAST_GRASS_UNCERTAINTY=mcdp      # mcdp (teacher-free, default) | ema (baseline)
# FAST_GRASS_T=3                   # MCDP stochastic dropout passes
# FAST_GRASS_MC_DROPOUT_P=0.3      # MCDP dropout probability
# FAST_GRASS_L=128                 # MCDP top-L shortlist / softmax prefilter (cost ~ batch*L*T)
# FAST_GRASS_EMA_ALPHA=1.0         # ema mode only: teacher decay; 1.0 = frozen base teacher
# FAST_GRASS_SELECTION_MODE=topk   # topk | softmax
# FAST_GRASS_M=1                   # negatives per query
# FAST_GRASS_NO_REGISTRY=1         # ablation: fully disable the retired registry R
# FAST_GRASS_NO_EVAL=1             # skip post-train BRIGHT eval (run it later, sequentially,
#                                 #   via run_all_evals.py / run_evaluate_singularity.sh — required
#                                 #   for parallel sweeps, the eval scratch dir is shared across runs)

mkdir -p logs

echo "🌿 Starting Fast-GRASS Training (1-GPU, Algorithm 1 over the global cache)..."
echo "   SUFFIX=${FAST_GRASS_MODEL_SUFFIX:-} | EPOCHS=${FAST_GRASS_NUM_EPOCHS:-cfg} | B_doc=${FAST_GRASS_B_DOC:-cfg} | LAMBDA=${FAST_GRASS_LAMBDA:-cfg} | UNC=${FAST_GRASS_UNCERTAINTY:-cfg} | T=${FAST_GRASS_T:-cfg} | MCDP_P=${FAST_GRASS_MC_DROPOUT_P:-cfg} | L=${FAST_GRASS_L:-cfg} | EMA_ALPHA=${FAST_GRASS_EMA_ALPHA:-cfg} | SELECT=${FAST_GRASS_SELECTION_MODE:-cfg} | M=${FAST_GRASS_M:-cfg} | NO_R=${FAST_GRASS_NO_REGISTRY:-0} | NO_EVAL=${FAST_GRASS_NO_EVAL:-0}"

singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_fast_grass.py \
        ${FAST_GRASS_MODEL_SUFFIX:+--model_suffix $FAST_GRASS_MODEL_SUFFIX} \
        ${FAST_GRASS_NUM_EPOCHS:+--num_epochs $FAST_GRASS_NUM_EPOCHS} \
        ${FAST_GRASS_B_DOC:+--B_doc $FAST_GRASS_B_DOC} \
        ${FAST_GRASS_LAMBDA:+--lambda_val $FAST_GRASS_LAMBDA} \
        ${FAST_GRASS_UNCERTAINTY:+--uncertainty $FAST_GRASS_UNCERTAINTY} \
        ${FAST_GRASS_T:+--T $FAST_GRASS_T} \
        ${FAST_GRASS_MC_DROPOUT_P:+--mc_dropout_p $FAST_GRASS_MC_DROPOUT_P} \
        ${FAST_GRASS_L:+--L $FAST_GRASS_L} \
        ${FAST_GRASS_EMA_ALPHA:+--ema_alpha $FAST_GRASS_EMA_ALPHA} \
        ${FAST_GRASS_SELECTION_MODE:+--selection_mode $FAST_GRASS_SELECTION_MODE} \
        ${FAST_GRASS_M:+--m $FAST_GRASS_M} \
        ${FAST_GRASS_NO_REGISTRY:+--no_registry} \
        ${FAST_GRASS_NO_EVAL:+--no_eval} \
        ${FAST_GRASS_DEBUG:+--debug}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Fast-GRASS training completed successfully"
else
    echo "❌ Fast-GRASS training failed with code $EXIT_CODE"
fi

echo "=========================================="
echo "Job $SLURM_JOB_ID Completed"
echo "=========================================="

exit $EXIT_CODE
