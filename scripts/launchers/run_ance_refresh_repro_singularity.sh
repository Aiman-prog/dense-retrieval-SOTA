#!/usr/bin/env bash

# Reproduce job 204931's refresh crash (P-ANCE-05) with the real inferencer, no new code.
#
# The crash was at IMPORT time, 31s into the first refresh, so it needs the inferencer to
# spawn ONE child -- not a finished refresh, not a trainer. run_ance_data_gen.py is already
# a standalone entry point and models/ance_paper_roberta/checkpoint-10000 is the checkpoint
# the dead child was given. Loading the 8.8M-entry corpus lookup first is what makes the
# parent big enough to trigger it; synthetic matrices never did.
#
#   sbatch [--export=ALL,ANCE_MKL_FIX=1] scripts/launchers/run_ance_refresh_repro_singularity.sh
# Reproduced: FAILED ~2.5 min (220386). Fixed: TIMEOUT at the wall, still encoding (220456).

#SBATCH --job-name=ance-refresh-repro
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # partition cap
#SBATCH --gpus-per-task=1            # partition cap; the inferencer only ever uses one
#SBATCH --mem-per-cpu=8000M          # 2 x 8000M = 16 GB, the partition ceiling; the
                                     # corpus lookup peaked at 5.5 GB (job 204930)
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/ance_refresh_repro_%j.out
#SBATCH --error=logs/ance_refresh_repro_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH:-}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=16            # exactly what the ance_paper launcher sets

# The real inferencer is masked to GPU 1 of two; this partition allows one, so device 0
# is the single difference this reproduction cannot carry over. It reproduced anyway.
export CUDA_VISIBLE_DEVICES=0

if [ -n "${ANCE_MKL_FIX:-}" ]; then
    # Before python starts and into the container: the child inherits the parent's env.
    export MKL_THREADING_LAYER=GNU
    export SINGULARITYENV_MKL_THREADING_LAYER=GNU
    echo "🩹 remedy applied: MKL_THREADING_LAYER=GNU"
else
    echo "🔬 no remedy applied — measuring whether the crash reproduces"
fi

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
P="${DATA_BASE_DIR}/data/processed"
MODEL_DIR="${DATA_BASE_DIR}/models/ance_paper_roberta"
WORK_ROOT="${DATA_BASE_DIR}/temp_ance_paper_workdir/repro-${SLURM_JOB_ID}"

mkdir -p logs

if [ ! -d "${MODEL_DIR}/checkpoint-10000" ]; then
    echo "❌ ${MODEL_DIR}/checkpoint-10000 is gone — that checkpoint IS the reproduction."
    echo "   Without it there is nothing for the inferencer to pick up."
    exit 2
fi

echo "📍 model dir : ${MODEL_DIR} (polls for the newest checkpoint)"
echo "📍 work root : ${WORK_ROOT}"
ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null

# No --overwrite, no training, nothing published into a real run's work root: this writes
# only under its own repro-<jobid> directory.
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_ance_data_gen.py \
        --output_model_dir "${MODEL_DIR}" \
        --work_root        "${WORK_ROOT}" \
        --run_id           "repro-${SLURM_JOB_ID}" \
        --corpus_file      "${P}/msmarco_corpus.jsonl" \
        --query_file       "${P}/msmarco_train_queries.jsonl" \
        --qrels_file       "${P}/msmarco_train_qrels.txt" \
        --recipe           ance_paper

EXIT_CODE=$?
echo "REPRO_EXIT=${EXIT_CODE}"
echo "   nonzero within a few minutes => reproduced (check .err for the child's output)"
echo "   killed at the 30m wall       => NOT reproduced; the encode simply ran on"
exit $EXIT_CODE
