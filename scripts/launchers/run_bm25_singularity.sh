#!/usr/bin/env bash

#SBATCH --job-name=eval-bm25
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3900M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/eval_bm25_%j.out
#SBATCH --error=logs/eval_bm25_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# --- Environment ---
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Java 11 required by pyserini/Lucene (see README for setup)
export JAVA_HOME="/home/${USER}/.jdk21"   # P9: the /scratch copy was written during the BeeGFS fault and is broken
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

mkdir -p logs

echo "Starting BM25 evaluation (pyserini, k1=0.9, b=0.4)"

# No --nv: BM25 is CPU-only
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python -u scripts/run_bm25_evals.py

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "BM25 evaluation completed successfully"
else
    echo "BM25 evaluation failed with exit code $EXIT_CODE"
fi
exit $EXIT_CODE
