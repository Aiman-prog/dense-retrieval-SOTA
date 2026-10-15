#!/usr/bin/env bash

#SBATCH --job-name=gc-matrix
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=logs/gc_matrix_%j.out
#SBATCH --error=logs/gc_matrix_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-SOTA

# Is a FASTER async Fast-GRASS run available at an IDENTICAL score?
#
# Only mathematically-neutral changes qualify: same gradients, same data, same order.
# That rules out step count, batch size, sequence lengths, B_doc and mine_chunk_factor,
# and leaves exactly two candidates, measured here as a 2x2:
#
#   gradient checkpointing {on, off}  -- identical gradients, pure compute<->memory trade.
#                                        run_async_fast_grass_train.py:285-287 turns it on
#                                        unconditionally. CLAUDE.md records a 60.09 GiB
#                                        peak WITH it at p1024, so an OOM at q1024 is a
#                                        real possible answer, not a failure of the job.
#   torch.compile {on, off}           -- the run logs "torch.compile enabled", but
#                                        CLAUDE.md warns it "falls back to eager
#                                        silently". If eager and compiled time the same,
#                                        compile is inert and there is nothing to win.
#
# REPORT ONLY. Trains nothing that is kept, writes no checkpoint anyone evaluates, and
# cannot disturb the finished s6_* results. A cell that OOMs is recorded and the matrix
# continues.

export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:/home/${USER}/dense-retrieval-SOTA/scripts:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"
STEPS="${GC_STEPS:-100}"
WARMUP="${GC_WARMUP:-10}"

mkdir -p logs

echo "🔬 gradient-checkpointing x torch.compile matrix"
echo "   steps=${STEPS} warmup=${WARMUP} (production shape: bs64 m1 group2 q1024/p512)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for GC in on off; do
  for CP in on off; do
    GC_FLAG=""; [ "$GC" = "off" ] && GC_FLAG="--no_gradient_checkpointing"
    CP_FLAG=""; [ "$CP" = "off" ] && CP_FLAG="--no_compile"
    echo ""
    echo "=================================================================="
    echo "  CELL: gradient_checkpointing=${GC}  compile=${CP}"
    echo "=================================================================="
    singularity exec --nv \
        --bind /scratch/${USER}:/scratch/${USER} \
        --bind /home/${USER}:/home/${USER} \
        ${CONTAINER} \
        python -u scripts/dev/fast_grass_train_timing.py \
            --steps "${STEPS}" \
            --warmup_steps "${WARMUP}" \
            --no_checkpoint_probe \
            ${GC_FLAG} ${CP_FLAG}
    CELL_EXIT=$?
    if [ $CELL_EXIT -ne 0 ]; then
        echo "  ⚠️  cell gc=${GC} compile=${CP} exited ${CELL_EXIT} (OOM is a valid result)"
    fi
  done
done

echo ""
echo "=========================================="
echo "Job $SLURM_JOB_ID matrix complete"
echo "JSONs: analysis/async_fast_grass_timing/train_timing_*.json"
echo "Compare seconds_per_train_step and peak_mem_allocated_bytes across the four."
echo "=========================================="
