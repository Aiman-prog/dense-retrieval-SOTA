# Dense Retrieval SOTA Benchmark

Benchmarking state-of-the-art dense retrieval methods on the [BRIGHT benchmark](https://huggingface.co/datasets/xlangai/BRIGHT) using **Tevatron** for training and **GradCache** for memory-efficient cross-batch negatives.

## Training Methods

- **In-Batch Negatives**: Baseline using negatives from the same batch
- **Cross-Batch Negatives**: RocketQA-style training with 2048 virtual batch size via GradCache
- **ANCE**: Iterative hard negative mining with approximate nearest neighbors

## Quick Start (DelftBlue HPC)

```bash
# 1. Setup environment (downloads container, installs Tevatron, applies patches)
./setup.sh

# 2. Download models and data for offline mode
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    /scratch/${USER}/containers/pytorch_2.1.sif \
    python scripts/prepare_models.py

# 3. Preprocess BRIGHT benchmark data
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    /scratch/${USER}/containers/pytorch_2.1.sif \
    python src/data/preprocessor.py

# 4. Submit training jobs
sbatch scripts/run_inbatch_singularity.sh      # In-batch baseline
sbatch scripts/run_crossbatch_singularity.sh   # Cross-batch (2048)
sbatch scripts/run_ance_singularity.sh         # ANCE iterative

# 5. Evaluate trained models
sbatch scripts/run_evaluate_singularity.sh

# 6. Evaluate all checkpoints (0%-100% training progress)
sbatch scripts/run_eval_checkpoints_singularity.sh

# 7. BM25 baseline — CPU-only, no GPU (see BM25 Setup section below first)
sbatch scripts/run_bm25_singularity.sh

# 7. Plot Recall@1000 curve (no GPU needed)
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    /scratch/${USER}/containers/pytorch_2.1.sif \
    python scripts/plot_recall_curve.py \
        --results_file /scratch/${USER}/dense-retrieval-SOTA/results/bright_benchmark/inbatch_reasonir_neg/checkpoint_results.json

# Plot all 3 metrics (MRR, NDCG@10, Recall@1000)
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    /scratch/${USER}/containers/pytorch_2.1.sif \
    python scripts/plot_recall_curve.py \
        --results_file /scratch/${USER}/dense-retrieval-SOTA/results/bright_benchmark/inbatch_reasonir_neg/checkpoint_results.json \
        --metrics recall_1000 ndcg_cut_10 recip_rank
```

## BM25 Setup (Pyserini)

Pyserini wraps Lucene and requires **Java 11+**, which is not in the base container.
Run these once on the login node before submitting the BM25 job:

```bash
# 1. Download JDK 21 (Temurin) to scratch — Lucene 9+ requires Java 17+
JDK_DIR="/scratch/${USER}/.jdk21"
mkdir -p "$JDK_DIR"
wget "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.3%2B9/OpenJDK21U-jdk_x64_linux_hotspot_21.0.3_9.tar.gz" \
     -O /tmp/jdk21.tar.gz
tar xzf /tmp/jdk21.tar.gz -C "$JDK_DIR" --strip-components=1
"$JDK_DIR/bin/java" -version   # verify: openjdk 21.x.x

# 2. Install pyserini into ~/.local (persists across jobs)
singularity exec \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    /scratch/${USER}/containers/pytorch_2.1.sif \
    pip install --user --quiet pyserini
```

Then submit: `sbatch scripts/run_bm25_singularity.sh`

Results land in `/scratch/${USER}/dense-retrieval-SOTA/results/bright_benchmark/{domain}_results_bm25.json`.

---

## Project Structure

```
dense-retrieval-SOTA/
├── config/
│   └── config.yaml              # Central configuration (paths, hyperparameters)
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── evaluation/              # TREC evaluation wrapper
│   └── utils/
│       └── helpers.py           # Path management and config loading
├── scripts/
│   ├── train_inbatch.py         # In-batch negatives training
│   ├── train_crossbatch.py      # Cross-batch negatives (GradCache)
│   ├── train_ance.py            # ANCE iterative training
│   ├── run_*_singularity.sh     # SLURM job scripts
│   ├── prepare_models.py        # Download models for offline mode
│   └── preprocessor.py          # Preprocess BRIGHT data
├── docs/                        # All topic documentation
│   ├── DELFTBLUE_SETUP.md       # Detailed HPC setup guide
│   ├── GPU_CHECKLIST.md         # Per-experiment submit + success signals
│   ├── lambda_pilot*.md         # Stage 7 pilot and its verdict
│   ├── *fast_grass*.md          # Architecture and implementation details
│   └── assets/                  # Figures and reference PDFs
├── tests/                       # CPU test suites and smokes (no GPU)
├── setup.sh                     # One-command setup for DelftBlue
└── requirements.txt             # Local development (use setup.sh for HPC)
```

## Key Configuration

Edit `config/config.yaml` to adjust:
- **Model**: Base model, max lengths
- **Training**: Batch sizes, learning rates, epochs
- **Paths**: Data directories, model output locations
- **Evaluation**: BRIGHT domains, metrics

## Important Files

- **[setup.sh](setup.sh)**: Automated setup (container + Tevatron patches)
- **[config/config.yaml](config/config.yaml)**: All hyperparameters and paths
- **[docs/DELFTBLUE_SETUP.md](docs/DELFTBLUE_SETUP.md)**: Troubleshooting and detailed setup
- **[docs/GPU_CHECKLIST.md](docs/GPU_CHECKLIST.md)**: What to submit, and how to tell it worked
- **[src/utils/helpers.py](src/utils/helpers.py)**: Centralized path management

## Environment

- **HPC**: `pytorch_2.1.sif` Singularity container, but see below — the container name is
  historical and does **not** describe the runtime
- **GPUs**: NVIDIA A100 (80GB) or V100 (32GB)
- **Tevatron**: Patched version from commit `8f31cd8`
- **Data**: BRIGHT benchmark + ReasonIR training data
- **Actual runtime**: torch **2.10.0+cu128**, transformers 4.40.2 — resolved from
  `~/.local`, which shadows the container. The container supplies CUDA and a torch 2.1
  that nothing uses; it has no `transformers` at all. Full resolved list:
  [docs/DELFTBLUE_ENVIRONMENT.md](docs/DELFTBLUE_ENVIRONMENT.md). See defect **P7** in
  `CONSOLIDATION_STATUS.md`.

## Clean Reset (DelftBlue)

If setup breaks or you want to start fresh, run these commands to wipe everything and re-run `setup.sh`:

```bash
# Remove all user-installed Python packages (tevatron, transformers, etc.)
rm -rf ~/.local/lib/python3.10/site-packages/*

# Remove the Singularity container (will be re-downloaded by setup.sh)
rm -f /scratch/${USER}/containers/pytorch_2.1.sif

# Remove downloaded models and processed data
rm -rf /scratch/${USER}/dense-retrieval-SOTA/data

# Then re-run setup
./setup.sh
```

To only reset Python packages (most common fix):
```bash
rm -rf ~/.local/lib/python3.10/site-packages/*
./setup.sh
```

## Notes

- All scratch paths resolve via `DATA_BASE_DIR` environment variable
- Tevatron patches remove Qwen multimodal dependencies (see docs/DELFTBLUE_SETUP.md)
- Cross-batch training achieves 2048 virtual batch: 64/device × 2 GPUs × 16 accumulation steps
