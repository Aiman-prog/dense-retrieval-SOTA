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
    python scripts/preprocessor.py

# 4. Submit training jobs
sbatch scripts/run_inbatch_singularity.sh      # In-batch baseline
sbatch scripts/run_crossbatch_singularity.sh   # Cross-batch (2048)
sbatch scripts/run_ance_singularity.sh         # ANCE iterative

# 5. Evaluate trained models
sbatch scripts/run_evaluate_singularity.sh
```

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
├── setup.sh                     # One-command setup for DelftBlue
├── DELFTBLUE_SETUP.md          # Detailed HPC setup guide
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
- **[DELFTBLUE_SETUP.md](DELFTBLUE_SETUP.md)**: Troubleshooting and detailed setup
- **[src/utils/helpers.py](src/utils/helpers.py)**: Centralized path management

## Environment

- **HPC**: Singularity containers (PyTorch 2.1, CUDA 11.8)
- **GPUs**: NVIDIA A100 (80GB) or V100 (32GB)
- **Tevatron**: Patched version from commit `8f31cd8`
- **Data**: BRIGHT benchmark + ReasonIR training data

## Notes

- All scratch paths resolve via `DATA_BASE_DIR` environment variable
- Tevatron patches remove Qwen multimodal dependencies (see DELFTBLUE_SETUP.md)
- Cross-batch training achieves 2048 virtual batch: 64/device × 2 GPUs × 16 accumulation steps
