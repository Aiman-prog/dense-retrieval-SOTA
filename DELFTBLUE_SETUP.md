# DelftBlue HPC Setup Guide

**Critical modifications required to run dense retrieval training on DelftBlue with Tevatron + Singularity**

---

## 1. Singularity Container Setup

### Install Required Packages Inside Container

When using the PyTorch Singularity container, you need to install Tevatron and dependencies:

```bash
# Launch container in writable mode (if possible) or use --fakeroot
singularity exec --nv /scratch/${USER}/containers/pytorch_2.1.sif bash

# Install Tevatron from GitHub (April 2025 version)
pip install --user git+https://github.com/texttron/tevatron.git@8f31cd8

# Install additional dependencies
pip install --user peft accelerate>=0.25.0
```

**Note:** The `--user` flag installs to `~/.local/lib/python3.10/site-packages/`

---

## 2. Tevatron Code Modifications (CRITICAL!)

Tevatron's recent version added multimodal support requiring Qwen models not available in transformers 4.36.0. You MUST patch these files:

### 2.1 Remove Qwen Import

**File:** `~/.local/lib/python3.10/site-packages/tevatron/retriever/modeling/dense.py`

**Line 3:** Comment out:
```python
# from transformers import Qwen2_5OmniThinkerForConditionalGeneration
```

**Line 43:** Comment out:
```python
# TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration
```

### 2.2 Remove MultiModalDenseModel Export

**File:** `~/.local/lib/python3.10/site-packages/tevatron/retriever/modeling/__init__.py`

Change:
```python
from .dense import DenseModel, MultiModalDenseModel
```

To:
```python
from .dense import DenseModel
```

### 2.3 Add Missing torch Import

**File:** `~/.local/lib/python3.10/site-packages/tevatron/retriever/driver/train.py`

**Line 1:** Add:
```python
import torch
```

This is required because line 77 uses `torch.float32` without importing torch.

### 2.4 Clear Python Cache

After modifications, remove cached bytecode:
```bash
rm ~/.local/lib/python3.10/site-packages/tevatron/retriever/modeling/__pycache__/dense.cpython-*.pyc
```

---

## 3. Training Script Requirements

### 3.1 Essential Arguments for Cross-Batch Training

```python
args_list = [
    # GradCache for cross-batch negatives
    '--grad_cache', 'True',
    '--gc_q_chunk_size', str(chunk_size),
    '--gc_p_chunk_size', str(chunk_size),
    '--per_device_train_batch_size', str(per_device_batch),
    '--gradient_accumulation_steps', str(acc_steps),

    # CRITICAL: Precision settings
    '--fp16', 'False',
    '--bf16', 'True',
    '--dtype', 'bfloat16',  # ← MUST HAVE! Prevents precision mismatch

    # Optimizer
    '--attn_implementation', 'eager',
    '--optim', 'adamw_torch_fused',

    # Standard training args
    '--learning_rate', str(learning_rate),
    '--num_train_epochs', str(num_epochs),
    '--train_group_size', '1',  # 1 positive + cross-batch negatives
    '--query_max_len', str(max_q),
    '--passage_max_len', str(max_p),
    '--max_grad_norm', '1.0',
    '--logging_steps', '10',
    '--overwrite_output_dir', 'True',
    '--dataloader_num_workers', '4',
]
```

### 3.2 Arguments to AVOID

**Do NOT use these - they cause crashes or break training:**

```python
# ❌ BREAKS TRAINING (causes stuck loss ~76)
'--normalize', 'True',
'--temperature', '0.05',

# ❌ CRASHES with BERT models
'--gradient_checkpointing', 'True',  # AttributeError: 'BertModel' object has no attribute 'model'
```

**Why they fail:**
- `--normalize` and `--temperature`: Not recognized by Tevatron CLI, cause loss computation issues
- `--gradient_checkpointing`: Tevatron's implementation incompatible with BERT architecture

---

## 4. SLURM Configuration

### gpu-a100 Configuration (2 GPUs, 2048 virtual batch)

```bash
#SBATCH --job-name=rocketqa-a100-2048
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # 2 GPUs
#SBATCH --cpus-per-task=2           # 2 CPUs per GPU
#SBATCH --gpus-per-task=1           # Total: 2 GPUs
#SBATCH --mem-per-gpu=16G           # 16GB RAM per GPU
```

### Environment Variables

```bash
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export PYTHONPATH="/home/${USER}/dense-retrieval-SOTA/src:${PYTHONPATH}"
export APPTAINER_CACHEDIR=/scratch/${USER}/.apptainer

# Offline mode (pre-downloaded models)
export HF_HOME="${DATA_BASE_DIR}/data/bright"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CUDA configuration for A100
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
```

---

## 5. Training Expectations

### Loss Behavior

**Cross-batch training (2048 negatives) vs In-batch (64 negatives):**

| Metric | In-batch | Cross-batch | Explanation |
|--------|----------|-------------|-------------|
| Initial loss | ~5 | ~80 | More negatives = higher loss scale |
| Final loss | ~1.5 | ~5-10 | Still higher, but proportional |
| Loss scale | Lower | 10-15x higher | **This is NORMAL!** |
| Convergence | Faster | Slower initially | Takes ~0.3 epochs to start dropping |

**Key insight:** Training loss is NOT comparable between methods. Higher crossbatch loss does NOT mean worse model!

### Expected Timeline (A100, 2 epochs, ~22K examples)

- Total steps: ~216 (108 steps/epoch)
- Time per step: ~11 seconds
- Total time: ~40 minutes
- Loss trajectory: 82 → 56 → 39 → 20 → ~5-10 (final)

---

## 6. Troubleshooting

### Issue: Loss stuck at ~76, barely decreasing

**Cause:** `--normalize` and `--temperature` arguments present
**Fix:** Remove these arguments

### Issue: AttributeError: 'BertModel' object has no attribute 'model'

**Cause:** `--gradient_checkpointing True` enabled
**Fix:** Remove gradient checkpointing

### Issue: ImportError for Qwen2_5OmniThinkerForConditionalGeneration

**Cause:** Tevatron code not patched
**Fix:** Apply all modifications in Section 2

### Issue: NameError: name 'torch' is not defined

**Cause:** Missing torch import in train.py
**Fix:** Add `import torch` to train.py (Section 2.3)

---

## 7. Verification Checklist

Before running training, verify:

- [ ] Tevatron installed from GitHub (commit 8f31cd8 or later)
- [ ] All 4 Tevatron patches applied (Qwen, MultiModal, torch import, cache cleared)
- [ ] Training script has `--dtype bfloat16`
- [ ] Training script does NOT have `--normalize` or `--temperature`
- [ ] Training script does NOT have `--gradient_checkpointing`
- [ ] SLURM script uses `--mem-per-gpu` (not `--mem`)
- [ ] Environment variables set correctly
- [ ] Singularity container has correct binds

---

## 8. Summary of Changes from Standard Tevatron

**What's different:**
1. Removed Qwen/multimodal code (not needed for dense retrieval)
2. Added missing `import torch`
3. Don't use `--normalize` or `--temperature` CLI args (DenseModel handles internally)
4. Don't use `--gradient_checkpointing` (incompatible with BERT)
5. MUST use `--dtype bfloat16` with `--bf16 True`

**Why these changes:**
- Tevatron's recent multimodal additions are incompatible with older transformers
- Some CLI args break GradCache training
- Precision mismatch causes extreme loss values
- BERT models have different architecture than models gradient checkpointing was designed for

---

**Last Updated:** 2026-02-07
**Tested On:** DelftBlue HPC, gpu-a100 partition, PyTorch 2.1 Singularity container