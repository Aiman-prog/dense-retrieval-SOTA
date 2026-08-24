"""
Train In-Batch Negatives model on ReasonIR-HQ.
Refactored to use centralized context management from config.yaml.
MODIFIED: Now uses the training_mixture directory for data.
"""

import sys
import os
import logging
import math
from pathlib import Path

# Add src to path so we can import project utils
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_training_context, patch_tevatron_loss, load_config, set_seed, \
                          log_startup_config
from data.preprocessor import MIXTURE_FILES, require_mixture_files
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.driver.train import main as tevatron_train_main

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

# 🩹 Tevatron bug: EncoderModel.gradient_checkpointing_enable() calls
# `self.encoder.model.gradient_checkpointing_enable()`, but for a full fine-tune
# `self.encoder` IS the HF model (XLMRobertaModel — no `.model` attr), so HF Trainer's
# gradient_checkpointing=True crashes with "'XLMRobertaModel' object has no attribute 'model'".
# Forward straight to the encoder — mirrors exactly what run_fast_grass does (proven to fit
# batch 64 @ q1024/p512). Accept **kwargs so Trainer's gradient_checkpointing_kwargs= is
# absorbed; we intentionally don't forward it (matches the no-arg fast_grass call).
def _tevatron_gc_enable(self, **kwargs):
    self.encoder.gradient_checkpointing_enable()
DenseModel.gradient_checkpointing_enable = _tevatron_gc_enable

def main():
    # 1. Get unified context (Hyperparameters + Absolute Paths)
    config = load_config()
    set_seed(config.get('seed', 42))
    ctx = get_training_context("inbatch")
    log_startup_config("inbatch", ctx)
    
    # --- PATH MODIFICATION: Resolve the mixture directory ---
    # We look for the folder named 'training_mixture' inside the data directory
    processed_dir = Path(ctx['processed_dir'])
    mixture_dir = processed_dir / "training_mixture"
    
    # Tevatron accepts one glob; strict validation below guarantees it contains only
    # the three declared mixed-training components.
    training_data_path = str(mixture_dir / "*.jsonl")

    mixture_files = require_mixture_files(mixture_dir, MIXTURE_FILES)

    # --- Count training examples and calculate checkpoint intervals ---
    num_examples = 0
    for f in mixture_files:
        with open(f) as fh:
            num_examples += sum(1 for _ in fh)

    batch_size = ctx['args']['batch_size']
    num_epochs = ctx['args']['num_epochs']
    total_steps = math.ceil(num_examples / batch_size) * num_epochs
    save_steps = max(1, total_steps // 5)  # Save at 20% intervals

    # --- CONFIG PARAMETER PRINTS ---
    print("\n" + "="*40)
    print("🛠️  VERIFYING CONFIGURATION PARAMETERS")
    print(f"▶️  Train Group Size:   {ctx['args'].get('train_group_size')}")
    print(f"▶️  Batch Size:         {ctx['args'].get('batch_size')}")
    print(f"▶️  Learning Rate:      {ctx['args'].get('learning_rate')}")
    print(f"▶️  Num Epochs:         {ctx['args'].get('num_epochs')}")
    print(f"📂 Training Data:      {training_data_path}")
    print(f"📊 Training examples:  {num_examples}")
    print(f"📊 Total steps:        {total_steps}, saving every {save_steps} steps")
    print("="*40 + "\n")
    
    # 2. Map YAML/Context to Tevatron Arguments
    training_args = [
        '--output_dir', str(ctx['output_dir']),
        '--model_name_or_path', ctx['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', training_data_path,    # Updated to use the glob path
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(ctx['args']['batch_size']),
        '--learning_rate', str(ctx['args']['learning_rate']),
        '--num_train_epochs', str(ctx['args']['num_epochs']),
        '--train_group_size', str(ctx['args']['train_group_size']),
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),
        '--bf16', str(ctx['args']['bf16']),
        '--fp16', 'False',
        '--dtype', 'bfloat16' if ctx['args']['bf16'] else 'float32',
        '--logging_steps', '100',
        '--overwrite_output_dir', 'True',
        '--save_strategy', 'steps',
        '--save_steps', str(save_steps),
        '--save_total_limit', '6',
        '--attn_implementation', 'eager',     # XLM-RoBERTa has no sdpa in this transformers ver
        '--gradient_checkpointing', 'True',   # THE memory fix: frees the eager-attention activations
                                              # (~50GB at q1024). This is what fast_grass actually used
                                              # (its else-branch, since bitsandbytes is absent here).
        '--dataloader_num_workers', str(ctx['args']['dataloader_num_workers']),
        '--optim', 'adamw_torch_fused',       # bitsandbytes not available in this container
        '--warmup_ratio', str(ctx['args']['warmup_ratio']),
        '--weight_decay', str(ctx['args']['weight_decay']),
        '--max_grad_norm', str(ctx['args']['max_grad_norm']),
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
        '--temperature', str(ctx['temperature']),
        '--seed', str(config.get('seed', 42)),
    ]

    # 3. Inject Arguments into sys.argv
    sys.argv = ['train.py'] + training_args

    # 4. Run Training Directly
    try:
        print(f"🚀 Starting In-Batch Training for model: {ctx['args']['model_name']}")
        print(f"📂 Loading data from: {training_data_path}")
        
        patch_tevatron_loss(ctx['temperature'])
        tevatron_train_main()
        print(f"✅ Training completed. Model saved to: {ctx['output_dir']}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise e

if __name__ == "__main__":
    main()
