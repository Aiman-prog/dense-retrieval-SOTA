import sys
import os
from pathlib import Path
from tevatron.retriever.driver.train import main as tevatron_train_main

# Setup pathing
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import get_training_context, load_config

def main():
    # 1. Configuration & Paths via Centralized Context
    ctx = get_training_context("crossbatch")
    recipe = ctx['args'] 
    config_yaml = load_config()
    
    # Resolve the mixture directory path
    processed_dir = Path(ctx['train_file']).parent.resolve()
    mixture_dir = processed_dir / "training_mixture"
    training_data_path = str(mixture_dir / "*.jsonl")

    if not mixture_dir.exists():
        print(f"❌ ERROR: Training mixture directory not found: {mixture_dir}")
        sys.exit(1)

    # 2. Argument Construction
    # RocketQA uses massive batch sizes (up to 4096). [cite: 237]
    # A100 PRODUCTION MODE: 64 per device * 2 GPUs * 16 steps = 2048 Virtual Batch
    per_device_batch = 64
    acc_steps = 16
    chunk_size = 16  # A100s have 80GB VRAM - larger chunks

    args_list = [
        '--output_dir', str(ctx['output_dir']),
        '--model_name_or_path', config_yaml['model']['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', training_data_path,
        '--dataset_split', 'train',
        '--do_train',

        # RocketQA Strategy 1: Cross-batch negatives via GradCache [cite: 10, 59]
        '--grad_cache', 'True',
        '--gc_q_chunk_size', str(chunk_size),
        '--gc_p_chunk_size', str(chunk_size),
        '--per_device_train_batch_size', str(per_device_batch),
        '--gradient_accumulation_steps', str(acc_steps),

        # Precision: bf16 for A100 (better than fp16)
        '--fp16', 'False',
        '--bf16', 'True',
        '--dtype', 'bfloat16',           # Force model weights to BF16 on load
        '--attn_implementation', 'eager',
        '--optim', 'adamw_torch_fused',      # Uses more efficient GPU kernels

        # Core Hyperparameters [cite: 244, 245]
        '--learning_rate', str(recipe['learning_rate']),
        '--num_train_epochs', str(recipe['num_epochs']),
        '--train_group_size', '1',       # 1 positive + in-batch negatives [cite: 39, 40]
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),

        # Stability & Debugging
        '--max_grad_norm', '1.0',        # Prevent gradient explosion → NaN/Inf → SIGFPE
        '--logging_steps', '10',
        '--overwrite_output_dir', 'True',
        '--dataloader_num_workers', '4',  # A100 production mode
    ]

    sys.argv = ['train.py'] + args_list
    tevatron_train_main()

if __name__ == "__main__":
    main()