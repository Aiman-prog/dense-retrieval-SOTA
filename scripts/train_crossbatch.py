import sys
import os
from pathlib import Path
from tevatron.retriever.driver.train import main as tevatron_train_main

# Setup pathing
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import get_training_context

def main():
    # 1. Configuration & Paths via Centralized Context
    ctx = get_training_context("crossbatch")
    recipe = ctx['args'] 
    
    # Resolve the mixture directory path
    processed_dir = Path(ctx['processed_dir'])
    mixture_dir = processed_dir / "training_mixture"
    training_data_path = str(mixture_dir / "*.jsonl")

    if not mixture_dir.exists():
        print(f"❌ ERROR: Training mixture directory not found: {mixture_dir}")
        sys.exit(1)

    # 2. Argument Construction
    # RocketQA uses massive batch sizes (up to 4096).
    # A100 PRODUCTION MODE: 64 per device * 2 GPUs * 16 steps = 2048 Virtual Batch
    per_device_batch = recipe['per_device_batch_size']
    acc_steps = recipe['gradient_accumulation_steps']
    chunk_size = recipe['gc_q_chunk_size']  # A100s have 80GB VRAM - larger chunks

    args_list = [
        '--output_dir', str(ctx['output_dir']),
        '--model_name_or_path', ctx['base_model'],
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
        '--train_group_size', str(recipe['train_group_size']),       # 1 positive + in-batch negatives
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),

        # Stability & Debugging
        '--max_grad_norm', str(recipe['max_grad_norm']),        # Prevent gradient explosion → NaN/Inf → SIGFPE
        '--logging_steps', str(recipe['logging_steps']),
        '--overwrite_output_dir', 'True',
        '--save_strategy', 'steps',
        '--save_steps', '100',
        '--save_total_limit', '3',
        '--dataloader_num_workers', str(recipe['dataloader_num_workers']),  # A100 production mode
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
        '--temperature', str(ctx['temperature']),
        '--warmup_ratio', str(recipe['warmup_ratio']),
        '--weight_decay', str(recipe['weight_decay']),
    ]

    # LoRA: freeze base model, only train adapter matrices
    if recipe.get('lora', False):
        args_list += [
            '--lora', 'True',
            '--lora_r', str(recipe.get('lora_r', 16)),
            '--lora_alpha', str(recipe.get('lora_alpha', 32)),
            '--lora_dropout', str(recipe.get('lora_dropout', 0.1)),
            '--lora_target_modules', recipe.get('lora_target_modules', 'query,key,value'),
        ]

    sys.argv = ['train.py'] + args_list
    
    # Patch GradCache loss to use temperature scaling
    # Fix for bug where SimpleContrastiveLoss/DistributedContrastiveLoss ignore temperature
    from models.temperature_scaled_loss import TemperatureScaledContrastiveLoss, DistributedTemperatureScaledContrastiveLoss
    import tevatron.retriever.gc_trainer as gc_module
    
    # Create wrapper classes that use our temperature value
    temp = ctx['temperature']
    class SimpleContrastiveLossPatched(TemperatureScaledContrastiveLoss):
        def __init__(self):
            super().__init__(temperature=temp)
    
    class DistributedContrastiveLossPatched(DistributedTemperatureScaledContrastiveLoss):
        def __init__(self, n_target: int = 0, scale_loss: bool = True):
            super().__init__(temperature=temp, n_target=n_target, scale_loss=scale_loss)
    
    gc_module.SimpleContrastiveLoss = SimpleContrastiveLossPatched
    gc_module.DistributedContrastiveLoss = DistributedContrastiveLossPatched
    
    tevatron_train_main()

if __name__ == "__main__":
    main()