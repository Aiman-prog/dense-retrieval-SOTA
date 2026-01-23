"""
Train In-Batch Negatives model on ReasonIR-HQ.
Refactored to use centralized context management from config.yaml.
"""

import sys
import logging
from pathlib import Path

# Add src to path so we can import project utils
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_training_context
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.driver.train import main as tevatron_train_main

# 🩹 Tevatron Bug Patch
# DenseModel requires _keys_to_ignore_on_save to exist to prevent 
# errors during checkpoint saving in some Tevatron versions.
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

def main():
    # 1. Get unified context (Hyperparameters + Absolute Paths)
    # This calls helpers.py, which detects if you are on DelftBlue or local
    # and resolves all paths defined in your config.yaml paths section.
    ctx = get_training_context("inbatch") 
    
    # 2. Map YAML/Context to Tevatron Arguments
    # Instead of building strings manually, we pull directly from the ctx dictionary.
    training_args = [
        '--output_dir', str(ctx['output_dir']),
        '--model_name_or_path', ctx['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', str(ctx['train_file']),
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(ctx['args']['batch_size']),
        '--learning_rate', str(ctx['args']['learning_rate']),
        '--num_train_epochs', str(ctx['args']['num_epochs']),
        '--train_group_size', str(ctx['args']['train_group_size']),
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),
        '--bf16', str(ctx['args']['bf16']),
        '--logging_steps', '10',
        '--overwrite_output_dir', 'True',
        '--attn_implementation', 'eager',  # Ensures compatibility across different GPU architectures
    ]

    # 3. Inject Arguments into sys.argv
    # Tevatron's driver reads from sys.argv; we "trick" it by overwriting the list.
    sys.argv = ['train.py'] + training_args

    # 4. Run Training Directly
    try:
        print(f"🚀 Starting In-Batch Training for model: {ctx['args']['model_name']}")
        tevatron_train_main()
        print(f"✅ Training completed. Model saved to: {ctx['output_dir']}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Re-raise ensures SLURM correctly identifies the job failure
        raise e

if __name__ == "__main__":
    main()