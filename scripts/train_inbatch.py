"""
Train In-Batch Negatives model on ReasonIR-HQ.
Refactored to run directly in Python (NO SUBPROCESS) to allow patching of Tevatron bugs.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path so we can import project utils
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_data_base_dir


from tevatron.retriever.modeling import DenseModel
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)
    print("🩹 Applied patch: Added _keys_to_ignore_on_save to DenseModel")

# NOW import the training driver (after applying the patch)
from tevatron.retriever.driver.train import main as tevatron_train_main

def main():
    # 1. Configuration
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    base_dir = get_data_base_dir()
    
    # 2. Paths
    processed_dir = Path(os.environ.get('PROCESSED_DATA_DIR', f'{base_dir}/data/processed'))
    train_file = processed_dir / 'train_reasonir.jsonl'
    output_dir = Path(f'{base_dir}/models/inbatch_reasonir')

    # 3. Validation
    if not train_file.exists():
        print(f"❌ ERROR: Training file not found: {train_file}")
        print(f"   Please run: python src/data/preprocessor.py")
        sys.exit(1)

    # 4. Setup Arguments
    # We grab config from environment variables (set by SLURM script) or defaults
    batch_size = os.environ.get('BATCH_SIZE', '64')
    num_epochs = os.environ.get('NUM_EPOCHS', '3')
    
    print("=" * 80)
    print(f"Training In-Batch Model (Direct Python Mode)")
    print(f"Data: {train_file}")
    print(f"Output: {output_dir}")
    print(f"Batch Size: {batch_size} | Epochs: {num_epochs}")
    print("=" * 80)

    # 5. Construct Argument List
    # We construct the list exactly as if we typed it in the terminal
    training_args = [
        '--output_dir', str(output_dir),
        '--model_name_or_path', config['model']['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', str(train_file),
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(batch_size),
        '--learning_rate', '1e-5',
        '--num_train_epochs', num_epochs,
        '--train_group_size', '1',
        '--query_max_len', '128',
        '--passage_max_len', '512',
        '--dataloader_num_workers', '0',  # Matched to gpu-a100-small
        '--fp16', 'False',
        '--bf16', 'True',
        '--overwrite_output_dir',         # Fixes "directory exists" error
        '--logging_steps', '10',
        '--attn_implementation', 'eager', # Avoids FlashAttention errors on some cards
    ]

    # 6. Inject Arguments into sys.argv
    # Tevatron reads sys.argv internally, so we overwrite it to "trick" it
    sys.argv = ['train_inbatch.py'] + training_args

    # 7. Run Training Directly
    try:
        tevatron_train_main()
        print(f"\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        # Re-raise to ensure SLURM marks job as failed
        raise e

if __name__ == "__main__":
    main()