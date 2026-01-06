"""Train RocketQA on BRIGHT biology domain (Cluster Optimized)."""

import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from data.bright_loader import BRIGHTLoader
from data.preprocessor import BRIGHTPreprocessor
from utils.helpers import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs (Cluster default: 4)")
    args = parser.parse_args()

    # Load config
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    domain = 'biology'

    # ---------------------------------------------------------
    # Steps 1-3: Prepare Data (Runs once on the head node)
    # ---------------------------------------------------------
    print(f"Preparing data for {domain}...")
    
    loader = BRIGHTLoader(config_path=str(project_root / 'config' / 'config.yaml'))
    loader.load_dataset()
    domain_data = loader.get_data_split(domain)
    
    preprocessor = BRIGHTPreprocessor(output_dir=str(project_root / 'data/processed'))
    train_file_path = preprocessor.prepare_train_data(
        domain_data['queries'],
        domain_data['corpus'],
        domain_data['qrels'],
        filename=f'{domain}_rocketqa_train.jsonl',
        hard_negatives=None # In-Batch Mode
    )
    
    # Tevatron needs a directory, so we isolate the file
    train_dir = project_root / 'data/processed' / f'{domain}_rocketqa_train'
    if train_dir.exists(): shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_file_path, train_dir / f'{domain}_rocketqa_train.jsonl')
    
    # ---------------------------------------------------------
    # Step 4: Train (Cluster Configuration)
    # ---------------------------------------------------------
    output_dir = project_root / f'data/models/rocketqa_{domain}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE: Comment out the top block for Laptop, use bottom block instead.
    
    # --- OPTION A: CLUSTER MODE (DEFAULT) ---
    # cmd = [
    #     'torchrun',
    #     '--nproc_per_node', str(args.num_gpus),
    #     '-m', 'tevatron.driver.train',
    #     '--output_dir', str(output_dir),
    #     '--model_name_or_path', config['model']['base_model'],
    #     '--train_dir', str(train_dir),
    #     '--do_train',
    #     '--per_device_train_batch_size', '128',
    #     '--learning_rate', '1e-5',
    #     '--num_train_epochs', '3',
    #     '--grad_cache',
    #     '--gradient_accumulation_steps', '1',
    #     '--dataloader_num_workers', '4',
    #     '--fp16',
        
    #     # [LAPTOP USER]: COMMENT OUT THESE 2 LINES TO RUN LOCALLY
    #     '--negatives_x_device',           # <--- CRASHES ON SINGLE GPU
    #     '--ddp_find_unused_parameters', 'False'
    # ]

    # --- OPTION B: LAPTOP MODE (Single GPU) ---
    # Use this for single GPU training (no distributed training)
    cmd = [
        sys.executable, '-m', 'tevatron.driver.train',
        '--output_dir', str(output_dir),
        '--model_name_or_path', config['model']['base_model'],
        '--train_dir', str(train_dir),
        '--do_train',
        '--per_device_train_batch_size', '32', # Lower batch size for laptop
        '--learning_rate', '1e-5',
        '--num_train_epochs', '3',
        '--train_n_passages', '1',  # Critical: Set to 1 for in-batch negatives (RocketQA)
        # '--grad_cache',  # Critical for RocketQA - enables large effective batch sizes
        # Note: fp16 with grad_cache may have issues, but grad_cache is more important
        # '--fp16',  # Commented out - use grad_cache without fp16 to avoid scaler issues
        # Explicitly disable distributed training for single GPU
    ]

    print(f"Running command in single-GPU mode (laptop mode)...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()