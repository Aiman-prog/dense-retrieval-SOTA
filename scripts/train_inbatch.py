"""Train In-Batch Negatives model on ReasonIR-HQ dataset (Single GPU, Simple)."""

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
from utils.helpers import load_config, get_data_base_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, help="Path to training file (default: auto-generate)")
    args = parser.parse_args()

    # Load config
    config = load_config(str(project_root / 'config' / 'config.yaml'))

    # ---------------------------------------------------------
    # Steps 1-2: Prepare ReasonIR-HQ Training Data
    # ---------------------------------------------------------
    print("=" * 80)
    print("Preparing ReasonIR-HQ training data...")
    print("=" * 80)
    
    # Step 1: Load BRIGHT dataset and create ID mapping (needed for ReasonIR-HQ)
    print("Step 1: Loading BRIGHT dataset for document ID mapping...")
    loader = BRIGHTLoader(config_path=str(project_root / 'config' / 'config.yaml'))
    loader.load_dataset()
    id2doc = loader.get_all_documents_id_map()
    print(f"✅ Created ID-to-text mapping for {len(id2doc)} documents")
    
    # Step 2: Prepare ReasonIR-HQ training data
    print("\nStep 2: Preparing ReasonIR-HQ training data...")
    preprocessor = BRIGHTPreprocessor()
    reasonir_config = config['dataset']['reasonir']
    
    if args.train_file and Path(args.train_file).exists():
        # Use existing training file if provided
        print(f"Using existing training file: {args.train_file}")
        train_file_path = Path(args.train_file)
    else:
        # Generate training data
        train_file_path = Path(preprocessor.prepare_reasonir_hq_train_data(
            id2doc=id2doc,
            dataset_name=reasonir_config['name'],
            subset=reasonir_config['subset'],
            cache_dir=reasonir_config.get('cache_dir'),
            filename='train_reasonir.jsonl'
        ))
        print(f"✅ Generated training file: {train_file_path}")
    
    # GitHub Tevatron uses dataset_path pointing directly to the file
    # No need to create a directory - just use the file path directly
    print(f"✅ Training file ready: {train_file_path}")
    
    # ---------------------------------------------------------
    # Step 3: Train (Single GPU, In-Batch Negatives)
    # ---------------------------------------------------------
    # Always use DelftBlue structure: /scratch/${USER}/dense-retrieval-SOTA/models/inbatch_reasonir
    base_dir = get_data_base_dir()
    output_dir = Path(f'{base_dir}/models/inbatch_reasonir')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Step 3: Training In-Batch Negatives model on ReasonIR-HQ...")
    print("=" * 80)
    # Get batch size from environment variable (default: 32 for gpu-v100 with FP32)
    # Using gradient accumulation to achieve effective batch size 64
    import os
    batch_size = int(os.environ.get('BATCH_SIZE', '32'))
    gradient_accumulation_steps = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', '2'))
    
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_file_path}")
    print(f"Batch size: {batch_size} (single GPU, in-batch negatives)")
    
    # Verify CUDA is available before training (GPU job requirement)
    import torch
    if not torch.cuda.is_available():
        print("⚠️  ERROR: CUDA not detected by PyTorch!")
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA version (PyTorch): {torch.version.cuda}")
        print("   This is a GPU job - CUDA should be available")
        print("   Check: module list, nvidia-smi, PyTorch CUDA installation")
        sys.exit(1)
    
    print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
    # Single GPU mode - simple in-batch negatives (no grad_cache, no gradient accumulation)
    # GitHub Tevatron uses dataset_name and dataset_path instead of train_dir
    cmd = [
        sys.executable, '-m', 'tevatron.retriever.driver.train',
        '--output_dir', str(output_dir),
        '--model_name_or_path', config['model']['base_model'],
        '--attn_implementation', 'eager',  # BERT doesn't support Flash Attention 2
        '--dataset_name', 'json',  # GitHub Tevatron: specify dataset format
        '--dataset_path', str(train_file_path),  # GitHub Tevatron: direct file path
        '--dataset_split', 'train',  # GitHub Tevatron: specify split
        '--do_train',
        '--per_device_train_batch_size', str(batch_size),  # Configurable batch size (default: 64 for gpu-a100-small)
        '--learning_rate', '1e-5',
        '--num_train_epochs', '3',
        '--train_group_size', '1',  # GitHub Tevatron: number of positive passages per query
        '--query_max_len', '128',  # Standard query length (required by Tevatron)
        '--passage_max_len', '512',  # Standard passage length (required by Tevatron)
        '--gradient_accumulation_steps', str(gradient_accumulation_steps),  # Accumulate gradients to simulate larger batch
        '--dataloader_num_workers', '4',  # 4 workers for gpu-v100 (4 CPUs available)
        '--fp16', 'False',  # FP16 causes scaler errors - using FP32 instead
        '--bf16', 'False'  # BF16 not tested - using FP32 for stability
    ]
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Running in single-GPU mode with FP32 (in-batch negatives, physical batch: {batch_size}, accumulation: {gradient_accumulation_steps}, effective batch: {effective_batch_size})...")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Training completed! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
