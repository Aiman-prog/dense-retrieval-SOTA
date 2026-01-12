"""Pre-download models for offline training on DelftBlue."""

import os
import sys
from pathlib import Path

# Add src to path (same pattern as train_rocketqa.py)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config

# Try to import get_data_base_dir, define it locally if not available
try:
    from utils.helpers import get_data_base_dir
except ImportError:
    # Fallback: define get_data_base_dir if it's not in helpers.py
    def get_data_base_dir() -> str:
        """Get base directory for all data (datasets, processed files, models)."""
        if 'DATA_BASE_DIR' in os.environ:
            return os.environ['DATA_BASE_DIR']
        user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        return f'/scratch/{user}/dense-retrieval-SOTA'

def main():
    """Download models to cache for offline training."""
    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    
    model_name = config['model']['base_model']
    
    # Set cache location (matches bright_loader.py and run_rocketqa_delftblue.sh)
    base_dir = get_data_base_dir()
    cache_dir = f'{base_dir}/data/bright'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HuggingFace cache environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    
    print("=" * 80)
    print("Pre-downloading Models for Offline Training")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    try:
        print(f"Downloading {model_name}...")
        from sentence_transformers import SentenceTransformer
        
        # Download the model (this will cache it)
        model = SentenceTransformer(model_name)
        print(f"✅ Model '{model_name}' downloaded successfully!")
        
        # Verify it's cached
        model_cache_path = f"{cache_dir}/hub/models--{model_name.replace('/', '--')}"
        if os.path.exists(model_cache_path):
            print(f"✅ Model cached at: {model_cache_path}")
        else:
            print(f"⚠️  Warning: Model cache path not found at expected location")
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ Model preparation complete!")
    print("You can now run training in offline mode.")
    print("=" * 80)

if __name__ == "__main__":
    main()

