"""Pre-download models for offline training on DelftBlue."""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config, get_data_base_dir

def main():
    # 1. Setup Paths
    config_path = project_root / 'config' / 'config.yaml'
    config = load_config(str(config_path))
    model_name = config['model']['base_model']
    
    base_dir = get_data_base_dir()
    cache_dir = Path(base_dir) / 'data' / 'bright'

    # 2. CLEANUP: If flat files exist, they block the hub creation.
    # We check for config.json in the root. If it's there, we wipe the folder.
    if (cache_dir / "config.json").exists():
        print(f"🧹 Cleaning up legacy flat files in {cache_dir}...")
        shutil.rmtree(cache_dir)
    
    os.makedirs(cache_dir, exist_ok=True)

    # 3. ENVIRONMENT: Only set HF_HOME to force modern Hub structure
    # We explicitly UNSET the old deprecated ones to avoid confusion
    os.environ.pop('TRANSFORMERS_CACHE', None)
    os.environ.pop('HF_DATASETS_CACHE', None)
    os.environ['HF_HOME'] = str(cache_dir)
    
    # Ensure we aren't in offline mode while downloading!
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    print("=" * 80)
    print("Pre-downloading Models for Offline Training")
    print(f"Model: {model_name}")
    print(f"Target HF_HOME: {cache_dir}")
    print("=" * 80)

    try:
        from transformers import AutoTokenizer, AutoModel
        from datasets import load_dataset
        
        print(f"📥 Downloading {model_name}...")
        # Note: No cache_dir= argument! We rely on os.environ['HF_HOME']
        AutoTokenizer.from_pretrained(model_name)
        AutoModel.from_pretrained(model_name)
        
        print(f"✅ Model download successful!")

        # Download MS MARCO dataset for training mixture
        print("\n" + "=" * 80)
        print("📥 Downloading MS MARCO dataset for training mixture...")
        msmarco_config = config['data'].get('msmarco', {})
        msmarco_name = msmarco_config.get('name', 'sentence-transformers/msmarco-hard-negatives')
        msmarco_subset = msmarco_config.get('subset', 'triplet')
        
        print(f"Dataset: {msmarco_name}")
        print(f"Subset: {msmarco_subset}")
        load_dataset(msmarco_name, msmarco_subset, split='train', cache_dir=str(cache_dir))
        print(f"✅ MS MARCO dataset downloaded!")

        # 4. VERIFICATION: Look for the specific Hub snapshots folder
        repo_id = model_name.replace("/", "--")
        hub_path = cache_dir / "hub" / f"models--{repo_id}" / "snapshots"
        
        if hub_path.exists() and any(hub_path.iterdir()):
            actual_snapshot = list(hub_path.iterdir())[0]
            print(f"\n✨ SUCCESS: Hub structure created!")
            print(f"📍 Snapshot Path: {actual_snapshot}")
        else:
            print(f"❌ ERROR: Hub structure still missing in {cache_dir}/hub")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()