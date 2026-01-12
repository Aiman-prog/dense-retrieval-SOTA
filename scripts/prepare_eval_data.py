"""Pre-download earth_science dataset for offline evaluation."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

from data.bright_loader import BRIGHTLoader
from utils.helpers import load_config

def main():
    print("Pre-downloading earth_science dataset...")
    
    # Load config
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    
    # Initialize loader
    loader = BRIGHTLoader(config_path=str(project_root / 'config' / 'config.yaml'))
    
    # Load dataset (this downloads if not cached)
    print("Loading BRIGHT dataset...")
    loader.load_dataset()
    
    # Access earth_science to ensure it's downloaded
    print("Accessing earth_science domain...")
    corpus = loader.get_corpus('earth_science')
    queries = loader.get_queries('earth_science')
    qrels = loader.get_qrels('earth_science')
    
    print(f"\nDownloaded successfully!")
    print(f"  Documents: {len(corpus)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")
    print(f"\nDataset cached at: {loader.cache_dir}")

if __name__ == "__main__":
    main()

