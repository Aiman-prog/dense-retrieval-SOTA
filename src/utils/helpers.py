"""Helper utility functions for Path and Context Management."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml"):
    """
    Finds the project root and loads the config file.
    """
    # 1. Get the directory where THIS file (helpers.py) lives
    # 2. Go up two levels to reach the project root (src/utils -> project_root)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"❌ Config not found at {full_path}. Check your folder structure!")
        
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_base_dir() -> Path:
    """Get base directory for all data, returning a Path object."""
    if 'DATA_BASE_DIR' in os.environ:
        return Path(os.environ['DATA_BASE_DIR'])
    
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    return Path(f'/scratch/{user}/dense-retrieval-SOTA')

def get_path(key: str, model_name: str = None) -> Path:
    """
    Centralized path resolver.
    Example: get_path('processed') -> /scratch/user/.../data/processed
    """
    config = load_config()
    base = get_data_base_dir()
    p_cfg = config['paths']
    
    path_map = {
        "base": base,
        "data": base / p_cfg['data_dir'],
        "processed": base / p_cfg['processed_dir'],
        "bright": base / p_cfg['bright_cache'],
        "models": base / p_cfg['models_dir'],
        "results": base / p_cfg['results_dir'],
        "temp_ance": base / "temp_ance_workdir"
    }
    
    if model_name:
        return path_map["models"] / model_name
    return path_map.get(key)

def get_training_context(training_type: str = "inbatch") -> Dict[str, Any]:
    config = load_config()
    recipe = config['training'][training_type]
    model_name = config['model']['base_model']
    
    # Force absolute path resolution
    cache_base = get_path("bright").resolve() / "hub"
    repo_id = model_name.replace("/", "--")
    snapshot_dir = cache_base / f"models--{repo_id}" / "snapshots"
    
    final_base_model = model_name # Default fallback

    if snapshot_dir.exists():
        # Filter out hidden files and get actual directories
        snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
        if snapshots:
            # Sort to get the most recent or consistent one
            chosen_snapshot = sorted(snapshots)[-1]
            # Check if config.json is there (exists() or is_symlink() for HF cache)
            cfg = chosen_snapshot / "config.json"
            if cfg.exists() or cfg.is_symlink():
                final_base_model = str(chosen_snapshot)

    return {
        "args": recipe,
        "base_model": final_base_model,
        "max_q": config['model']['query_max_len'],
        "max_p": config['model']['passage_max_len'],
        "pooling": config['model'].get('pooling', 'cls'),
        "normalize": config['model'].get('normalize', False),
        "temperature": config['model'].get('temperature', 0.02),
        "processed_dir": get_path("processed"),
        "output_dir": get_path("models", recipe['model_name']),
        "cache_dir": str(get_path("bright").resolve())
    }