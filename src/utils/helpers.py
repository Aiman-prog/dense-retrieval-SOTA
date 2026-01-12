"""Helper utility functions."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def get_data_base_dir() -> str:
    """
    Get base directory for all data (datasets, processed files, models).
    
    Uses environment variable DATA_BASE_DIR if set, otherwise defaults to
    DelftBlue structure: /scratch/${USER}/dense-retrieval-SOTA
    
    To override (e.g., on macOS):
        export DATA_BASE_DIR="/path/to/your/data"
    
    Returns:
        Base directory path
    """
    # Allow override via environment variable (professional standard)
    if 'DATA_BASE_DIR' in os.environ:
        return os.environ['DATA_BASE_DIR']
    
    # Default to DelftBlue structure
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    return f'/scratch/{user}/dense-retrieval-SOTA'

