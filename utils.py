import yaml
import os
from pathlib import Path

def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    Returns a dictionary with configuration values.
    If file doesn't exist or error occurs, returns empty dict (relying on code defaults).
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        else:
            # Try to look for it in the parent directory or relative to the script if not found in cwd
            script_dir = Path(__file__).parent
            alt_path = script_dir / config_path
            if alt_path.exists():
                 with open(alt_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            
            print(f"⚠️  Config file '{config_path}' not found. Using internal defaults.")
            return {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}. Using internal defaults.")
        return {}
