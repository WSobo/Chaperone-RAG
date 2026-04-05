import yaml
import os
from pathlib import Path
from chaperone.utils.logger import logger

def load_config(config_path: str = "configs/agent_prompts.yaml") -> dict:
    """Loads a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found at: {path.absolute()}")
        return {}

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return {}
