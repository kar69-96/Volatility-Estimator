"""
Configuration loading utilities.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


# Custom Exception Classes
class ConfigError(Exception):
    """Raised when there are issues with configuration."""
    pass


class DataError(Exception):
    """Raised when there are issues with data loading or validation."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration

    Raises:
        ConfigError: If config file cannot be loaded
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config file: {e}") from e
    except Exception as e:
        raise ConfigError(f"Failed to load config: {e}") from e


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that config contains required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names

    Raises:
        ConfigError: If required keys are missing
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")

