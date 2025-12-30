"""
Caching utilities for data persistence.
"""

from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_cache_path(cache_dir: str, filename: str) -> Path:
    """
    Get full path for a cache file, ensuring directory exists.

    Args:
        cache_dir: Cache directory path
        filename: Cache filename

    Returns:
        Full path to cache file
    """
    cache_path = ensure_directory(cache_dir)
    return cache_path / filename

