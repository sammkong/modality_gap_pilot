"""
Torch-based cache helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_cache(data: Any, cache_path: str | Path) -> Path:
    """Save data to a torch cache file, creating parent directories when needed."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def load_cache(cache_path: str | Path, *, map_location: str | torch.device | None = None) -> Any:
    """
    Load data from a torch cache file.

    Raises FileNotFoundError if the cache file does not exist.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Cache path is not a file: {path}")

    return torch.load(path, map_location=map_location)
