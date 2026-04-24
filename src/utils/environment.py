"""
Environment helpers for local VS Code and Google Colab execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "COLAB_GPU" in os.environ


def get_base_path(config: Mapping[str, Any]) -> Path:
    """
    Resolve the base project path from config based on the current environment.

    Expected config shape:
        {"paths": {"base_local": "./", "base_colab": "/content/drive/..."}}
    """
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("Missing 'paths' section in config.")

    key = "base_colab" if is_colab() else "base_local"
    base_path = paths.get(key)
    if not base_path:
        raise KeyError(f"Missing '{key}' in config['paths'].")

    return Path(base_path).expanduser().resolve()


def resolve_from_base(config: Mapping[str, Any], relative_path: str) -> Path:
    """Return an absolute path below the configured environment-specific base path."""
    if not relative_path:
        raise ValueError("relative_path must be a non-empty string.")
    if Path(relative_path).is_absolute():
        raise ValueError("relative_path must be a relative path, not an absolute path.")

    return get_base_path(config) / relative_path
