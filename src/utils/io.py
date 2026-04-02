"""Checkpoint and artifact I/O utilities.

Provides atomic file operations for saving/loading experiment artifacts.
All I/O in this codebase goes through these utilities to ensure
consistency and atomicity.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(state: dict[str, Any], path: str | Path, atomic: bool = True) -> None:
    """Save a checkpoint with optional atomic write.

    Atomic writes prevent corruption from interrupted saves (e.g., OOM
    during checkpointing, SIGTERM from a job scheduler).

    Args:
        state: Dictionary containing the checkpoint state.
        path: Destination file path.
        atomic: If True, write to a temp file first, then rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        temp_path = path.with_suffix(".tmp")
        torch.save(state, temp_path)
        temp_path.rename(path)
    else:
        torch.save(state, path)

    logger.info("checkpoint_saved", path=str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint.

    Args:
        path: Checkpoint file path.
        map_location: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    logger.info("checkpoint_loaded", path=str(path))
    return checkpoint


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("json_saved", path=str(path))


def load_json(path: str | Path) -> Any:
    """Load JSON data."""
    with open(path) as f:
        return json.load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info("yaml_saved", path=str(path))


def load_yaml(path: str | Path) -> Any:
    """Load YAML data."""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: str | Path) -> None:
    """Remove all contents of a directory without removing the directory itself."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        path.mkdir(parents=True)
        logger.info("directory_cleaned", path=str(path))
