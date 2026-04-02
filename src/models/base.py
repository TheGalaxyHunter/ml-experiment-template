"""Abstract base model with save/load and standard interface.

All models in this framework must subclass BaseModel. This enforces
a consistent interface that the trainer can rely on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base model.

    Subclass this to add a new model. You must implement:
        - forward(): the forward pass
        - compute_loss(): loss computation given a batch

    The base class provides:
        - save() / load(): checkpoint serialization
        - num_parameters(): parameter counting
        - freeze() / unfreeze(): parameter freezing
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def compute_loss(
        self, batch: tuple[torch.Tensor, ...], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Compute loss for a batch.

        Must return a dict with at least a 'loss' key.
        Additional keys are logged as metrics.

        Example return: {'loss': loss_val, 'accuracy': acc}
        """
        ...

    def save(self, path: str | Path, extra_state: Optional[dict[str, Any]] = None) -> None:
        """Save model checkpoint.

        Args:
            path: File path for the checkpoint.
            extra_state: Additional state to save (e.g., optimizer state, epoch).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
        }
        if extra_state:
            checkpoint.update(extra_state)

        # Atomic save: write to temp file, then rename
        temp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(path)

        logger.info("saved_checkpoint", path=str(path), params=self.num_parameters())

    @classmethod
    def load(cls, path: str | Path, model: BaseModel, strict: bool = True) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Checkpoint file path.
            model: Model instance to load weights into.
            strict: Whether to strictly enforce state dict key matching.

        Returns:
            The full checkpoint dict (includes any extra state).
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("loaded_checkpoint", path=str(path))
        return checkpoint

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze all parameters (disable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("model_frozen", model=self.__class__.__name__)

    def unfreeze(self) -> None:
        """Unfreeze all parameters (enable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("model_unfrozen", model=self.__class__.__name__)
