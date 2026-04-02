"""Metric computation and reporting.

Provides a registry-based evaluator that computes metrics on model
predictions. Supports classification and regression metrics out of
the box, and is extensible via the register_metric decorator.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global metric registry
_METRIC_REGISTRY: dict[str, Callable[..., float]] = {}


def register_metric(name: str) -> Callable[..., Any]:
    """Decorator to register a metric function.

    Usage:
        @register_metric("my_metric")
        def my_metric(preds: Tensor, targets: Tensor) -> float:
            ...
    """

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        _METRIC_REGISTRY[name] = fn
        return fn

    return decorator


@register_metric("accuracy")
def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


@register_metric("mse")
def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error."""
    return F.mse_loss(preds, targets).item()


@register_metric("mae")
def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean absolute error."""
    return F.l1_loss(preds, targets).item()


@register_metric("cross_entropy")
def cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute cross-entropy loss."""
    return F.cross_entropy(preds, targets).item()


class Evaluator:
    """Metric evaluator.

    Collects predictions and targets across batches, then computes
    all requested metrics at once.
    """

    def __init__(self, metrics: Optional[list[str]] = None) -> None:
        """Initialize evaluator.

        Args:
            metrics: List of metric names to compute. Must be registered
                in the metric registry. Defaults to ["accuracy"].
        """
        self.metric_names = metrics or ["accuracy"]
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

        # Validate metric names
        for name in self.metric_names:
            if name not in _METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric '{name}'. Available: {list(_METRIC_REGISTRY.keys())}"
                )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of predictions and targets."""
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Compute all registered metrics on accumulated predictions."""
        if not self._preds:
            logger.warning("evaluator_empty", msg="No predictions collected.")
            return {}

        all_preds = torch.cat(self._preds, dim=0)
        all_targets = torch.cat(self._targets, dim=0)

        results = {}
        for name in self.metric_names:
            metric_fn = _METRIC_REGISTRY[name]
            results[name] = metric_fn(all_preds, all_targets)

        logger.info("evaluation_complete", **results)
        return results

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self._preds.clear()
        self._targets.clear()
