"""Training callbacks: early stopping, checkpointing, logging.

Callbacks decouple cross-cutting concerns from the training loop.
The trainer calls hooks at defined points; callbacks react accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = get_logger(__name__)


class Callback:
    """Base callback class. Override any hook you need."""

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, epoch: int, trainer: Trainer) -> None:
        pass

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, Any], trainer: Trainer
    ) -> Optional[bool]:
        """Return True to trigger early stopping."""
        return None


class CallbackList:
    """Container that dispatches calls to multiple callbacks."""

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def on_train_begin(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, Any], trainer: Trainer
    ) -> bool:
        """Returns True if any callback signals early stopping."""
        for cb in self.callbacks:
            result = cb.on_epoch_end(epoch, metrics, trainer)
            if result is True:
                return True
        return False


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.wait = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, Any], trainer: Trainer
    ) -> Optional[bool]:
        current = metrics.get(self.monitor)
        if current is None:
            return None

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(
                    "early_stopping",
                    monitor=self.monitor,
                    best=self.best,
                    patience=self.patience,
                )
                return True
        return None


class ModelCheckpoint(Callback):
    """Save model checkpoints based on a monitored metric."""

    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_every_n_epochs: int = 5,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.best: Optional[float] = None

    def _is_better(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, Any], trainer: Trainer
    ) -> Optional[bool]:
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Periodic save
        if (epoch + 1) % self.save_every_n_epochs == 0:
            path = self.dirpath / f"epoch_{epoch:04d}.pt"
            trainer.save_checkpoint(path)

        # Best model save
        if self.save_best:
            current = metrics.get(self.monitor)
            if current is not None and self._is_better(current):
                self.best = current
                trainer.best_metric = current
                path = self.dirpath / "best.pt"
                trainer.save_checkpoint(path)
                logger.info("new_best_model", monitor=self.monitor, value=current, epoch=epoch)

        return None


class MetricLogger(Callback):
    """Log metrics at the end of each epoch."""

    def __init__(self, log_to_console: bool = True) -> None:
        self.log_to_console = log_to_console
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, Any], trainer: Trainer
    ) -> Optional[bool]:
        record = {"epoch": epoch, **metrics}
        self.history.append(record)

        if self.log_to_console:
            parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            logger.info("metrics", epoch=epoch, **{k: v for k, v in metrics.items()})
        return None
