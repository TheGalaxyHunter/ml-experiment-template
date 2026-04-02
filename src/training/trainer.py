"""Generic training loop with callbacks, gradient accumulation, and mixed precision.

The trainer is model-agnostic. It relies on the BaseModel interface
(forward, compute_loss) and the BaseDataModule interface (dataloaders).
All customization happens through callbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, _LRScheduler

from src.config import TrainingConfig
from src.models.base import BaseModel
from src.training.callbacks import Callback, CallbackList
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Generic training loop.

    Supports:
        - Gradient accumulation
        - Mixed precision training (AMP)
        - Gradient clipping
        - Configurable optimizer and scheduler
        - Callback hooks at every stage of training
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        callbacks: Optional[list[Callback]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callback_list = CallbackList(callbacks or [])

        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=config.use_amp)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric: Optional[float] = None

        logger.info(
            "trainer_initialized",
            device=str(self.device),
            params=model.num_parameters(),
            amp=config.use_amp,
            grad_accum=config.gradient_accumulation_steps,
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader[Any],
        val_loader: Optional[torch.utils.data.DataLoader[Any]] = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with training history (losses, metrics per epoch).
        """
        history: dict[str, list[Any]] = {"train_loss": [], "val_loss": []}

        self.callback_list.on_train_begin(trainer=self)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            self.callback_list.on_epoch_begin(epoch=epoch, trainer=self)

            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])

            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                history["val_loss"].append(val_metrics.get("loss", 0.0))

            all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            if self.scheduler is not None:
                self.scheduler.step()
                all_metrics["lr"] = self.scheduler.get_last_lr()[0]

            logger.info("epoch_complete", epoch=epoch, **all_metrics)

            should_stop = self.callback_list.on_epoch_end(
                epoch=epoch, metrics=all_metrics, trainer=self
            )

            if should_stop:
                logger.info("early_stopping_triggered", epoch=epoch)
                break

        self.callback_list.on_train_end(trainer=self)
        return history

    def _train_epoch(self, loader: torch.utils.data.DataLoader[Any]) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp,
            ):
                outputs = self.model.compute_loss(batch)
                loss = outputs["loss"] / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    @torch.no_grad()
    def _validate_epoch(self, loader: torch.utils.data.DataLoader[Any]) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model.compute_loss(batch)
            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def _build_optimizer(self) -> Optimizer:
        """Build optimizer from config."""
        params = self.model.parameters()
        cfg = self.config

        optimizers: dict[str, type[Optimizer]] = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }

        opt_cls = optimizers.get(cfg.optimizer.value)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
        if cfg.optimizer.value == "sgd":
            kwargs["momentum"] = 0.9

        return opt_cls(params, **kwargs)

    def _build_scheduler(self) -> Optional[_LRScheduler]:
        """Build learning rate scheduler from config."""
        schedulers: dict[str, type[_LRScheduler]] = {
            "cosine": CosineAnnealingLR,
            "linear": LinearLR,
            "step": StepLR,
        }

        if self.config.scheduler.value == "none":
            return None

        sched_cls = schedulers.get(self.config.scheduler.value)
        if sched_cls is None:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        if self.config.scheduler.value == "cosine":
            return sched_cls(self.optimizer, T_max=self.config.epochs)
        if self.config.scheduler.value == "step":
            return sched_cls(self.optimizer, step_size=30, gamma=0.1)
        return sched_cls(self.optimizer, total_iters=self.config.epochs)

    def _to_device(self, batch: Any) -> Any:
        """Move batch to the training device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state (model + optimizer + scheduler + trainer state)."""
        extra_state: dict[str, Any] = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }
        if self.scheduler is not None:
            extra_state["scheduler_state_dict"] = self.scheduler.state_dict()
        self.model.save(path, extra_state=extra_state)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load full training state and resume training."""
        checkpoint = BaseModel.load(path, self.model)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric")
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("training_resumed", epoch=self.current_epoch, step=self.global_step)
