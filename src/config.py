"""Pydantic configuration validation.

Validates Hydra configs at load time so misconfigurations fail fast,
not three hours into a training run.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Activation(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class Scheduler(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class MonitorMode(str, Enum):
    MIN = "min"
    MAX = "max"


class TrackingBackend(str, Enum):
    WANDB = "wandb"
    MLFLOW = "mlflow"


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    name: str
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = Field(ge=0.0, le=1.0, default=0.1)
    activation: Activation = Activation.RELU
    input_dim: int = Field(gt=0, default=784)
    output_dim: int = Field(gt=0, default=10)


class DataConfig(BaseModel):
    """Data loading configuration."""

    name: str = "default"
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: Optional[str] = "data/test"
    batch_size: int = Field(gt=0, default=64)
    num_workers: int = Field(ge=0, default=4)
    pin_memory: bool = True
    shuffle_train: bool = True


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = True
    patience: int = Field(gt=0, default=10)
    min_delta: float = Field(ge=0.0, default=1e-4)


class DistributedConfig(BaseModel):
    """Distributed training configuration."""

    enabled: bool = False
    backend: str = "nccl"


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(gt=0, default=100)
    lr: float = Field(gt=0.0, default=1e-3)
    weight_decay: float = Field(ge=0.0, default=1e-4)
    optimizer: Optimizer = Optimizer.ADAMW
    scheduler: Scheduler = Scheduler.COSINE

    gradient_accumulation_steps: int = Field(gt=0, default=1)
    max_grad_norm: float = Field(gt=0.0, default=1.0)

    use_amp: bool = True

    save_every_n_epochs: int = Field(gt=0, default=5)
    save_best: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: MonitorMode = MonitorMode.MIN

    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    @field_validator("lr")
    @classmethod
    def lr_must_be_reasonable(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError(f"Learning rate {v} is suspiciously high. Did you mean {v:.0e}?")
        return v


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    enabled: bool = True
    backend: TrackingBackend = TrackingBackend.WANDB
    project: str = "my-experiment"
    tags: list[str] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    name: str = "my-experiment"
    seed: int = Field(ge=0, default=42)
    output_dir: str = "outputs"


class ExperimentConfig(BaseModel):
    """Root configuration that composes all sub-configs."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="mlp"))
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    def summary(self) -> str:
        """Return a human-readable config summary."""
        lines = [
            f"Experiment: {self.project.name}",
            f"  Seed: {self.project.seed}",
            f"  Model: {self.model.name} (dims={self.model.hidden_dims})",
            f"  LR: {self.training.lr}, Epochs: {self.training.epochs}",
            f"  AMP: {self.training.use_amp}, Grad Accum: {self.training.gradient_accumulation_steps}",
            f"  Tracking: {self.tracking.backend.value} ({'enabled' if self.tracking.enabled else 'disabled'})",
        ]
        return "\n".join(lines)
