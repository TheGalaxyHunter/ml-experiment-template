"""Tests for Pydantic config validation."""

from __future__ import annotations

import pytest

from src.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


class TestModelConfig:
    def test_default_values(self) -> None:
        cfg = ModelConfig(name="test")
        assert cfg.name == "test"
        assert cfg.hidden_dims == [256, 128, 64]
        assert cfg.dropout == 0.1
        assert cfg.input_dim == 784

    def test_custom_values(self) -> None:
        cfg = ModelConfig(
            name="custom",
            hidden_dims=[512, 256],
            dropout=0.5,
            input_dim=100,
            output_dim=5,
        )
        assert cfg.hidden_dims == [512, 256]
        assert cfg.dropout == 0.5

    def test_invalid_dropout(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(name="bad", dropout=1.5)

    def test_invalid_input_dim(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(name="bad", input_dim=-1)


class TestTrainingConfig:
    def test_default_values(self) -> None:
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.lr == 1e-3
        assert cfg.use_amp is True

    def test_high_lr_rejected(self) -> None:
        with pytest.raises(ValueError, match="suspiciously high"):
            TrainingConfig(lr=10.0)

    def test_valid_lr(self) -> None:
        cfg = TrainingConfig(lr=0.5)
        assert cfg.lr == 0.5

    def test_early_stopping_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.early_stopping.enabled is True
        assert cfg.early_stopping.patience == 10


class TestDataConfig:
    def test_default_values(self) -> None:
        cfg = DataConfig()
        assert cfg.batch_size == 64
        assert cfg.num_workers == 4
        assert cfg.pin_memory is True

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError):
            DataConfig(batch_size=0)


class TestExperimentConfig:
    def test_full_config(self) -> None:
        cfg = ExperimentConfig(
            model=ModelConfig(name="mlp"),
            training=TrainingConfig(lr=1e-4, epochs=50),
        )
        assert cfg.model.name == "mlp"
        assert cfg.training.lr == 1e-4
        assert cfg.training.epochs == 50

    def test_summary(self) -> None:
        cfg = ExperimentConfig(model=ModelConfig(name="mlp"))
        summary = cfg.summary()
        assert "mlp" in summary
        assert "Seed" in summary
