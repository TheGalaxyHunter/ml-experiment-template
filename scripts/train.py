"""Main training entry point.

This is a Hydra application. Config resolution order:
1. configs/config.yaml (base defaults)
2. Config group overrides (model=X, data=Y, training=Z)
3. Experiment configs (+experiment=example)
4. CLI overrides (training.lr=1e-4)
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.training.callbacks import EarlyStopping, MetricLogger, ModelCheckpoint
from src.training.reproducibility import set_seed
from src.utils.logging import get_logger, setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a model with the given configuration."""
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    # Convert OmegaConf to dict, then validate with Pydantic
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    config = ExperimentConfig(**raw_config)  # type: ignore[arg-type]

    logger.info("experiment_started", config=config.summary())

    # Reproducibility
    set_seed(config.project.seed)

    # Build callbacks
    callbacks = [
        MetricLogger(),
        ModelCheckpoint(
            dirpath=Path("checkpoints"),
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode.value,
            save_best=config.training.save_best,
            save_every_n_epochs=config.training.save_every_n_epochs,
        ),
    ]

    if config.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=config.training.monitor_metric,
                patience=config.training.early_stopping.patience,
                min_delta=config.training.early_stopping.min_delta,
                mode=config.training.monitor_mode.value,
            )
        )

    logger.info(
        "training_config",
        model=config.model.name,
        epochs=config.training.epochs,
        lr=config.training.lr,
        optimizer=config.training.optimizer.value,
        amp=config.training.use_amp,
    )

    # NOTE: To run actual training, implement a concrete model and data module.
    # This template provides the infrastructure; plug in your model and data:
    #
    #   from src.models.my_model import MyModel
    #   from src.data.my_dataset import MyDataModule
    #
    #   model = MyModel(config.model)
    #   data = MyDataModule(config.data)
    #   data.setup(stage="fit")
    #
    #   trainer = Trainer(model=model, config=config.training, callbacks=callbacks)
    #   trainer.fit(data.train_dataloader(), data.val_dataloader())

    logger.info("template_ready", msg="Implement your model and data module to start training.")


if __name__ == "__main__":
    main()
