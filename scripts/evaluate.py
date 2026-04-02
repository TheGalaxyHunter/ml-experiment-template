"""Evaluation entry point.

Load a trained model and compute metrics on the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.training.reproducibility import set_seed
from src.utils.logging import get_logger, setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    raw_config = OmegaConf.to_container(cfg, resolve=True)
    config = ExperimentConfig(**raw_config)  # type: ignore[arg-type]

    set_seed(config.project.seed)

    logger.info("evaluation_started", config=config.summary())

    # NOTE: Implement evaluation by loading your model and test data:
    #
    #   from src.models.my_model import MyModel
    #   from src.data.my_dataset import MyDataModule
    #   from src.evaluation.evaluator import Evaluator
    #
    #   model = MyModel(config.model)
    #   BaseModel.load("checkpoints/best.pt", model)
    #
    #   data = MyDataModule(config.data)
    #   data.setup(stage="test")
    #
    #   evaluator = Evaluator(metrics=["accuracy", "cross_entropy"])
    #   for batch in data.test_dataloader():
    #       preds = model(batch[0])
    #       evaluator.update(preds, batch[1])
    #
    #   results = evaluator.compute()
    #   logger.info("results", **results)

    logger.info("template_ready", msg="Implement your model and data module to run evaluation.")


if __name__ == "__main__":
    main()
