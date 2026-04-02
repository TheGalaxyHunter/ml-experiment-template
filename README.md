# ml-experiment-template

**Opinionated ML experiment framework for reproducible research.**

Every ML team reinvents experiment infrastructure. This template encodes hard-won patterns from shipping ML at Microsoft and building research pipelines at IISc and Amazon. Start a new experiment in minutes, not hours.

---

## Features

- **Hydra config management**: composable, overridable YAML configs with full type validation
- **Experiment tracking**: first-class support for Weights & Biases and MLflow
- **Reproducibility**: deterministic seed management across Python, NumPy, PyTorch, and CUDA
- **Training infrastructure**: callbacks, gradient accumulation, mixed-precision (AMP), multi-GPU via DDP
- **Checkpoint management**: atomic saves, best-model tracking, automatic resume
- **CI/CD pipeline**: lint, type-check, and test on every push
- **Docker support**: reproducible environments from dev to production

## Quick Start

### Option 1: Cookiecutter (recommended)

```bash
pip install cookiecutter
cookiecutter gh:TheGalaxyHunter/ml-experiment-template
```

### Option 2: Clone and customize

```bash
git clone https://github.com/TheGalaxyHunter/ml-experiment-template.git my-experiment
cd my-experiment
rm -rf .git && git init
pip install -e ".[dev]"
```

### Run an experiment

```bash
# Train with default config
python scripts/train.py

# Override config from the command line (Hydra)
python scripts/train.py training.lr=1e-4 training.epochs=50

# Use a named experiment config
python scripts/train.py +experiment=example

# Multi-run sweep
python scripts/train.py --multirun training.lr=1e-3,1e-4,1e-5
```

## Project Structure

```
ml-experiment-template/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile                    # train, test, lint, format
├── Dockerfile
├── .github/workflows/ci.yml   # CI pipeline
├── configs/
│   ├── config.yaml             # Main Hydra config
│   ├── model/default.yaml
│   ├── data/default.yaml
│   ├── training/default.yaml
│   └── experiment/example.yaml # Named experiment overrides
├── src/
│   ├── config.py               # Pydantic config validation
│   ├── data/base.py            # Abstract dataset + dataloader factory
│   ├── models/base.py          # Abstract model with save/load
│   ├── training/
│   │   ├── trainer.py          # Training loop with callbacks
│   │   ├── callbacks.py        # EarlyStopping, Checkpointing, Logging
│   │   └── reproducibility.py  # Seed management
│   ├── evaluation/evaluator.py # Metrics and reporting
│   └── utils/
│       ├── logging.py          # Structured logging
│       └── io.py               # Checkpoint I/O
├── scripts/
│   ├── train.py                # Main entry point
│   └── evaluate.py
├── tests/
│   ├── test_config.py
│   └── test_reproducibility.py
└── docs/
    └── philosophy.md           # Design decisions explained
```

## Philosophy

This template is opinionated by design. Every choice serves a specific purpose:

| Choice | Rationale |
|--------|-----------|
| Hydra for config | Composable configs, CLI overrides, multirun sweeps. No more argparse spaghetti. |
| Pydantic validation | Catch config errors at load time, not 3 hours into training. |
| Abstract base classes | Enforce interface contracts. New models/datasets plug in without touching the trainer. |
| Structured logging | Machine-parseable logs. Grep-friendly. Dashboard-ready. |
| Separate concerns | `src/` for library code, `scripts/` for entry points, `configs/` for experiments. |

For the full rationale, see [docs/philosophy.md](docs/philosophy.md).

## Common Commands

```bash
make train                  # Run training with default config
make test                   # Run test suite
make lint                   # Lint with ruff
make format                 # Auto-format with ruff
make typecheck              # Type checking with mypy
make docker-build           # Build Docker image
make docker-run             # Run training in Docker
make clean                  # Remove artifacts and caches
```

## Extending the Template

### Add a new model

1. Create `src/models/my_model.py`
2. Subclass `BaseModel` and implement `forward()`, `compute_loss()`
3. Add a config file `configs/model/my_model.yaml`
4. Run: `python scripts/train.py model=my_model`

### Add a new dataset

1. Create `src/data/my_dataset.py`
2. Subclass `BaseDataModule` and implement `setup()`, `train_dataloader()`, `val_dataloader()`
3. Add a config file `configs/data/my_dataset.yaml`
4. Run: `python scripts/train.py data=my_dataset`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `pyproject.toml` for full dependency list

## License

MIT. See [LICENSE](LICENSE) for details.

---

Built by [Ankit Das](https://github.com/TheGalaxyHunter) at [Twinning Labs](https://twinninglabs.com).
