# Design Philosophy

This document explains the reasoning behind every major design decision in this template. These are opinionated choices, and every opinion has a reason.

## Why Hydra for configuration

**The problem:** ML experiments have dozens of parameters. Teams typically start with `argparse`, graduate to YAML files, and eventually build custom config systems. Each approach has the same failure modes: no composition, no validation, painful overrides.

**Why Hydra solves this:**

- **Composable configs.** Break config into groups (model, data, training) and compose them. Switch models by changing one flag: `python train.py model=transformer`.
- **CLI overrides.** Override any parameter from the command line: `training.lr=1e-4`. No code changes needed for hyperparameter sweeps.
- **Multirun sweeps.** Built-in sweep support: `--multirun training.lr=1e-3,1e-4,1e-5`.
- **Output directory management.** Each run gets a timestamped output directory. No more overwriting previous results.
- **Typed configs.** Combined with Pydantic, configs are validated at load time. A typo in your config fails immediately, not three hours into training.

**Alternatives considered:**

- `argparse`: no composition, no grouping, scales poorly.
- `fire`: clever but implicit; hard to document and validate.
- `jsonargparse`: good for Lightning, but heavier than needed for general use.
- Plain YAML: no validation, no overrides, no sweeps.

## Why Pydantic for config validation

**The problem:** YAML configs are stringly-typed. A learning rate of `"0.001"` (string) silently behaves differently than `0.001` (float) in some frameworks. A missing key crashes at runtime, not at config load time.

**Why Pydantic:**

- Type coercion: `"0.001"` becomes `0.001` automatically.
- Validation at construction: catch errors before they waste GPU hours.
- Field constraints: `lr: float = Field(gt=0)` rejects negative learning rates.
- Custom validators: `lr_must_be_reasonable` catches obviously wrong values.
- Self-documenting: the config class IS the documentation.

## Why abstract base classes

**The problem:** Every new team member implements models and datasets slightly differently. The trainer breaks when someone forgets `compute_loss()`. Integration becomes a guessing game.

**Why ABCs:**

- **Enforce contracts.** `BaseModel` requires `forward()` and `compute_loss()`. Forget one, and Python raises `TypeError` at instantiation, not during training.
- **Enable composition.** The trainer only depends on the `BaseModel` interface. Any model that satisfies the contract works.
- **Reduce onboarding friction.** New contributors know exactly what to implement. No tribal knowledge required.
- **Testable in isolation.** Each component can be tested against its interface without the full training pipeline.

## Why structured logging (not print statements)

**The problem:** `print("loss:", loss)` works for debugging. It does not work when you need to:
- Search logs from 100 parallel runs
- Build monitoring dashboards
- Debug a training run that failed at 3 AM

**Why structlog:**

- **Machine-parseable.** JSON output in production, human-readable in development.
- **Grep-friendly.** Every log entry has structured fields: `grep "metric=val_loss" logs.txt`.
- **Context propagation.** Bind context once (e.g., experiment name) and it appears in all subsequent logs.
- **Zero print statements.** This is a hard rule. All output goes through the logger. This makes it trivial to redirect, filter, or aggregate logs.

## Why this folder structure

```
src/          # Library code (importable, testable)
scripts/      # Entry points (Hydra apps, CLI tools)
configs/      # Experiment configurations (no code)
tests/        # Test suite
docs/         # Documentation
```

**The reasoning:**

- **`src/` is a library.** It has no side effects on import. You can `from src.models.base import BaseModel` in a notebook, a test, or a script.
- **`scripts/` are entry points.** They wire together the library components and run experiments. They should be thin: mostly config resolution and component assembly.
- **`configs/` are data.** They describe experiments, not implement them. Changing an experiment should never require changing code.
- **`tests/` mirror `src/`.** `test_config.py` tests `config.py`. This makes it obvious what is tested and what is not.

**Why not a flat structure:** Flat works for small projects. Beyond a few files, it becomes impossible to tell what depends on what. This structure enforces separation of concerns at the file system level.

## Why callbacks instead of hooks in the training loop

**The problem:** Training loops accumulate cross-cutting concerns: logging, checkpointing, early stopping, learning rate scheduling, metric tracking. Putting all of this in one function creates an unmaintainable monolith.

**Why callbacks:**

- **Single responsibility.** Each callback does one thing: `EarlyStopping` stops training; `ModelCheckpoint` saves models; `MetricLogger` logs metrics.
- **Composable.** Add or remove callbacks without modifying the trainer.
- **Testable.** Test `EarlyStopping` in isolation with mock metrics. No GPU needed.
- **Familiar pattern.** Keras, PyTorch Lightning, and fastai all use callbacks. The mental model is well-established.

## Why atomic checkpoint saves

**The problem:** A training job saves a checkpoint every N epochs. If the save is interrupted (OOM, SIGTERM, disk full), the checkpoint file is corrupted. The next time you resume training, it fails to load.

**Why atomic saves:**

- Write to a temporary file first, then rename. Rename is atomic on all major filesystems.
- If the write is interrupted, the temp file is corrupt but the previous checkpoint is untouched.
- Cost: one extra file operation. Benefit: never lose a checkpoint.

## Why seed management is its own module

**The problem:** "Just set the seed" is not enough. You need to seed Python's `random`, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN, and CUBLAS. Miss one, and your "reproducible" experiment produces different results on different runs.

**Why a dedicated module:**

- **Comprehensive.** `set_seed()` handles all random sources in one call.
- **Worker-aware.** `get_seed_worker_fn()` ensures DataLoader workers are seeded correctly.
- **Verifiable.** `check_reproducibility()` actually tests that a function produces identical outputs.
- **Fork-safe.** `fork_rng()` isolates stochastic operations without polluting the global RNG state.

## On mixed precision (AMP)

Mixed precision is on by default because:
- 2x memory reduction (important for large models)
- 1.5-3x throughput improvement on modern GPUs
- Negligible quality impact for most tasks

It can be disabled with `training.use_amp=false` for tasks where precision matters (e.g., reinforcement learning, some GAN training).

## On gradient accumulation

Gradient accumulation is the poor person's large batch size. It lets you simulate `batch_size * accumulation_steps` effective batch size on a single GPU. This is critical when:
- Your model barely fits in GPU memory
- You want batch size 256 but only have a 16GB GPU
- You are doing distributed training and need consistent effective batch sizes

Default is 1 (no accumulation). Set `training.gradient_accumulation_steps=4` to simulate 4x batch size.
