"""Reproducibility utilities.

Deterministic seed management for Python, NumPy, PyTorch, and CUDA.
Call set_seed() once at the start of any experiment. This module also
provides context managers for reproducible data loading.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch for deterministic behavior.
            This may reduce performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # PyTorch 2.0+ deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                torch.use_deterministic_algorithms(False)
                logger.warning(
                    "deterministic_fallback",
                    msg="Some ops lack deterministic implementations. Falling back.",
                )

    logger.info("seed_set", seed=seed, deterministic=deterministic)


def get_seed_worker_fn(seed: int) -> callable:
    """Return a worker_init_fn for DataLoader that seeds each worker.

    This ensures that each DataLoader worker has a different but
    reproducible random state.

    Usage:
        DataLoader(..., worker_init_fn=get_seed_worker_fn(42))
    """

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


@contextmanager
def fork_rng(seed: Optional[int] = None) -> Generator[None, None, None]:
    """Context manager that forks the RNG state.

    Useful for running stochastic operations (e.g., data augmentation)
    without affecting the global RNG state.

    Args:
        seed: Optional seed to use within the context. If None, uses a
            random seed derived from the current state.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        for i, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, i)


def check_reproducibility(seed: int, fn: callable, n_checks: int = 3) -> bool:
    """Verify that a function produces identical results across calls.

    Args:
        seed: Seed to use for each call.
        fn: A callable that returns a tensor.
        n_checks: Number of times to call fn for comparison.

    Returns:
        True if all calls produce identical results.
    """
    results = []
    for _ in range(n_checks):
        set_seed(seed, deterministic=True)
        result = fn()
        if isinstance(result, torch.Tensor):
            results.append(result.clone())
        else:
            results.append(result)

    for i in range(1, len(results)):
        if isinstance(results[0], torch.Tensor):
            if not torch.equal(results[0], results[i]):
                return False
        elif results[0] != results[i]:
            return False

    return True
