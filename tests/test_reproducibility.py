"""Tests for reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.training.reproducibility import check_reproducibility, fork_rng, set_seed


class TestSetSeed:
    def test_python_random_deterministic(self) -> None:
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_deterministic(self) -> None:
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestForkRng:
    def test_rng_state_preserved(self) -> None:
        set_seed(42)
        before = torch.randn(5)

        set_seed(42)
        with fork_rng():
            # Consume some randomness inside the fork
            _ = torch.randn(100)

        after = torch.randn(5)
        assert torch.equal(before, after)

    def test_fork_with_seed(self) -> None:
        with fork_rng(seed=99):
            a = torch.randn(5)
        with fork_rng(seed=99):
            b = torch.randn(5)
        assert torch.equal(a, b)


class TestCheckReproducibility:
    def test_deterministic_function(self) -> None:
        def fn() -> torch.Tensor:
            return torch.randn(10)

        assert check_reproducibility(seed=42, fn=fn) is True

    def test_with_numpy(self) -> None:
        def fn() -> torch.Tensor:
            arr = np.random.rand(10)
            return torch.from_numpy(arr)

        assert check_reproducibility(seed=42, fn=fn) is True
