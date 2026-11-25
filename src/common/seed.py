"""Seeding for reproducible runs.

Call ``seed_everything(seed)`` once at the very start of every run (standard seeds: 0, 1, 2).
For PyTorch DataLoaders with workers, also pass ``worker_init_fn=seed_worker`` and
``generator=torch_generator(seed)`` so augmentations are reproducible.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

STANDARD_SEEDS: tuple[int, ...] = (0, 1, 2)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    Args:
        seed: The run seed. Use 0, 1 or 2 for reported experiments.
        deterministic: Also force deterministic cuDNN kernels. Slower; leave off for
            training and turn on only when debugging a reproducibility problem.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (seeds third-party code that uses the legacy global RNG)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker its own reproducible NumPy/random seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for a DataLoader's shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def numpy_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``; prefer this over the global RNG in new code."""
    return np.random.default_rng(seed)
