from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.config import load_project_config, validate_project_config


class ReproducibilityError(ValueError):
    """Raised when reproducibility settings are invalid."""


@dataclass(slots=True)
class ReproducibilityContext:
    """
    Random-number generators used by the project.

    This dataclass is intentionally not frozen because random generators
    advance their internal state whenever random values are drawn.
    """

    seed: int
    deterministic_algorithms: bool
    numpy_generator: np.random.Generator
    dataloader_generator: torch.Generator
    dataloader_worker_init_fn: Callable[[int], None]


def _seed_dataloader_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy inside one PyTorch DataLoader worker.

    PyTorch assigns each worker its own reproducible initial seed. We reuse
    that value for Python and NumPy so every worker has a different but
    reproducible random sequence.
    """
    del worker_id  # The worker-specific seed is obtained from PyTorch.

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
) -> ReproducibilityContext:
    """
    Seed Python, NumPy, PyTorch, and future DataLoader workers.

    Args:
        seed:
            Common experiment seed.
        deterministic_algorithms:
            Ask PyTorch to use deterministic implementations where available.
        warn_only:
            Warn instead of raising when an operation has no deterministic
            implementation.

    Returns:
        A context containing explicitly seeded generators for NumPy and
        PyTorch DataLoaders.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReproducibilityError(
            f"seed must be an integer; received {type(seed).__name__}."
        )

    # NumPy's legacy global seeding API accepts unsigned 32-bit values.
    if not 0 <= seed < 2**32:
        raise ReproducibilityError(
            f"seed must be between 0 and {2**32 - 1}; received {seed}."
        )

    # Global generators used by Python, NumPy libraries, and PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Preferred NumPy API for code that we write ourselves.
    numpy_generator = np.random.default_rng(seed)

    # Dedicated generator used later by DataLoader shuffling and sampling.
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    # Prefer deterministic operations where PyTorch provides them.
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=warn_only,
    )

    # These settings matter when the same code later runs with CUDA/cuDNN.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic_algorithms

        if deterministic_algorithms:
            torch.backends.cudnn.benchmark = False

    return ReproducibilityContext(
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        numpy_generator=numpy_generator,
        dataloader_generator=dataloader_generator,
        dataloader_worker_init_fn=_seed_dataloader_worker,
    )


def main() -> None:
    """Demonstrate that reseeding reproduces the same random sequences."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs" / "base.yaml"

    config = load_project_config(config_path)
    validate_project_config(config, project_root=repository_root)

    # First run.
    first_context = seed_everything(config.project.seed)

    first_values = {
        "python": random.random(),
        "numpy_global": float(np.random.random()),
        "numpy_generator": float(first_context.numpy_generator.random()),
        "torch_global": float(torch.rand(1).item()),
        "dataloader": float(
            torch.rand(
                1,
                generator=first_context.dataloader_generator,
            ).item()
        ),
    }

    # Reset every generator to the same starting state.
    second_context = seed_everything(config.project.seed)

    second_values = {
        "python": random.random(),
        "numpy_global": float(np.random.random()),
        "numpy_generator": float(second_context.numpy_generator.random()),
        "torch_global": float(torch.rand(1).item()),
        "dataloader": float(
            torch.rand(
                1,
                generator=second_context.dataloader_generator,
            ).item()
        ),
    }

    print("Reproducibility configured successfully.")
    print()
    print(f"Seed: {config.project.seed}")
    print(f"Deterministic algorithms: {second_context.deterministic_algorithms}")
    print()
    print("First run:")
    print(first_values)
    print()
    print("Second run:")
    print(second_values)
    print()
    print(f"Sequences match: {first_values == second_values}")


if __name__ == "__main__":
    main()