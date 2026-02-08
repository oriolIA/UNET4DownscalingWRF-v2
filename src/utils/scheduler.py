"""
Learning rate schedulers for WRF downscaling.
"""

import logging
from typing import Optional

import torch
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

logger = logging.getLogger(__name__)


class LRSchedulerFactory:
    """Factory for creating LR schedulers."""

    _schedulers = {
        "step": StepLR,
        "exp": ExponentialLR,
        "cosine": CosineAnnealingLR,
        "plateau": ReduceLROnPlateau,
    }

    @classmethod
    def create(
        cls,
        name: str,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        step_size: int = 30,
        gamma: float = 0.1,
        eta_min: float = 1e-6,
        patience: int = 10,
        factor: float = 0.5,
    ):
        """Create scheduler by name."""
        if name not in cls._schedulers:
            raise ValueError(f"Unknown scheduler: {name}. Available: {list(cls._schedulers.keys())}")

        if name == "step":
            return StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif name == "exp":
            return ExponentialLR(optimizer, gamma=gamma)

        elif name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

        elif name == "plateau":
            return ReduceLROnPlateau(
                optimizer, mode="min", factor=factor, patience=patience, verbose=True
            )

    @classmethod
    def list_schedulers(cls) -> list:
        """List available schedulers."""
        return list(cls._schedulers.keys())


if __name__ == "__main__":
    import torch.nn as nn
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = LRSchedulerFactory.create("cosine", optimizer, epochs=100)
    print(f"Scheduler: {type(scheduler).__name__}")
    print(f"Available: {LRSchedulerFactory.list_schedulers()}")
