"""
Training pipeline with callbacks for UNET4DownscalingWRF-v2.

Modular training with early stopping, logging, and checkpointing.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import TrainingConfig

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_begin(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer") -> None:
        pass

    def on_batch_begin(self, trainer: "Trainer", batch_idx: int) -> None:
        pass

    def on_batch_end(self, trainer: "Trainer", batch_idx: int) -> None:
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state: Optional[Dict[str, Any]] = None

    def on_epoch_end(self, trainer: "Trainer") -> None:
        current_loss = trainer.epoch_metrics.get("val_loss", float("inf"))

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {trainer.epoch} epochs")
            trainer.should_stop = True


class CheckpointCallback(Callback):
    """Model checkpointing callback."""

    def __init__(self, save_dir: str, every_n_epochs: int = 10, save_best: bool = True):
        self.save_dir = Path(save_dir)
        self.every_n_epochs = every_n_epochs
        self.save_best = save_best
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer: "Trainer") -> None:
        # Save every N epochs
        if trainer.epoch % self.every_n_epochs == 0:
            self._save_checkpoint(trainer, f"checkpoint_epoch_{trainer.epoch}.pt")

        # Save best model
        if self.save_best:
            current_loss = trainer.epoch_metrics.get("val_loss", float("inf"))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._save_checkpoint(trainer, "best_model.pt")

    def _save_checkpoint(self, trainer: "Trainer", filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "epoch": trainer.epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": trainer.epoch_metrics,
        }, path)
        logger.info(f"Saved checkpoint: {path}")


class LoggingCallback(Callback):
    """Logging callback."""

    def __init__(self, log_every: int = 10):
        self.log_every = log_every

    def on_batch_end(self, trainer: "Trainer", batch_idx: int) -> None:
        if batch_idx % self.log_every == 0:
            loss = trainer.batch_metrics.get("loss", 0)
            logger.info(f"Batch {batch_idx}: loss={loss:.4f}")


class Trainer:
    """Training manager with callback support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        output_dir: str = "outputs",
        callbacks: Optional[List[Callback]] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or []
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.epoch = 0
        self.should_stop = False

        self.batch_metrics: Dict[str, float] = {}
        self.epoch_metrics: Dict[str, float] = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Run training loop."""
        epochs = epochs or self.config.epochs

        logger.info(f"Starting training for {epochs} epochs on {self.device}")

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            # Train epoch
            train_loss = self._train_epoch()

            # Validate
            val_loss = self._validate() if self.val_loader else 0

            # Compute metrics
            self.epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self)

            # Check early stopping
            if self.should_stop:
                break

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info("Training complete")
        return self.epoch_metrics

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Forward pass
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            self.batch_metrics = {"loss": loss.item()}

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx)

        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)

        for batch in self.val_loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

        return total_loss / n_batches

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        self.model.eval()
        inputs = inputs.to(self.device)

        with torch.no_grad():
            return self.model(inputs)

    def save_model(self, filename: str) -> None:
        """Save model to file."""
        path = self.output_dir / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.epoch_metrics,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        logger.info(f"Model loaded from {path}")
