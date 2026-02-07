"""
UNET4DownscalingWRF-v2 - Main entry point.

Refactored, modular wind field downscaling.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

config.config.config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    ExperimentConfig,
    UpsampleMode,
    BottleneckType,
)
from .models.unet import UNetFactory
from .data.wrf_dataset import WRFDataModule
from .training.trainer import Trainer, EarlyStopping, CheckpointCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UNET4DownscalingWRF-v2: Modular Wind Field Downscaling"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "info"],
        default="info",
        help="Operation mode",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="resunet",
        choices=["simple", "resunet", "resunet_v2"],
        help="Model architecture",
    )
    parser.add_argument("--n_filters", type=int, default=64, help="Base filters")
    parser.add_argument("--bottleneck", type=str, default="basic", choices=["basic", "dilated", "enhanced"])
    parser.add_argument("--upsampling", type=str, default="bilinear", choices=["bilinear", "convtranspose"])

    # Data
    parser.add_argument("--predictors", type=str, help="Path to predictor NetCDF files")
    parser.add_argument("--targets", type=str, help="Path to target NetCDF files")
    parser.add_argument("--input_vars", nargs="+", default=["U", "V", "W", "T", "P", "HGT", "TKE"])
    parser.add_argument("--target_vars", nargs="+", default=["U", "V"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_path", type=str, help="Path to trained model for prediction")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def create_model(args: argparse.Namespace) -> nn.Module:
    """Create model from args."""
    model_config = ModelConfig(
        in_channels=len(args.input_vars),
        out_channels=len(args.target_vars),
        n_filters=args.n_filters,
        bottleneck=BottleneckType(args.bottleneck),
        upsampling=UpsampleMode(args.upsampling),
    )
    return UNetFactory.create(args.model, model_config)


def train(args: argparse.Namespace):
    """Training pipeline."""
    logger.info("Initializing training pipeline")

    # Create model
    model = create_model(args)
    logger.info(f"Model: {args.model}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data module
    data_module = WRFDataModule(
        predictors_path=args.predictors,
        targets_path=args.targets,
        input_vars=args.input_vars,
        target_vars=args.target_vars,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
    )
    data_module.prepare_data()
    data_module.setup()

    # Training config
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()

    # Callbacks
    callbacks = [
        CheckpointCallback(args.output_dir, every_n_epochs=10, save_best=True),
    ]
    if args.patience:
        callbacks.append(EarlyStopping(patience=args.patience))

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        config=training_config,
        output_dir=args.output_dir,
        callbacks=callbacks,
        device=args.device,
    )

    # Train
    trainer.train(epochs=args.epochs)
    trainer.save_model("final_model.pt")

    logger.info(f"Training complete. Outputs saved to {args.output_dir}")


def predict(args: argparse.Namespace):
    """Inference pipeline."""
    if not args.model_path:
        logger.error("Model path required for prediction")
        return

    logger.info(f"Running inference with model: {args.model_path}")

    # Load model
    model = create_model(args)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Model loaded. Ready for inference.")


def main():
    args = parse_args()

    if args.mode == "info":
        logger.info("UNET4DownscalingWRF-v2")
        logger.info(f"Available models: {UNetFactory.list_models()}")
        logger.info(f"Device: {args.device}")
        logger.info("Use --mode train or --mode predict")

    elif args.mode == "train":
        train(args)

    elif args.mode == "predict":
        predict(args)


if __name__ == "__main__":
    main()
