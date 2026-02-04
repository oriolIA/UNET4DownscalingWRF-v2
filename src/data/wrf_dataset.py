"""
Data loading utilities for WRF data.

Handles NetCDF loading, preprocessing, and dataset creation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from .preprocessing import WRFNormalizer, DataPreprocessor

logger = logging.getLogger(__name__)


class WRFDataset(Dataset):
    """Dataset for WRF downscaling."""

    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]

    def __init__(
        self,
        predictor_files: List[Path],
        target_files: Optional[List[Path]] = None,
        input_vars: List[str] = None,
        target_vars: List[str] = None,
        patch_size: int = 64,
        transform: Optional[callable] = None,
        is_train: bool = True,
    ):
        self.predictor_files = predictor_files
        self.target_files = target_files or predictor_files
        self.input_vars = input_vars or self.VARIABLES
        self.target_vars = target_vars or ["U", "V"]
        self.patch_size = patch_size
        self.transform = transform
        self.is_train = is_train

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load spatial dimensions from first file."""
        with xr.open_dataset(self.predictor_files[0]) as ds:
            self.lat_size = ds.dims.get("lat", ds.dims.get("y", 50))
            self.lon_size = ds.dims.get("lon", ds.dims.get("x", 51))
            self.time_size = ds.dims.get("time", 24)
            self.levels = ds.dims.get("lev", 9)

    def _load_sample(self, file_path: Path, variables: List[str]) -> np.ndarray:
        """Load and stack variables from a NetCDF file."""
        with xr.open_dataset(file_path) as ds:
            arrays = []
            for var in variables:
                if var in ds.variables:
                    data = ds[var].values
                    # Average over levels if multi-level
                    if len(data.shape) == 4:  # (time, lev, lat, lon)
                        data = data.mean(axis=1)
                    arrays.append(data)
                else:
                    shape = (self.time_size, self.lat_size, self.lon_size)
                    arrays.append(np.zeros(shape, dtype=np.float32))

            return np.stack(arrays, axis=-1)  # (time, lat, lon, ch)

    def __len__(self) -> int:
        return len(self.predictor_files) * self.time_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx = idx // self.time_size
        time_idx = idx % self.time_size

        # Load data
        predictors = self._load_sample(self.predictor_files[file_idx], self.input_vars)
        targets = self._load_sample(self.target_files[file_idx], self.target_vars)

        # Extract single time step
        x = predictors[time_idx]  # (lat, lon, ch_in)
        y = targets[time_idx]  # (lat, lon, ch_out)

        # Normalize
        for i, var in enumerate(self.input_vars):
            x[..., i] = WRFNormalizer.normalize(var, x[..., i])
        for i, var in enumerate(self.target_vars):
            y[..., i] = WRFNormalizer.normalize(var, y[..., i])

        # Convert to tensors (ch, lat, lon)
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy(y.transpose(2, 0, 1)).float()

        # Random crop for training
        if self.is_train and self.patch_size > 0:
            _, h, w = x.shape
            if h >= self.patch_size and w >= self.patch_size:
                i = np.random.randint(0, h - self.patch_size + 1)
                j = np.random.randint(0, w - self.patch_size + 1)
                x = x[:, i:i + self.patch_size, j:j + self.patch_size]
                y = y[:, i:i + self.patch_size, j:j + self.patch_size]

        return {"input": x, "target": y, "file": str(self.predictor_files[file_idx]), "time_idx": time_idx}


class WRFDataModule:
    """Data module for WRF downscaling experiments."""

    def __init__(
        self,
        predictors_path: str,
        targets_path: Optional[str] = None,
        input_vars: List[str] = None,
        target_vars: List[str] = None,
        batch_size: int = 32,
        patch_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        pin_memory: bool = True,
    ):
        self.predictors_path = Path(predictors_path)
        self.targets_path = Path(targets_path) if targets_path else self.predictors_path
        self.input_vars = input_vars or WRFDataset.VARIABLES
        self.target_vars = target_vars or ["U", "V"]
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.pin_memory = pin_memory

        self.predictor_files: List[Path] = []
        self.train_dataset: Optional[WRFDataset] = None
        self.val_dataset: Optional[WRFDataset] = None
        self.test_dataset: Optional[WRFDataset] = None

    def prepare_data(self):
        """Discover NetCDF files."""
        self.predictor_files = sorted(self.predictors_path.glob("*.nc"))
        logger.info(f"Found {len(self.predictor_files)} NetCDF files in {self.predictors_path}")

        if not self.predictor_files:
            raise ValueError(f"No NetCDF files found in {self.predictors_path}")

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test splits."""
        if not self.predictor_files:
            self.prepare_data()

        n_files = len(self.predictor_files)
        n_train = int(n_files * self.train_ratio)
        n_val = int(n_files * (1 - self.train_ratio) / 2)

        train_files = self.predictor_files[:n_train]
        val_files = self.predictor_files[n_train:n_train + n_val]
        test_files = self.predictor_files[n_train + n_val:]

        self.train_dataset = WRFDataset(
            train_files, self.targets_path, self.input_vars, self.target_vars,
            self.patch_size, is_train=True,
        )
        self.val_dataset = WRFDataset(
            val_files, self.targets_path, self.input_vars, self.target_vars,
            patch_size=0, is_train=False,
        )
        self.test_dataset = WRFDataset(
            test_files, self.targets_path, self.input_vars, self.target_vars,
            patch_size=0, is_train=False,
        )

        logger.info(f"Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
