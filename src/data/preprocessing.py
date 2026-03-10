"""
Preprocessing utilities for WRF data.

Normalization and feature engineering.
"""

import numpy as np


class WRFNormalizer:
    """Normalize WRF variables for neural network input."""

    STATS = {
        "U": {"mean": 0.0, "std": 10.0, "clip": (-30, 30)},
        "V": {"mean": 0.0, "std": 10.0, "clip": (-30, 30)},
        "W": {"mean": 0.0, "std": 5.0, "clip": (-15, 15)},
        "T": {"mean": 288.0, "std": 20.0, "clip": (250, 330)},
        "P": {"mean": 90000.0, "std": 10000.0, "clip": (50000, 110000)},
        "HGT": {"mean": 500.0, "std": 500.0, "clip": (0, 4000)},
        "TKE": {"mean": 0.0, "std": 5.0, "clip": (0, 25)},
    }

    DEFAULT = {"mean": 0.0, "std": 1.0, "clip": None}

    @classmethod
    def normalize(cls, var_name: str, data: np.ndarray) -> np.ndarray:
        """Normalize a single variable."""
        stats = cls.STATS.get(var_name, cls.DEFAULT)

        if stats["clip"] is not None:
            data = np.clip(data, stats["clip"][0], stats["clip"][1])

        data = (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data

    @classmethod
    def denormalize(cls, var_name: str, data: np.ndarray) -> np.ndarray:
        """Denormalize a single variable."""
        stats = cls.STATS.get(var_name, cls.DEFAULT)
        return data * stats["std"] + stats["mean"]

    @classmethod
    def get_scale_factor(cls) -> float:
        """Get default scale factor for upsampling."""
        return 2.0


class DataPreprocessor:
    """Data preprocessing pipeline."""

    def __init__(self, normalize: bool = True, scale: float = 2.0):
        self.normalize = normalize
        self.scale = scale

    def preprocess(
        self, input_data: np.ndarray, target_data: np.ndarray = None
    ) -> tuple:
        """Preprocess input and target data."""
        if self.normalize:
            # Apply normalization
            input_data = self._normalize_array(input_data)

        if target_data is not None and self.normalize:
            target_data = self._normalize_array(target_data)

        return input_data, target_data

    def _normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Normalize array using global statistics."""
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / (std + 1e-8)
        return data

    def augment(self, input_data: np.ndarray, target_data: np.ndarray) -> tuple:
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            input_data = np.flip(input_data, axis=1).copy()
            target_data = np.flip(target_data, axis=1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            input_data = np.flip(input_data, axis=0).copy()
            target_data = np.flip(target_data, axis=0).copy()

        return input_data, target_data
