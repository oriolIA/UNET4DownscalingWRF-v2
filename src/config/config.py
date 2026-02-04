"""
Configuration system for UNET4DownscalingWRF-v2.

Uses dataclasses for type-safe, composable configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class NormType(Enum):
    """Normalization type for residual blocks."""
    BATCH = "batch"
    INSTANCE = "instance"
    GROUP = "group"


class UpsampleMode(Enum):
    """Upsampling mode for decoder."""
    BILINEAR = "bilinear"
    CONV_TRANSPOSE = "convtranspose"
    PIXEL_SHUFFLE = "pixelshuffle"


class BottleneckType(Enum):
    """Bottleneck type."""
    BASIC = "basic"
    DILATED = "dilated"
    ENHANCED = "enhanced"


@dataclass
class ModelConfig:
    """Base model configuration."""
    in_channels: int = 7
    out_channels: int = 1
    n_filters: int = 64
    dropout: float = 0.0
    dilation: int = 1
    use_attention: bool = False
    bottleneck: BottleneckType = BottleneckType.BASIC
    upsampling: UpsampleMode = UpsampleMode.BILINEAR
    norm_type: NormType = NormType.INSTANCE


@dataclass
class ResUNetConfig(ModelConfig):
    """ResUNet specific configuration."""
    pass


@dataclass
class ResUNetV2Config(ModelConfig):
    """ResUNetV2 specific configuration."""
    bottleneck_dilations: Tuple[int, int] = (2, 4)


@dataclass
class DataConfig:
    """Data loading configuration."""
    predictors_path: str = ""
    targets_path: str = ""
    input_vars: List[str] = field(default_factory=lambda: ["U", "V", "W", "T", "P", "HGT", "TKE"])
    target_vars: List[str] = field(default_factory=lambda: ["U", "V"])
    train_heights: List[int] = field(default_factory=lambda: [80])
    target_heights: List[int] = field(default_factory=lambda: [80])
    patch_size: int = 64
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.8


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    gradient_clip: Optional[float] = None
    validate_every: int = 1
    save_every: int = 10
    patience: Optional[int] = None
    mixed_precision: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs"
    experiment_name: Optional[str] = None
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
