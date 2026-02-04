"""
UNet model definitions for UNET4DownscalingWRF-v2.

Clean, composable implementations with factory pattern.
"""

import logging
from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ResidualBlock,
    PreActivationResBlock,
    DilatedBottleneck,
    EnhancedDilatedBottleneck,
    UpBlock,
    AttentionGate,
)
from ..config.config import (
    ModelConfig,
    ResUNetConfig,
    ResUNetV2Config,
    UpsampleMode,
    BottleneckType,
)

logger = logging.getLogger(__name__)


class BaseUNet(nn.Module, ABC):
    """Abstract base class for UNet architectures."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._build_model()

    @abstractmethod
    def _build_model(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def _make_upsample(
        self, in_channels: int, out_channels: int, mode: UpsampleMode
    ) -> nn.Module:
        """Create upsampling layer."""
        if mode == UpsampleMode.BILINEAR:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        elif mode == UpsampleMode.CONV_TRANSPOSE:
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            raise ValueError(f"Unknown upsampling mode: {mode}")

    def _align_and_concat(
        self, upsampled: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """Align spatial dimensions and concatenate skip connections."""
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        return torch.cat([upsampled, skip], dim=1)


class SimpleUNet(BaseUNet):
    """Simple UNet with double convolutions."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _build_model(self):
        cfg = self.config
        n = cfg.n_filters

        # Encoder
        self.enc1 = self._double_conv(cfg.in_channels, n)
        self.enc2 = self._double_conv(n, n * 2)
        self.enc3 = self._double_conv(n * 2, n * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._double_conv(n * 4, n * 8)

        # Decoder
        self.up3 = self._make_upsample(n * 8, n * 4, cfg.upsampling)
        self.dec3 = self._double_conv(n * 8, n * 4)

        self.up2 = self._make_upsample(n * 4, n * 2, cfg.upsampling)
        self.dec2 = self._double_conv(n * 4, n * 2)

        self.up1 = self._make_upsample(n * 2, n, cfg.upsampling)
        self.dec1 = self._double_conv(n * 2, n)

        # Output
        self.final = nn.Conv2d(n, cfg.out_channels, kernel_size=1)

    def _double_conv(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self._align_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)


class ResUNet(BaseUNet):
    """Residual UNet with configurable components."""

    def __init__(self, config: ResUNetConfig):
        super().__init__(config)

    def _build_model(self):
        cfg = self.config
        n = cfg.n_filters

        # Log configuration
        logger.info(f"ResUNet: {n} filters, bottleneck={cfg.bottleneck}, upsampling={cfg.upsampling}")

        # Encoder
        self.enc1 = ResidualBlock(cfg.in_channels, n, cfg.dilation, cfg.dropout, cfg.use_attention)
        self.enc2 = ResidualBlock(n, n * 2, cfg.dilation, cfg.dropout, cfg.use_attention)
        self.enc3 = ResidualBlock(n * 2, n * 4, cfg.dilation, cfg.dropout, cfg.use_attention)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        if cfg.bottleneck == BottleneckType.ENHANCED:
            self.bottleneck = EnhancedDilatedBottleneck(n * 4, n * 8)
        elif cfg.bottleneck == BottleneckType.DILATED:
            self.bottleneck = DilatedBottleneck(n * 4, n * 8)
        else:
            self.bottleneck = ResidualBlock(
                n * 4, n * 8, cfg.dilation, cfg.dropout, cfg.use_attention
            )

        # Decoder
        self.up3 = self._make_upsample(n * 8, n * 4, cfg.upsampling)
        self.up2 = self._make_upsample(n * 4, n * 2, cfg.upsampling)
        self.up1 = self._make_upsample(n * 2, n, cfg.upsampling)

        self.dec3 = ResidualBlock(n * 8, n * 4, cfg.dilation, cfg.dropout, cfg.use_attention)
        self.dec2 = ResidualBlock(n * 4, n * 2, cfg.dilation, cfg.dropout, cfg.use_attention)
        self.dec1 = ResidualBlock(n * 2, n, cfg.dilation, cfg.dropout, cfg.use_attention)

        # Output
        self.final = nn.Conv2d(n, cfg.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self._align_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)


class ResUNetV2(BaseUNet):
    """Coherent Residual UNet (V2) with pre-activation blocks."""

    def __init__(self, config: ResUNetV2Config):
        self.bottleneck_dilations = config.bottleneck_dilations
        super().__init__(config)

    def _build_model(self):
        cfg = self.config
        n = cfg.n_filters
        d1, d2 = self.bottleneck_dilations

        # Encoder (stride-based downsampling)
        self.enc1 = PreActivationResBlock(cfg.in_channels, n, stride=1, dilation=1)
        self.enc2 = PreActivationResBlock(n, n * 2, stride=2, dilation=1)
        self.enc3 = PreActivationResBlock(n * 2, n * 4, stride=2, dilation=1)

        # Bottleneck
        self.bottleneck1 = PreActivationResBlock(
            n * 4, n * 8, stride=2, dilation=d1, use_attention=cfg.use_attention
        )
        self.bottleneck2 = PreActivationResBlock(
            n * 8, n * 8, stride=1, dilation=d2, use_attention=cfg.use_attention
        )

        # Decoder
        self.up3 = UpBlock(n * 8, n * 4)
        self.dec3 = PreActivationResBlock(n * 8, n * 4, stride=1, use_attention=cfg.use_attention)

        self.up2 = UpBlock(n * 4, n * 2)
        self.dec2 = PreActivationResBlock(n * 4, n * 2, stride=1)

        self.up1 = UpBlock(n * 2, n)
        self.dec1 = PreActivationResBlock(n * 2, n, stride=1)

        # Output
        self.final = nn.Conv2d(n, cfg.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck1(e3)
        b = self.bottleneck2(b)

        # Decoder
        d3 = self.up3(b)
        d3 = self._align_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)


# Factory pattern for model creation
class UNetFactory:
    """Factory for creating UNet models."""

    _models = {
        "simple": SimpleUNet,
        "resunet": ResUNet,
        "resunet_v2": ResUNetV2,
    }

    @classmethod
    def create(cls, name: str, config: ModelConfig) -> nn.Module:
        """Create a model by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name](config)

    @classmethod
    def register(cls, name: str, model_cls: Type[nn.Module]) -> None:
        """Register a new model type."""
        cls._models[name] = model_cls

    @classmethod
    def list_models(cls) -> list:
        """List available models."""
        return list(cls._models.keys())
