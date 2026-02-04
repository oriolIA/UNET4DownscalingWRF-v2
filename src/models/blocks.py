"""
Model blocks for UNET4DownscalingWRF-v2.

Building blocks with clean interfaces and composition.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.config import NormType

logger = logging.getLogger(__name__)


class Norm2d(nn.Module):
    """Normalization layer factory."""

    def __init__(self, num_features: int, norm_type: NormType):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == NormType.BATCH:
            self.norm = nn.BatchNorm2d(num_features)
        elif norm_type == NormType.INSTANCE:
            self.norm = nn.InstanceNorm2d(num_features)
        elif norm_type == NormType.GROUP:
            self.norm = nn.GroupNorm(32, num_features)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResidualBlock(nn.Module):
    """Residual block with normalization and optional attention/dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout: float = 0.0,
        use_attention: bool = False,
        norm_type: NormType = NormType.INSTANCE,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = Norm2d(out_channels, norm_type)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = Norm2d(out_channels, norm_type)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Attention mechanism
        self.attention = SqueezeExcitation(out_channels) if use_attention else None

        # Residual connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.attention:
            out = self.attention(out)

        out = out + residual
        out = F.relu(out)
        return out


class PreActivationResBlock(nn.Module):
    """Pre-activation residual block (ResNet V2 style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        use_attention: bool = False,
        norm_type: NormType = NormType.INSTANCE,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Norm2d(in_channels, norm_type)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation
        )

        self.norm2 = Norm2d(out_channels, norm_type)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation
        )

        self.attention = SqueezeExcitation(out_channels) if use_attention else None

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.attention:
            out = self.attention(out)

        return out + residual


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DilatedBottleneck(nn.Module):
    """Dilated bottleneck for larger receptive field."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=2, dilation=2
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        return out + x  # Residual


class EnhancedDilatedBottleneck(nn.Module):
    """Multi-branch dilated bottleneck."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Branch 1: dilation 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.InstanceNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 4, out_channels // 4, kernel_size=3, padding=1, dilation=1
            ),
        )

        # Branch 2: dilation 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.InstanceNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 4, out_channels // 4, kernel_size=3, padding=2, dilation=2
            ),
        )

        # Branch 3: dilation 4
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.InstanceNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 4, out_channels // 4, kernel_size=3, padding=4, dilation=4
            ),
        )

        # Branch 4: dilation 8
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.InstanceNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 4, out_channels // 4, kernel_size=3, padding=8, dilation=8
            ),
        )

        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attention = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        out = self.attention(out)
        return out + x  # Residual


class UpBlock(nn.Module):
    """Upsampling block with configurable mode."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)


class AttentionGate(nn.Module):
    """Attention gating for feature refinement."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * self.sigmoid(psi)
