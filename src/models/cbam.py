"""
CBAM (Convolutional Block Attention Module) implementation.

Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
https://arxiv.org/abs/1807.06521

CBAM consists of:
1. Channel Attention Module: Uses both AvgPool and MaxPool for richer feature descriptors
2. Spatial Attention Module: Focuses on spatial locations that matter

This is more comprehensive than SqueezeExcitation (which only uses AvgPool).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module.
    
    Uses both average-pooled and max-pooled features to infer attention maps.
    More expressive than SqueezeExcitation which only uses average pooling.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (both pools go through same FC layers)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average branch
        avg_out = self.mlp(self.avg_pool(x))
        # Max branch
        max_out = self.mlp(self.max_pool(x))
        # Combine
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module.
    
    Generates a spatial attention map to focus on informative regions.
    Uses channel-wise average and max pooling before convolution.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # Conv after concatenating avg and max pooled features
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate
        combined = torch.cat([avg_out, max_out], dim=1)
        # Conv + sigmoid
        out = self.sigmoid(self.conv(combined))
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Sequential application of Channel Attention followed by Spatial Attention.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for attention (default: 16)
        spatial_kernel: Kernel size for spatial attention (default: 7)
    
    Example:
        >>> cbam = CBAM(channels=64, reduction=16)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = cbam(x)  # Same shape as input
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Channel Attention
        x = x * self.channel_attention(x)
        # Step 2: Spatial Attention
        x = x * self.spatial_attention(x)
        return x


class CBAMResidualBlock(nn.Module):
    """Residual Block with CBAM attention.
    
    Integrates CBAM into a residual learning framework.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout: float = 0.0,
        use_cbam: bool = True,
        reduction: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # CBAM attention
        self.cbam = CBAM(out_channels, reduction) if use_cbam else None

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

        # Apply CBAM before residual addition
        if self.cbam:
            out = self.cbam(out)

        out = out + residual
        out = F.relu(out)
        return out
