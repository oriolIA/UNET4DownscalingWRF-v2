"""
ResUNet Model for WRF Downscaling v2

Improved UNet architecture with:
- Residual connections
- Attention gates
- Better skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = out + identity
        return F.relu(out)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUNet(nn.Module):
    """
    ResUNet for WRF downscaling.
    
    Architecture:
    - Encoder: Residual blocks with pooling
    - Bottleneck: Residual blocks
    - Decoder: Upsampling with attention gates
    - Skip connections: Attention-gated
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        features: list = [64, 128, 256, 512]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features:
            self.encoders.append(ResidualBlock(in_ch, feature))
            self.attentions.append(AttentionGate(feature, feature, feature // 2))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_decoders = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                ResidualBlock(feature * 2, feature)
            )
            self.attention_decoders.append(
                AttentionGate(feature, feature, feature // 2)
            )
        
        # Final output
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder with attention
        for i, (upconv, decoder, attention) in enumerate(
            zip(self.upconvs, self.decoders, self.attention_decoders)
        ):
            # Upsample
            x = upconv(x)
            
            # Attention gate on skip connection
            skip = skip_connections[i]
            skip_attended = attention(x, skip)
            
            # Handle size mismatch
            if x.shape != skip_attended.shape:
                x = F.interpolate(x, size=skip_attended.shape[2:])
            
            # Concatenate and decode
            x = torch.cat([skip_attended, x], dim=1)
            x = decoder(x)
        
        # Final output
        return self.final(x)


class ResUNetSmall(nn.Module):
    """Smaller ResUNet for faster training."""
    
    def __init__(self, in_channels: int = 7, out_channels: int = 2):
        super().__init__()
        features = [32, 64, 128, 256]
        self.model = ResUNet(in_channels, out_channels, features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # Test
    model = ResUNet(in_channels=7, out_channels=2)
    
    # Dummy input: (batch, channels, height, width)
    x = torch.randn(1, 7, 100, 100)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
