"""
ResUNet Model for WRF Downscaling v2

Improved UNet architecture with:
- Residual connections
- Attention gates
- Squeeze-and-Excitation blocks
- Multi-scale output (deep supervision)
- Multiple encoder options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    """Residual block with SE attention."""
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels, reduction)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE
        out = self.se(out)
        
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


class EncoderBlock(nn.Module):
    """Encoder block with configurable architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_residual: bool = True, use_se: bool = True):
        super().__init__()
        
        if use_se:
            self.block = SEResidualBlock(in_channels, out_channels)
        elif use_residual:
            self.block = ResidualBlock(in_channels, out_channels)
        else:
            self.block = DoubleConv(in_channels, out_channels)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor):
        feat = self.block(x)
        return self.pool(feat), feat


class ResUNet(nn.Module):
    """
    ResUNet for WRF downscaling.
    
    Architecture:
    - Encoder: Residual/SE blocks with pooling
    - Bottleneck: SE-enhanced residual blocks
    - Decoder: Upsampling with attention gates
    - Skip connections: Attention-gated
    - Multi-scale output (deep supervision)
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        features: list = [64, 128, 256, 512],
        use_se: bool = True,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features:
            self.encoders.append(
                EncoderBlock(in_ch, feature, use_se=use_se)
            )
            self.attentions.append(AttentionGate(feature, feature, feature // 2))
            in_ch = feature
        
        # Bottleneck
        if use_se:
            self.bottleneck = SEResidualBlock(features[-1], features[-1] * 2)
        else:
            self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_decoders = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            if use_se:
                self.decoders.append(SEResidualBlock(feature * 2, feature))
            else:
                self.decoders.append(ResidualBlock(feature * 2, feature))
            self.attention_decoders.append(
                AttentionGate(feature, feature, feature // 2)
            )
        
        # Deep supervision heads (multi-scale output)
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            # Use channels from each decoder level (reversed features)
            # After each decoder block, channels = features[::-1][i]
            for feature in reversed(features):
                self.deep_supervision_heads.append(
                    nn.Sequential(
                        nn.Conv2d(feature, feature // 2, 1),
                        nn.BatchNorm2d(feature // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(feature // 2, out_channels, 1)
                    )
                )
        
        # Final output
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        encoder_features = []
        
        for encoder in self.encoders:
            x, feat = encoder(x)
            skip_connections.append(feat)
            encoder_features.append(feat)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Store features for deep supervision
        deep_features = []
        
        # Decoder with attention
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder, attention) in enumerate(
            zip(self.upconvs, self.decoders, self.attention_decoders)
        ):
            # Upsample
            x = upconv(x)
            
            # Attention gate on skip connection
            skip = skip_connections[i]
            
            # Handle size mismatch BEFORE attention gate
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            skip_attended = attention(x, skip)
            
            # Concatenate and decode
            x = torch.cat([skip_attended, x], dim=1)
            x = decoder(x)
            
            # Store for deep supervision (before upsampling to next level)
            if self.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_features.append(x)
        
        # Final output
        output = self.final(x)
        
        # Deep supervision
        if self.use_deep_supervision:
            outputs = [output]
            for i, head in enumerate(self.deep_supervision_heads):
                if i < len(deep_features):
                    scale_out = head(deep_features[i])
                    outputs.append(scale_out)
            return outputs
        
        return output


class ResUNetSmall(nn.Module):
    """Smaller ResUNet for faster training."""
    
    def __init__(self, in_channels: int = 7, out_channels: int = 2):
        super().__init__()
        features = [32, 64, 128, 256]
        self.model = ResUNet(in_channels, out_channels, features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResUNetDeepSupervision(ResUNet):
    """ResUNet with explicit deep supervision."""
    pass  # Same as parent with deep supervision enabled


class UNetClassic(nn.Module):
    """Classic UNet without bells and whistles (baseline)."""
    
    def __init__(self, in_channels: int = 7, out_channels: int = 2):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)


# Model factory
def create_model(model_name: str, in_channels: int = 7, out_channels: int = 2, **kwargs):
    """Create model by name."""
    models = {
        "resunet": ResUNet,
        "resunet_small": ResUNetSmall,
        "resunet_se": lambda: ResUNet(in_channels, out_channels, use_se=True),
        "resunet_deep": lambda: ResUNet(in_channels, out_channels, use_deep_supervision=True),
        "unet_classic": UNetClassic,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


if __name__ == "__main__":
    # Test all models
    x = torch.randn(1, 7, 100, 100)
    
    print("=" * 50)
    print("Testing ResUNet (SE + Attention)")
    model = ResUNet(in_channels=7, out_channels=2, use_se=True)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, list):
            print(f"  Output scales: {[o.shape for o in y]}")
        else:
            print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("=" * 50)
    print("Testing ResUNet (no SE)")
    model = ResUNet(in_channels=7, out_channels=2, use_se=False)
    with torch.no_grad():
        y = model(x)
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("=" * 50)
    print("Testing UNet Classic")
    model = UNetClassic(in_channels=7, out_channels=2)
    with torch.no_grad():
        y = model(x)
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
