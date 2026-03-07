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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context.
    
    Captures features at multiple dilation rates for better
    multi-scale representation. Important for WRF downscaling
    where atmospheric features vary across scales.
    
    Reference: https://arxiv.org.05587
/abs/1706    """
    
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list = [6, 12, 18]):
        super().__init__()
        
        self.atrous_rates = atrous_rates
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions at different rates
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling - use LayerNorm instead of InstanceNorm for 1x1
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 1, 1], elementwise_affine=False),
            nn.ReLU(inplace=True)
        )
        
        # Project all to output channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        
        # 1x1 conv
        feat1 = self.conv1(x)
        
        # Atrous convolutions
        feats = [feat1]
        for atrous_conv in self.atrous_convs:
            feats.append(atrous_conv(x))
        
        # Global pooling (upsampled)
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        feats.append(global_feat)
        
        # Concatenate and project
        out = torch.cat(feats, dim=1)
        out = self.project(out)
        
        return out


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


class ChannelAttention(nn.Module):
    """Channel Attention Module (part of CBAM)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module (part of CBAM)."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Combines channel and spatial attention for more focused feature learning.
    Reference: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


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


class CBAMResidualBlock(nn.Module):
    """Residual block with CBAM attention (channel + spatial)."""
    
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAM(out_channels, reduction, kernel_size)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply CBAM (channel + spatial attention)
        out = self.cbam(out)
        
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
                 attention: str = "se"):
        super().__init__()
        
        if attention == "cbam":
            self.block = CBAMResidualBlock(in_channels, out_channels)
        elif attention == "se":
            self.block = SEResidualBlock(in_channels, out_channels)
        elif attention == "none":
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
    - Encoder: Residual/SE/CBAM blocks with pooling
    - Bottleneck: SE/CBAM-enhanced residual blocks
    - Decoder: Upsampling with attention gates
    - Skip connections: Attention-gated
    - Multi-scale output (deep supervision)
    
    Attention options:
    - "se": Squeeze-and-Excitation (channel attention)
    - "cbam": CBAM (channel + spatial attention)
    - "none": Basic residual blocks
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        features: list = [64, 128, 256, 512],
        attention: str = "se",  # "se", "cbam", or "none"
        use_deep_supervision: bool = True,
        use_aspp: bool = True,  # Use ASPP in bottleneck
        aspp_rates: list = [6, 12, 18]  # Atrous rates for ASPP
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deep_supervision = use_deep_supervision
        self.attention = attention
        self.use_aspp = use_aspp
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        in_ch = in_channels
        for feature in features:
            self.encoders.append(
                EncoderBlock(in_ch, feature, attention=attention)
            )
            self.attentions.append(AttentionGate(feature, feature, feature // 2))
            in_ch = feature
        
        # Bottleneck with optional ASPP
        bottleneck_in = features[-1]
        bottleneck_out = features[-1] * 2
        
        if use_aspp:
            # ASPP: multi-scale atrous convolutions
            self.aspp = ASPP(bottleneck_in, bottleneck_out, aspp_rates)
            # Additional refinement after ASPP
            self.bottleneck_refine = self._make_bottleneck_block(bottleneck_out, bottleneck_out, attention)
        else:
            # Standard bottleneck
            self.bottleneck = self._make_bottleneck_block(bottleneck_in, bottleneck_out, attention)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_decoders = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                self._make_decoder_block(feature * 2, feature, attention)
            )
            self.attention_decoders.append(
                AttentionGate(feature, feature, feature // 2)
            )
        
        # Deep supervision heads (multi-scale output)
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
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
    
    def _make_bottleneck_block(self, in_ch: int, out_ch: int, attention: str):
        if attention == "cbam":
            return CBAMResidualBlock(in_ch, out_ch)
        elif attention == "se":
            return SEResidualBlock(in_ch, out_ch)
        else:
            return ResidualBlock(in_ch, out_ch)
    
    def _make_decoder_block(self, in_ch: int, out_ch: int, attention: str):
        if attention == "cbam":
            return CBAMResidualBlock(in_ch, out_ch)
        elif attention == "se":
            return SEResidualBlock(in_ch, out_ch)
        else:
            return ResidualBlock(in_ch, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        encoder_features = []
        
        for encoder in self.encoders:
            x, feat = encoder(x)
            skip_connections.append(feat)
            encoder_features.append(feat)
        
        # Bottleneck with optional ASPP for multi-scale context
        if self.use_aspp:
            x = self.aspp(x)
            x = self.bottleneck_refine(x)
        else:
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


class UNetPlusPlus(nn.Module):
    """
    U-Net++: Nested U-Net with dense skip connections.
    
    Reference: https://arxiv.org/abs/1807.10165
    
    Key improvements over standard U-Net:
    - Dense skip pathways (nested U-nets)
    - Deep supervision (multi-scale outputs)
    - Better feature fusion at different scales
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        features: list = [64, 128, 256, 512],
        attention: str = "se",
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder (standard)
        self.enc0 = DoubleConv(in_channels, features[0])
        self.enc1 = DoubleConv(features[0], features[1])
        self.enc2 = DoubleConv(features[1], features[2])
        self.enc3 = DoubleConv(features[2], features[3])
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3])
        
        # Upsample layers (match channels)
        self.up3 = nn.ConvTranspose2d(features[3], features[3], 2, stride=2)  # 512->512
        self.up2 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)  # 512->256
        self.up1 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)  # 256->128
        self.up0 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)  # 128->64
        
        # Decoder blocks with dense skip connections
        # Level 3: bottleneck + enc3 -> 512+512 = 1024 -> 512
        self.dec3 = DoubleConv(features[3] * 2, features[3])
        
        # Level 2: up3(dec3) + enc2 -> 256 + 256 = 512 -> 256
        self.dec2 = DoubleConv(features[2] + features[2], features[2])
        
        # Level 1: up2(d2) + enc1 -> 128 + 128 = 256 -> 128
        self.dec1 = DoubleConv(features[1] + features[1], features[1])
        
        # Level 0: up1(d1) + enc0 -> 64 + 64 = 128 -> 64
        self.dec0 = DoubleConv(features[0] + features[0], features[0])
        
        # Deep supervision heads
        if use_deep_supervision:
            self.deep_sup1 = nn.Conv2d(features[1], out_channels, 1)
            self.deep_sup2 = nn.Conv2d(features[2], out_channels, 1)
        
        # Final output
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)                           # 64, 128x128
        e1 = self.enc1(self.pool(e0))               # 128, 64x64
        e2 = self.enc2(self.pool(e1))               # 256, 32x32
        e3 = self.enc3(self.pool(e2))               # 512, 16x16
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))          # 512, 8x8
        
        # Decoder with dense skip connections
        # Level 3
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # 1024->512, 16x16
        
        # Level 2
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 768->256, 32x32
        
        # Level 1
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 384->128, 64x64
        
        # Level 0 (final)
        output = self.dec0(torch.cat([self.up0(d1), e0], dim=1))  # 192->64, 128x128
        output = self.final(output)
        
        # Deep supervision
        if self.use_deep_supervision:
            outputs = [output]
            
            sup1 = self.deep_sup1(d1)
            sup1_up = F.interpolate(sup1, size=output.shape[2:], mode='bilinear', align_corners=False)
            outputs.append(sup1_up)
            
            sup2 = self.deep_sup2(d2)
            sup2_up = F.interpolate(sup2, size=output.shape[2:], mode='bilinear', align_corners=False)
            outputs.append(sup2_up)
            
            return outputs
        
        return output


class UNetPlusPlusSmall(nn.Module):
    """Smaller UNet++ for faster training."""
    
    def __init__(self, in_channels: int = 7, out_channels: int = 2):
        super().__init__()
        features = [32, 64, 128, 256]
        self.model = UNetPlusPlus(in_channels, out_channels, features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Model factory
def create_model(model_name: str, in_channels: int = 7, out_channels: int = 2, **kwargs):
    """Create model by name.
    
    Options:
    - "resunet": ResUNet with SE attention (default)
    - "resunet_cbam": ResUNet with CBAM attention (channel + spatial)
    - "resunet_none": ResUNet without attention
    - "resunet_small": Smaller ResUNet
    - "resunet_deep": ResUNet with deep supervision
    - "resunet_aspp": ResUNet with ASPP multi-scale (RECOMMENDED for WRF)
    - "unet_classic": Classic UNet baseline
    - "unet++": UNet++ with dense skip connections
    - "unet++_small": Smaller UNet++
    - "efficientnet_b0": UNet with EfficientNet-B0 encoder
    - "efficientnet_b2": UNet with EfficientNet-B2 encoder
    - "efficientnet_b4": UNet with EfficientNet-B4 encoder
    - "convnext_tiny": UNet with ConvNeXt-Tiny encoder
    - "convnext_small": UNet with ConvNeXt-Small encoder
    """
    attention = kwargs.get("attention", "se")
    use_aspp = kwargs.get("use_aspp", False)
    
    # Import EfficientNet encoder
    try:
        from .efficientnet_encoder import create_encoder_model
        has_efficientnet = True
    except ImportError:
        has_efficientnet = False
    
    models = {
        "resunet": lambda: ResUNet(in_channels, out_channels, attention="se"),
        "resunet_cbam": lambda: ResUNet(in_channels, out_channels, attention="cbam"),
        "resunet_none": lambda: ResUNet(in_channels, out_channels, attention="none"),
        "resunet_small": ResUNetSmall,
        "resunet_deep": lambda: ResUNet(in_channels, out_channels, attention="se", use_deep_supervision=True),
        "resunet_aspp": lambda: ResUNet(in_channels, out_channels, attention="se", use_aspp=True),
        "resunet_aspp_cbam": lambda: ResUNet(in_channels, out_channels, attention="cbam", use_aspp=True),
        "unet_classic": UNetClassic,
        "unet++": lambda: UNetPlusPlus(in_channels, out_channels, attention=attention),
        "unet++_cbam": lambda: UNetPlusPlus(in_channels, out_channels, attention="cbam"),
        "unet++_small": UNetPlusPlusSmall,
    }
    
    # Add EfficientNet/ConvNeXt models if available
    if has_efficientnet:
        encoder_models = {
            "efficientnet_b0": lambda: create_encoder_model("b0", in_channels=in_channels, out_channels=out_channels, **kwargs),
            "efficientnet_b2": lambda: create_encoder_model("b2", in_channels=in_channels, out_channels=out_channels, **kwargs),
            "efficientnet_b4": lambda: create_encoder_model("b4", in_channels=in_channels, out_channels=out_channels, **kwargs),
            "convnext_tiny": lambda: create_encoder_model("convnext_tiny", in_channels=in_channels, out_channels=out_channels, **kwargs),
            "convnext_small": lambda: create_encoder_model("convnext_small", in_channels=in_channels, out_channels=out_channels, **kwargs),
        }
        models.update(encoder_models)
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


if __name__ == "__main__":
    # Test all models
    x = torch.randn(1, 7, 100, 100)
    
    print("=" * 50)
    print("Testing ResUNet (CBAM Attention)")
    model = ResUNet(in_channels=7, out_channels=2, attention="cbam")
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
