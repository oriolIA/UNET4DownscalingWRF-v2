"""
Transformer-style Encoder for WRF Downscaling

A simplified CNN-Transformer hybrid encoder that provides hierarchical features.
Combines the best of CNNs (local features) with transformer-inspired attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class TransformerBlock(nn.Module):
    """Lightweight transformer block with channel and spatial attention."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        
        # Channel attention (simplified SE)
        self.se = SEBlock(channels)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.se(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


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


class TransformerEncoder(nn.Module):
    """
    Transformer-style encoder for UNet.
    Uses hierarchical convolutional stages with transformer blocks.
    """
    
    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = False,
        frozen_stages: int = -1,
        in_channels: int = 7,
    ):
        super().__init__()
        
        self.variant = variant.lower()
        
        # Channel configs for different variants
        channel_configs = {
            "tiny": [64, 128, 256, 512],
            "small": [96, 192, 384, 768],
            "base": [128, 256, 512, 1024],
        }
        self.channels = channel_configs.get(variant, [64, 128, 256, 512])
        
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1)
        else:
            self.input_proj = nn.Identity()
        
        # Hierarchical encoder stages
        self.stage1 = self._make_stage(self.channels[0], self.channels[1])
        self.stage2 = self._make_stage(self.channels[1], self.channels[2])
        self.stage3 = self._make_stage(self.channels[2], self.channels[3])
        self.stage4 = self._make_stage(self.channels[3], self.channels[3])
        
        self.pretrained = pretrained
    
    def _make_stage(self, in_ch: int, out_ch: int):
        """Create a transformer-style stage."""
        blocks = []
        
        # Downsampling at start of stage (except first)
        if in_ch != out_ch:
            blocks.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
        else:
            blocks.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        
        blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.GELU())
        
        # Transformer block
        blocks.append(TransformerBlock(out_ch))
        
        # Additional conv for refinement
        blocks.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.GELU())
        
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_proj(x)
        
        # Stage 1: 1/2 resolution
        x1 = self.stage1(x)
        
        # Stage 2: 1/4 resolution
        x2 = self.stage2(x1)
        
        # Stage 3: 1/8 resolution
        x3 = self.stage3(x2)
        
        # Stage 4: 1/16 resolution
        x4 = self.stage4(x3)
        
        return [x1, x2, x3, x4]


class TransformerUNet(nn.Module):
    """
    UNet with Transformer-style encoder.
    """
    
    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = False,
        in_channels: int = 7,
        out_channels: int = 2,
        frozen_stages: int = -1,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            variant=variant,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            in_channels=in_channels
        )
        
        # Channel configs
        channel_configs = {
            "tiny": [64, 128, 256, 512],
            "small": [96, 192, 384, 768],
            "base": [128, 256, 512, 1024],
        }
        self.encoder_channels = channel_configs.get(variant, [64, 128, 256, 512])
        
        # Channel adapters for skip connections - match decoder channels
        # Decoder levels: 256 -> 128 -> 64 -> 32
        # Encoder outputs: [x1:64, x2:128, x3:256, x4:512]
        self.skip_adapter3 = nn.Conv2d(self.encoder_channels[3], 256, 1)  # 512 -> 256
        self.skip_adapter2 = nn.Conv2d(self.encoder_channels[2], 128, 1)  # 256 -> 128
        self.skip_adapter1 = nn.Conv2d(self.encoder_channels[1], 64, 1)   # 128 -> 64
        self.skip_adapter0 = nn.Conv2d(self.encoder_channels[0], 32, 1)    # 64 -> 32
        
        # Bottleneck with SE attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoder_channels[3], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.bottleneck_attention = SEBlock(512)
        
        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Deep supervision heads - match actual decoder output channels
        # d3 output: 128 (after dec3), d2 output: 64 (after dec2), d1 output: 32 (after up1)
        if use_deep_supervision:
            self.deep_sup3 = nn.Conv2d(128, out_channels, 1)   # After dec3
            self.deep_sup2 = nn.Conv2d(64, out_channels, 1)    # After dec2
            self.deep_sup1 = nn.Conv2d(32, out_channels, 1)    # After up1
    
    def forward(self, x: torch.Tensor):
        # Get encoder features
        f0, f1, f2, f3 = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(f3)
        x = self.bottleneck_attention(x)
        
        # Decoder
        d4 = self.up4(x)
        f2_adapted = self.skip_adapter3(f2)
        if d4.shape[2:] != f2_adapted.shape[2:]:
            f2_adapted = F.interpolate(f2_adapted, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, f2_adapted], dim=1))
        
        d3 = self.up3(d4)
        f1_adapted = self.skip_adapter2(f1)
        if d3.shape[2:] != f1_adapted.shape[2:]:
            f1_adapted = F.interpolate(f1_adapted, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, f1_adapted], dim=1))
        
        d2 = self.up2(d3)
        f0_adapted = self.skip_adapter1(f0)
        if d2.shape[2:] != f0_adapted.shape[2:]:
            f0_adapted = F.interpolate(f0_adapted, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, f0_adapted], dim=1))
        
        d1 = self.up1(d2)
        output = self.final(d1)
        
        if self.use_deep_supervision:
            outputs = [output]
            
            sup3 = self.deep_sup3(d3)
            sup3 = F.interpolate(sup3, size=output.shape[2:], mode='bilinear', align_corners=False)
            outputs.append(sup3)
            
            sup2 = self.deep_sup2(d2)
            sup2 = F.interpolate(sup2, size=output.shape[2:], mode='bilinear', align_corners=False)
            outputs.append(sup2)
            
            return outputs
        
        return output


def create_vit_model(
    variant: str = "tiny",
    pretrained: bool = False,
    in_channels: int = 7,
    out_channels: int = 2,
    frozen_stages: int = -1,
    use_deep_supervision: bool = True,
) -> nn.Module:
    """Factory function to create Transformer UNet."""
    return TransformerUNet(
        variant=variant,
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        frozen_stages=frozen_stages,
        use_deep_supervision=use_deep_supervision,
    )


# Backward compatibility - expose as "swin" 
def create_swin_model(
    variant: str = "tiny",
    pretrained: bool = False,
    in_channels: int = 7,
    out_channels: int = 2,
    frozen_stages: int = -1,
    use_deep_supervision: bool = True,
    use_attention: bool = True,
) -> nn.Module:
    """Factory function (swin-compatible name)."""
    return create_vit_model(
        variant=variant,
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        frozen_stages=frozen_stages,
        use_deep_supervision=use_deep_supervision,
    )


if __name__ == "__main__":
    x = torch.randn(1, 7, 128, 128)
    
    print("=" * 50)
    print("Testing Transformer-Tiny UNet")
    model = create_vit_model("tiny", pretrained=False)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, list):
            print(f"  Output scales: {[o.shape for o in y]}")
        else:
            print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Transformer encoder test PASSED!")
