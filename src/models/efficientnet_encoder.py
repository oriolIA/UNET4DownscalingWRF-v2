"""
EfficientNet Encoder for WRF Downscaling

EfficientNet-based encoder with:
- EfficientNet-B0 to B7 backbone
- Pretrained ImageNet weights
- Multi-scale feature extraction
- Configurable frozen stages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet encoder for UNet.
    
    Reference: https://arxiv.org/abs/1905.11946
    """
    
    def __init__(
        self,
        variant: str = "b0",
        pretrained: bool = True,
        frozen_stages: int = -1,
        in_channels: int = 7,
    ):
        super().__init__()
        
        self.variant = variant.lower()
        self.frozen_stages = frozen_stages
        
        if in_channels != 3:
            self.input_projection = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_projection = nn.Identity()
        
        # Build EfficientNet backbone
        try:
            import torchvision
            if self.variant == "b0":
                self.backbone = torchvision.models.efficientnet_b0(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            elif self.variant == "b4":
                self.backbone = torchvision.models.efficientnet_b4(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            elif self.variant == "b2":
                self.backbone = torchvision.models.efficientnet_b2(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            else:
                self.backbone = torchvision.models.efficientnet_b0(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
        except ImportError:
            self.backbone = None
        
        if self.backbone:
            self.backbone.classifier = nn.Identity()
            self.backbone.avgpool = nn.Identity()
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        if self.frozen_stages < 0 or self.backbone is None:
            return
        
        for param in self.backbone.features[0].parameters():
            param.requires_grad = False
        
        for i in range(1, min(self.frozen_stages + 1, len(self.backbone.features))):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_projection(x)
        
        if self.backbone is None:
            return [x, x, x, x]
        
        backbone_features = self.backbone.features
        
        features = []
        prev_channels = x.shape[1]
        
        for i, layer in enumerate(backbone_features):
            x = layer(x)
            current_channels = x.shape[1]
            
            if current_channels != prev_channels:
                features.append(x)
                prev_channels = current_channels
        
        if len(features) > 4:
            features = features[-4:]
        
        while len(features) < 4:
            features.append(features[-1] if features else x)
        
        return features[:4]


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt encoder."""
    
    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        frozen_stages: int = -1,
        in_channels: int = 7,
    ):
        super().__init__()
        
        self.variant = variant.lower()
        
        if in_channels != 3:
            self.input_projection = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_projection = nn.Identity()
        
        try:
            import torchvision
            if self.variant == "tiny":
                self.backbone = torchvision.models.convnext_tiny(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            elif self.variant == "small":
                self.backbone = torchvision.models.convnext_small(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
            else:
                self.backbone = torchvision.models.convnext_tiny(
                    weights="IMAGENET1K_V1" if pretrained else None
                )
        except ImportError:
            self.backbone = None
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_projection(x)
        
        if self.backbone is None:
            return [x, x, x, x]
        
        features = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in [2, 4, 6, 8]:
                features.append(x)
        
        while len(features) < 4:
            features.append(features[-1] if features else x)
        
        return features[:4]


class UNetWithEfficientNet(nn.Module):
    """
    UNet with EfficientNet encoder.
    Simplified architecture for reliable operation.
    """
    
    def __init__(
        self,
        encoder_name: str = "b0",
        pretrained: bool = True,
        in_channels: int = 7,
        out_channels: int = 2,
        frozen_stages: int = -1,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder
        if encoder_name.startswith("convnext"):
            variant = encoder_name.replace("convnext_", "")
            self.encoder = ConvNeXtEncoder(
                variant=variant,
                pretrained=pretrained,
                frozen_stages=frozen_stages,
                in_channels=in_channels
            )
        else:
            self.encoder = EfficientNetEncoder(
                variant=encoder_name,
                pretrained=pretrained,
                frozen_stages=frozen_stages,
                in_channels=in_channels
            )
        
        # Get actual encoder channels
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 128, 128)
            encoder_features = self.encoder(dummy_input)
            self.encoder_channels = [f.shape[1] for f in encoder_features]
        
        # Build decoder with channel adapters
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoder_channels[3], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        
        # Channel adapters for skip connections
        self.skip_adapter3 = nn.Conv2d(self.encoder_channels[2], 256, 1)
        self.skip_adapter2 = nn.Conv2d(self.encoder_channels[1], 128, 1)
        self.skip_adapter1 = nn.Conv2d(self.encoder_channels[0], 64, 1)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Deep supervision
        if use_deep_supervision:
            self.deep_sup3 = nn.Conv2d(128, out_channels, 1)
            self.deep_sup2 = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x: torch.Tensor):
        # Encoder
        f1, f2, f3, f4 = self.encoder(x)
        # f1: smallest channels, largest spatial
        # f4: largest channels, smallest spatial
        
        # Bottleneck
        x = self.bottleneck(f4)
        
        # Decode with skip connections
        d4 = self.up4(x)
        
        # Adapt skip connection channels
        f3_adapted = self.skip_adapter3(f3)
        if d4.shape[2:] != f3_adapted.shape[2:]:
            f3_adapted = F.interpolate(f3_adapted, size=d4.shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.dec4(torch.cat([d4, f3_adapted], dim=1))
        
        d3 = self.up3(d4)
        f2_adapted = self.skip_adapter2(f2)
        if d3.shape[2:] != f2_adapted.shape[2:]:
            f2_adapted = F.interpolate(f2_adapted, size=d3.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.dec3(torch.cat([d3, f2_adapted], dim=1))
        
        d2 = self.up2(d3)
        f1_adapted = self.skip_adapter1(f1)
        if d2.shape[2:] != f1_adapted.shape[2:]:
            f1_adapted = F.interpolate(f1_adapted, size=d2.shape[2:], mode='bilinear', align_corners=False)
        
        d2 = self.dec2(torch.cat([d2, f1_adapted], dim=1))
        
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


def create_encoder_model(
    encoder: str = "efficientnet_b0",
    pretrained: bool = True,
    in_channels: int = 7,
    out_channels: int = 2,
    frozen_stages: int = -1,
    **kwargs
) -> nn.Module:
    """Factory function to create UNet with modern encoders."""
    return UNetWithEfficientNet(
        encoder_name=encoder,
        pretrained=pretrained,
        in_channels=in_channels,
        out_channels=out_channels,
        frozen_stages=frozen_stages,
        use_deep_supervision=kwargs.get('use_deep_supervision', True),
    )


if __name__ == "__main__":
    x = torch.randn(1, 7, 128, 128)
    
    print("=" * 50)
    print("Testing EfficientNet-B0 Encoder UNet")
    model = create_encoder_model("efficientnet_b0", pretrained=False)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, list):
            print(f"  Output scales: {[o.shape for o in y]}")
        else:
            print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
