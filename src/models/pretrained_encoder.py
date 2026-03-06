"""
Pretrained Encoder UNet for WRF Downscaling

UNet architecture with pretrained encoders (ResNet, EfficientNet)
for improved feature extraction and transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


class AttentionGate(nn.Module):
    """Attention gate module for feature refinement."""
    
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


class PretrainedEncoder(nn.Module):
    """Wrapper for pretrained encoders (ResNet, EfficientNet)."""
    
    ENCODER_CONFIGS = {
        "resnet18": {"model": models.resnet18, "features": [64, 128, 256, 512]},
        "resnet34": {"model": models.resnet34, "features": [64, 128, 256, 512]},
        "resnet50": {"model": models.resnet50, "features": [256, 512, 1024, 2048]},
        "efficientnet_b0": {"model": models.efficientnet_b0, "features": [24, 40, 112, 1280]},
    }
    
    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True, 
                 in_channels: int = 7, frozen_stages: int = -1):
        super().__init__()
        
        if encoder_name not in self.ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        self.encoder_name = encoder_name
        self.config = self.ENCODER_CONFIGS[encoder_name]
        
        # Load backbone
        if pretrained:
            if "resnet" in encoder_name:
                weights = getattr(models, f"ResNet{encoder_name.replace('resnet', '')}_Weights").DEFAULT if encoder_name != "resnet18" else models.ResNet18_Weights.DEFAULT
                self.backbone = self.config["model"](weights=weights)
            else:
                self.backbone = self.config["model"](weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            self.backbone = self.config["model"](weights=None)
        
        # Modify first conv for different input channels
        if in_channels != 3:
            if "resnet" in encoder_name:
                old_conv = self.backbone.conv1
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
                if pretrained:
                    with torch.no_grad():
                        old_w = old_conv.weight.clone()
                        new_w = torch.cat([old_w] * (in_channels // 3 + 1), dim=1)[:, :in_channels]
                        self.backbone.conv1.weight = nn.Parameter(new_w)
                else:
                    # Initialize randomly for non-pretrained
                    nn.init.kaiming_normal_(self.backbone.conv1.weight)
            elif "efficientnet" in encoder_name:
                old_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
                if pretrained:
                    with torch.no_grad():
                        old_w = old_conv.weight.clone()
                        new_w = torch.cat([old_w] * (in_channels // 3 + 1), dim=1)[:, :in_channels]
                        self.backbone.features[0][0].weight = nn.Parameter(new_w)
                else:
                    nn.init.kaiming_normal_(self.backbone.features[0][0].weight)
        
        # Freeze stages
        if frozen_stages >= 0:
            if "resnet" in encoder_name:
                for param in self.backbone.conv1.parameters():
                    param.requires_grad = False
            elif "efficientnet" in encoder_name:
                for param in self.backbone.features[0].parameters():
                    param.requires_grad = False
        if frozen_stages >= 1:
            if "resnet" in encoder_name:
                for param in self.backbone.layer1.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        if "resnet" in self.encoder_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            features.append(x)
            x = self.backbone.layer2(x)
            features.append(x)
            x = self.backbone.layer3(x)
            features.append(x)
            x = self.backbone.layer4(x)
            features.append(x)
        elif "efficientnet" in self.encoder_name:
            # EfficientNet: features returns [B, channels, H/32, W/32]
            x = self.backbone.features(x)
            # Create intermediate features via adaptive pooling
            # This is a simplification - for full efficiency would use hooks
            f1 = F.adaptive_avg_pool2d(x, (x.shape[2] * 4, x.shape[3] * 4))  # ~1/8
            f2 = F.adaptive_avg_pool2d(x, (x.shape[2] * 2, x.shape[3] * 2))  # ~1/16
            features = [f1, f2, x, x]  # Approximate for decoder
        return features
    
    def get_output_channels(self) -> List[int]:
        return self.config["features"]


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = AttentionGate(out_channels, skip_channels, out_channels // 2)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetPretrained(nn.Module):
    """UNet with pretrained encoder for WRF downscaling."""
    
    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True,
                 in_channels: int = 7, out_channels: int = 2, frozen_stages: int = -1,
                 use_attention: bool = True, use_deep_supervision: bool = True):
        super().__init__()
        
        # Encoder
        self.encoder = PretrainedEncoder(encoder_name, pretrained, in_channels, frozen_stages)
        encoder_channels = self.encoder.get_output_channels()  # [64, 128, 256, 512]
        
        # Bottleneck
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: 3 levels
        # encoder_channels[:-1] = [64, 128, 256], reversed = [256, 128, 64]
        decoder_channels = encoder_channels[:-1][::-1]  # [256, 128, 64]
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = bottleneck_channels if i == 0 else decoder_channels[i - 1]
            skip_ch = decoder_channels[i]
            out_ch = decoder_channels[i]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        # Final output
        self.final = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        
        # Deep supervision
        self.use_deep_supervision = use_deep_supervision
        if use_deep_supervision:
            self.deep_heads = nn.ModuleList()
            for ch in decoder_channels:
                self.deep_heads.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(ch, ch // 2, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch // 2, out_channels, 1),
                    )
                )
    
    def forward(self, x: torch.Tensor):
        # Encoder
        encoder_features = self.encoder(x)  # [64, 128, 256, 512]
        skip_features = encoder_features[::-1]  # [512, 256, 128, 64]
        
        # Bottleneck
        x = self.bottleneck(skip_features[0])  # 512 -> 512
        
        # Decoder
        deep_features = []
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_features[i + 1]
            x = decoder(x, skip)
            if self.use_deep_supervision:
                deep_features.append(x)
        
        # Final output
        output = self.final(x)
        
        if self.use_deep_supervision:
            outputs = [output]
            for i, head in enumerate(self.deep_heads):
                if i < len(deep_features):
                    outputs.append(head(deep_features[i]))
            return outputs
        
        return output


def create_model(encoder: str = "resnet18", in_channels: int = 7, 
                 out_channels: int = 2, **kwargs):
    """Create pretrained UNet model."""
    return UNetPretrained(encoder, in_channels=in_channels, 
                          out_channels=out_channels, **kwargs)


if __name__ == "__main__":
    # Test
    x = torch.randn(1, 7, 128, 128)
    
    print("=" * 50)
    print("Testing Pretrained Encoder UNet")
    print("=" * 50)
    
    for encoder in ["resnet18", "resnet34"]:
        print(f"\n{encoder}:")
        model = UNetPretrained(encoder, pretrained=False, in_channels=7, out_channels=2)
        
        with torch.no_grad():
            y = model(x)
            if isinstance(y, list):
                print(f"  Outputs: {[o.shape for o in y]}")
            else:
                print(f"  Output: {y.shape}")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
    
    # Test with pretrained ResNet18
    print("\n" + "=" * 50)
    print("Testing pretrained ResNet18 encoder")
    print("=" * 50)
    model = UNetPretrained("resnet18", pretrained=True, in_channels=7, out_channels=2)
    with torch.no_grad():
        y = model(x)
        print(f"  Output: {y[0].shape}")
    
    # Test frozen stages
    print("\n" + "=" * 50)
    print("Testing frozen encoder (first 2 stages)")
    print("=" * 50)
    model = UNetPretrained("resnet18", pretrained=False, in_channels=7, 
                          out_channels=2, frozen_stages=2)
    frozen = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"  Frozen: {frozen:,} | Trainable: {trainable:,}")
    
    print("\nDone!")
