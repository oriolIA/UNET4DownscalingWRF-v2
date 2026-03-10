"""
FPN (Feature Pyramid Network) Decoder for WRF Downscaling

FPN decoder adds top-down pathway with lateral connections for better
multi-scale feature fusion. Combined with attention gates for improved
feature selection.

Reference: https://arxiv.org/abs/1612.03144
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FPNBlock(nn.Module):
    """Single FPN block with lateral connection and top-down pathway."""
    
    def __init__(self, lateral_channels: int, feature_channels: int, output_channels: int):
        super().__init__()
        
        # Feature processing: 3x3 conv to reduce artifacts from upsampling
        self.feature_conv = nn.Sequential(
            nn.Conv2d(feature_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # For lateral feature that comes already processed
        self.lateral_bn = nn.BatchNorm2d(output_channels)
    
    def forward(self, lateral: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lateral: Already processed lateral features (from lateral conv)
            feature: Features from top-down pathway (to be upsampled)
        """
        # Upsample feature to match lateral spatial dimensions
        if feature.shape[2:] != lateral.shape[2:]:
            feature = F.interpolate(feature, size=lateral.shape[2:], mode='bilinear', align_corners=False)
        
        # Add lateral and feature (both should now have same channels)
        out = lateral + feature
        out = self.feature_conv(out)
        
        return out


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network Decoder with attention gates.
    
    Architecture:
    - Bottom-up pathway: Encoder features (different scales)
    - Top-down pathway: FPN feature fusion
    - Lateral connections: Channel matching
    - Attention gates: Feature selection
    """
    
    def __init__(
        self,
        encoder_features: List[int],  # [64, 128, 256, 512]
        decoder_features: int = 256,
        use_attention: bool = True,
        attention_type: str = "se"  # "se", "cbam", "none"
    ):
        super().__init__()
        
        self.encoder_features = encoder_features
        self.use_attention = use_attention
        self.decoder_features = decoder_features
        
        # Build lateral connections (1x1 convs to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, decoder_features, 1)
            for ch in encoder_features
        ])
        
        # Build FPN blocks (top-down pathway)
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(encoder_features[i], decoder_features, decoder_features)
            for i in range(len(encoder_features))
        ])
        
        # Attention gates for skip connections
        if use_attention:
            from .resunet import AttentionGate
            self.attention_gates = nn.ModuleList([
                AttentionGate(decoder_features, decoder_features, decoder_features // 2)
                for _ in encoder_features
            ])
        
        # Output heads for each scale (for deep supervision)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_features, decoder_features // 2, 3, padding=1),
                nn.BatchNorm2d(decoder_features // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_features // 2, 2, 1)  # 2 output channels
            )
            for _ in encoder_features
        ])
    
    def forward(
        self, 
        encoder_skips: List[torch.Tensor],
        bottleneck: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Args:
            encoder_skips: List of encoder features [P2, P3, P4, P5] (small to large)
            bottleneck: Bottleneck features
            
        Returns:
            List of multi-scale outputs [P2_out, P3_out, P4_out, P5_out]
        """
        # Reverse encoder features (largest to smallest for top-down)
        encoder_skips = encoder_skips[::-1]  # [P5, P4, P3, P2]
        
        # Build top-down pathway
        # Start with bottleneck
        laterals = []
        
        # Map encoder features (reversed) to FPN features
        # encoder_skips[0] = P5 (512 ch), encoder_skips[1] = P4 (256 ch), etc.
        # We need to match: P5->lateral[3], P4->lateral[2], P3->lateral[1], P2->lateral[0]
        for i, enc_feat in enumerate(encoder_skips):
            # Get corresponding lateral conv (reverse index)
            lateral_idx = len(encoder_skips) - 1 - i
            laterals.append(self.lateral_convs[lateral_idx](enc_feat))
        
        # Top-down pathway with FPN blocks
        fpn_features = []
        
        # First, process bottleneck to match FPN channel dimension
        bottleneck_h, bottleneck_w = bottleneck.shape[2], bottleneck.shape[3]
        x = nn.Conv2d(bottleneck.shape[1], self.decoder_features, 1).to(bottleneck.device)(bottleneck)
        x = nn.BatchNorm2d(self.decoder_features).to(bottleneck.device)(x)
        x = nn.ReLU(inplace=True)(x)
        
        for i, fpn_block in enumerate(self.fpn_blocks):
            # Use corresponding lateral feature
            lateral_feat = laterals[i]
            x = fpn_block(lateral_feat, x)
            fpn_features.append(x)
        
        # fpn_features is now [P5_out, P4_out, P3_out, P2_out] (large to small)
        # Reverse to get [P2_out, P3_out, P4_out, P5_out] (small to large)
        fpn_features = fpn_features[::-1]
        
        return fpn_features


class ResUNetFPN(nn.Module):
    """
    ResUNet with FPN Decoder for WRF Downscaling.
    
    Combines:
    - ResNet-style encoder with SE/CBAM attention
    - FPN top-down pathway for multi-scale fusion
    - Attention-gated skip connections
    - Multi-scale deep supervision
    
    Reference: https://arxiv.org/abs/1612.03144
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        features: List[int] = [64, 128, 256, 512],
        attention: str = "se",  # "se", "cbam", "none"
        use_aspp: bool = True,
        aspp_rates: List[int] = [6, 12, 18],
        fpn_features: int = 256,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_deep_supervision = use_deep_supervision
        self.fpn_features = fpn_features
        
        # Import ResUNet components
        from .resunet import (
            EncoderBlock, ASPP, SEResidualBlock, CBAMResidualBlock, 
            ResidualBlock, AttentionGate
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for feature in features:
            self.encoders.append(EncoderBlock(in_ch, feature, attention=attention))
            in_ch = feature
        
        # Bottleneck with optional ASPP
        if use_aspp:
            self.aspp = ASPP(features[-1], features[-1] * 2, aspp_rates)
            self.bottleneck_refine = self._make_bottleneck_block(features[-1] * 2, features[-1] * 2, attention)
        else:
            self.aspp = None
            self.bottleneck = self._make_bottleneck_block(features[-1], features[-1] * 2, attention)
        
        # FPN Decoder
        self.fpn_decoder = FPNDecoder(
            encoder_features=features,
            decoder_features=fpn_features,
            use_attention=True,
            attention_type=attention
        )
        
        # Final output conv (upsample all scales to target resolution)
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_features, fpn_features // 2, 3, padding=1),
            nn.BatchNorm2d(fpn_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_features // 2, out_channels, 1)
        )
        
        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fpn_features, fpn_features // 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fpn_features // 2, out_channels, 1)
                )
                for _ in features
            ])
    
    def _make_bottleneck_block(self, in_ch: int, out_ch: int, attention: str):
        from .resunet import CBAMResidualBlock, SEResidualBlock, ResidualBlock
        
        if attention == "cbam":
            return CBAMResidualBlock(in_ch, out_ch)
        elif attention == "se":
            return SEResidualBlock(in_ch, out_ch)
        else:
            return ResidualBlock(in_ch, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        
        for encoder in self.encoders:
            x, feat = encoder(x)
            skip_connections.append(feat)
        
        # Bottleneck with optional ASPP
        if self.aspp is not None:
            x = self.aspp(x)
            x = self.bottleneck_refine(x)
        else:
            x = self.bottleneck(x)
        
        # FPN Decoder
        fpn_features = self.fpn_decoder(skip_connections, x)
        
        # fpn_features = [P2, P3, P4, P5] (small to large scale)
        # Use largest scale (P5) for final output
        output = self.final_conv(fpn_features[-1])
        
        # Deep supervision
        if self.use_deep_supervision:
            outputs = [output]
            for i, head in enumerate(self.deep_supervision_heads):
                # Upsample each scale to target resolution
                scale_out = head(fpn_features[i])
                scale_out = F.interpolate(scale_out, size=output.shape[2:], mode='bilinear', align_corners=False)
                outputs.append(scale_out)
            return outputs
        
        return output


class ResUNetFPNSmall(nn.Module):
    """Smaller ResUNet-FPN for faster training."""
    
    def __init__(self, in_channels: int = 7, out_channels: int = 2):
        super().__init__()
        features = [32, 64, 128, 256]
        self.model = ResUNetFPN(in_channels, out_channels, features, fpn_features=128)


def create_fpn_model(
    model_name: str = "resunet_fpn",
    in_channels: int = 7,
    out_channels: int = 2,
    **kwargs
):
    """Create FPN-based model."""
    
    models = {
        "resunet_fpn": lambda: ResUNetFPN(
            in_channels, out_channels,
            features=[64, 128, 256, 512],
            fpn_features=kwargs.get("fpn_features", 256),
            attention=kwargs.get("attention", "se"),
            use_aspp=kwargs.get("use_aspp", True),
            use_deep_supervision=kwargs.get("use_deep_supervision", True)
        ),
        "resunet_fpn_small": ResUNetFPNSmall,
        "resunet_fpn_cbam": lambda: ResUNetFPN(
            in_channels, out_channels,
            features=[64, 128, 256, 512],
            fpn_features=kwargs.get("fpn_features", 256),
            attention="cbam",
            use_aspp=kwargs.get("use_aspp", True),
            use_deep_supervision=kwargs.get("use_deep_supervision", True)
        ),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


if __name__ == "__main__":
    # Test FPN model
    import torch
    
    print("=" * 50)
    print("Testing ResUNet-FPN")
    x = torch.randn(1, 7, 128, 128)
    
    model = ResUNetFPN(in_channels=7, out_channels=2)
    with torch.no_grad():
        y = model(x)
        if isinstance(y, list):
            print(f"  Multi-scale outputs: {[o.shape for o in y]}")
        else:
            print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("=" * 50)
    print("Testing ResUNet-FPN (CBAM)")
    model = ResUNetFPN(in_channels=7, out_channels=2, attention="cbam")
    with torch.no_grad():
        y = model(x)
    print(f"  Output: {y.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
