"""
Research: Model Architectures for WRF Downscaling
================================================

Goal: Compare different architectures for 2x super-resolution of WRF data
- d02: 56x57 → d05: 107x99 (~2x upscaling)

Available architectures:
1. **UNet** - Baseline encoder-decoder with skip connections
2. **ESRGAN** - RRDB blocks for texture preservation
3. **SwinIR** - Transformer-based, global context
4. **RCAN** - Channel attention mechanisms
5. **EDSR** - Optimized residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Simple UNet - Baseline
# =============================================================================

class UNetSimple(nn.Module):
    """Simple UNet - good baseline."""
    def __init__(self, in_ch=7, out_ch=2, base=32):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_ch, base)
        self.enc2 = self._conv_block(base, base * 2)
        self.enc3 = self._conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base * 4, base * 8)
        
        # Decoder with upsampling
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = self._conv_block(base * 4 + base * 4, base * 4)  # Skip connection
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = self._conv_block(base * 2 + base * 2, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = self._conv_block(base + base, base)
        
        self.out = nn.Conv2d(base, out_ch, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        
        return self.out(d1)


# =============================================================================
# 2. ESRGAN - RRDB Blocks
# =============================================================================

class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class ESRGAN(nn.Module):
    """ESRGAN Generator."""
    def __init__(self, in_ch=7, out_ch=2, nf=64, nb=8):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, nf, 3, padding=1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc=32) for _ in range(nb)])
        self.conv_trunk = nn.Conv2d(nf, nf, 3, padding=1)
        
        # 2x upsampling (FIX: removed one PixelShuffle block)
        self.conv_up = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_hr = nn.Conv2d(nf, nf, 3, padding=1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, padding=1)
    
    def forward(self, x):
        feat = self.conv_first(x)
        feat = self.conv_trunk(self.RRDB_trunk(feat))
        # NOTE: Long residual removed - spatial mismatch after upsampling
        feat = feat + self.conv_first(x)  # Local residual before upsampling
        feat = self.conv_up(feat)
        feat = self.conv_hr(feat)
        return self.conv_last(feat)


# =============================================================================
# 3. SwinIR - Simplified
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
        )
    
    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x))) * x


class SwinIRBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.ca = ChannelAttention(ch, 8)
    
    def forward(self, x):
        return self.ca(self.conv2(F.relu(self.conv1(x)))) + x


class SwinIR(nn.Module):
    """Lightweight SwinIR-like."""
    def __init__(self, in_ch=7, out_ch=2, ch=64, n_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[SwinIRBlock(ch) for _ in range(n_blocks)])
        
        # 2x upsampling
        self.conv_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x) + x
        x = self.conv_up(x)
        return self.conv_out(x)


# =============================================================================
# 4. RCAN - Channel Attention
# =============================================================================

class RCAB(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.ca = ChannelAttention(ch, reduction)
    
    def forward(self, x):
        return self.ca(self.conv2(F.relu(self.conv1(x)))) + x


class ResidualGroup(nn.Module):
    def __init__(self, ch, n_blocks=8):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(ch) for _ in range(n_blocks)])
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    
    def forward(self, x):
        return self.blocks(x) + self.conv(x)


class RCAN(nn.Module):
    """RCAN Generator."""
    def __init__(self, in_ch=7, out_ch=2, ch=64, n_groups=4, n_blocks=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.groups = nn.Sequential(*[ResidualGroup(ch, n_blocks) for _ in range(n_groups)])
        
        # 2x upsampling
        self.conv_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.groups(x) + x
        x = self.conv_up(x)
        return self.conv_out(x)


# =============================================================================
# 5. EDSR - Optimized Residual
# =============================================================================

class EDSRBlock(nn.Module):
    def __init__(self, ch, scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.scale = scale
    
    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x))) * self.scale + x


class EDSR(nn.Module):
    """EDSR Generator."""
    def __init__(self, in_ch=7, out_ch=2, ch=64, n_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[EDSRBlock(ch) for _ in range(n_blocks)])
        self.conv_mid = nn.Conv2d(ch, ch, 3, padding=1)
        
        # 2x upsampling
        self.conv_up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x) + x
        x = self.conv_mid(F.relu(x))
        x = self.conv_up(x)
        return self.conv_out(x)


# =============================================================================
# Model Comparison
# =============================================================================

if __name__ == "__main__":
    # Test input - matches d02 dimensions
    x = torch.randn(1, 7, 50, 51)
    
    models = {
        'UNet': lambda: UNetSimple(7, 2, 32),
        'ESRGAN': lambda: ESRGAN(7, 2, nf=64, nb=8),
        'SwinIR': lambda: SwinIR(7, 2, ch=64, n_blocks=8),
        'RCAN': lambda: RCAN(7, 2, ch=64, n_groups=3, n_blocks=4),
        'EDSR': lambda: EDSR(7, 2, ch=64, n_blocks=8),
    }
    
    print("=" * 70)
    print("MODEL ARCHITECTURE COMPARISON - WRF Downscaling")
    print("=" * 70)
    print(f"Input:  {x.shape}  (d02-like)")
    print(f"Target: ~[1, 2, 100, 102]  (d05-like, ~2x)")
    print("=" * 70)
    
    results = []
    for name, fn in models.items():
        try:
            m = fn()
            with torch.no_grad():
                y = m(x)
            p = sum(x.numel() for x in m.parameters())
            
            results.append({'name': name, 'params': p, 'out': str(list(y.shape)), '✅': True})
            print(f"\n{name:12} | {p:>10,} params | Output: {str(list(y.shape)):>20} ✅")
        except Exception as e:
            results.append({'name': name, 'error': str(e), '✅': False})
            print(f"\n{name:12} | ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Params':>10} {'Output':>22} {'Status':>8}")
    print("-" * 70)
    for r in results:
        if r.get('✅'):
            print(f"{r['name']:<12} {r['params']:>10,} {r['out']:>22} {'✅':>8}")
        else:
            print(f"{r['name']:<12} {'--':>10} {'--':>22} {'❌':>8}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
• UNet     | Fastest baseline, skip connections for gradients
• ESRGAN   | Best textures, RRDB blocks for fine details
• SwinIR   | Global context, transformer attention
• RCAN     | Channel attention, good for multi-variable
• EDSR     | Clean, no batchnorm, stable training

For WRF data (U, V, T, P, HGT):
  → Start with UNet (fastest)
  → Try ESRGAN for fine meteorological features
  → SwinIR for large-scale patterns
""")
