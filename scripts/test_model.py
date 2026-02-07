#!/usr/bin/env python3
"""
UNET4DownscalingWRF-v2 - Test Script
=====================================
Usage: python3 scripts/test_model.py
"""

import sys
from pathlib import Path

# Setup paths for imports
REPO_DIR = Path(__file__).parent.parent
SRC_DIR = REPO_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import torch
from models.unet import SimpleUNet, ResUNet, ResUNetV2, UNetFactory
from config.config import ModelConfig, ResUNetConfig, UpsampleMode, NormType

print("=" * 60)
print("UNET4DownscalingWRF-v2 - Model Test Suite")
print("=" * 60)

# Test 1: SimpleUNet
print("\n[1/3] Testing SimpleUNet...")
config = ModelConfig(
    in_channels=7,
    out_channels=2,
    n_filters=32,
    upsampling=UpsampleMode.CONV_TRANSPOSE,
)
model = SimpleUNet(config)
x = torch.randn(1, 7, 50, 51)
with torch.no_grad():
    y = model(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {y.shape}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Test 2: ResUNet
print("\n[2/3] Testing ResUNet...")
res_config = ResUNetConfig(
    in_channels=7,
    out_channels=1,
    n_filters=64,
    use_attention=True,
    norm_type=NormType.BATCH,
)
res_model = ResUNet(res_config)
x2 = torch.randn(1, 7, 100, 102)
with torch.no_grad():
    y2 = res_model(x2)
print(f"  Input:  {x2.shape}")
print(f"  Output: {y2.shape}")
print(f"  Params: {sum(p.numel() for p in res_model.parameters()):,}")

# Test 3: Metrics
print("\n[3/3] Testing metrics...")
from utils.metrics import all_metrics
m = all_metrics(y, y)
print(f"  MSE:   {m['mse']:.6f}")
print(f"  MAE:   {m['mae']:.6f}")
print(f"  PSNR:  {m['psnr']:.2f} dB")
print(f"  SSIM:  {m['ssim']:.4f}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
