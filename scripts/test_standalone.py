#!/usr/bin/env python3
"""
UNET4DownscalingWRF-v2 - Standalone Test Script
=================================================
Test the model architecture and metrics without package installation.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_dir / 'src'))

print("=" * 60)
print("UNET4DownscalingWRF-v2 - Standalone Test")
print("=" * 60)

# Import blocks first
from models.blocks import ResidualBlock, PreActivationResBlock
from config.config import (
    ModelConfig, ResUNetConfig, UpsampleMode, BottleneckType
)
from models.unet import SimpleUNet, ResUNet, ResUNetV2, UNetFactory

print("\n✓ Imports funcionen")

# Test configurations
configs = [
    ("SimpleUNet", lambda: SimpleUNet(ModelConfig(in_channels=7, out_channels=2, n_filters=32))),
    ("ResUNet", lambda: ResUNet(ResUNetConfig(in_channels=7, out_channels=2, n_filters=32))),
]

results = {}

for name, model_fn in configs:
    print(f"\n{'='*40}")
    print(f"Testing: {name}")
    print("="*40)
    
    model = model_fn()
    x = torch.randn(1, 7, 50, 51)
    
    with torch.no_grad():
        y = model(x)
    
    params = sum(p.numel() for p in model.parameters())
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {params:,}")
    
    results[name] = {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "params": params
    }

# Test metrics
from utils.metrics import all_metrics

print(f"\n{'='*60}")
print("Metrics Test")
print("="*60)

y_pred = torch.randn(1, 2, 100, 102)
y_true = torch.randn(1, 2, 100, 102)
m = all_metrics(y_pred, y_true)

print(f"  MSE:  {m['mse']:.6f}")
print(f"  MAE:  {m['mae']:.6f}")
print(f"  PSNR: {m['psnr']:.2f} dB")
print(f"  SSIM: {m['ssim']:.4f}")

# Save results
import json
output_dir = repo_dir / "outputs" / "test"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "standalone_test.json", 'w') as f:
    json.dump({
        "timestamp": str(datetime.now()),
        "models": results,
        "metrics_sample": m
    }, f, indent=2)

print(f"\n{'='*60}")
print("✅ TOTS ELS TESTS PASSATS!")
print("="*60)
