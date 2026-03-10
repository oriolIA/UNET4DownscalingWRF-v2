#!/usr/bin/env python3
"""Quick inference script - load 1 day and compare"""

import sys
sys.path.insert(0, './src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.resunet import ResUNet
import xarray as xr

# Config
LR_FILE = '/home/oriol/data/WRF/1469893/d04.asd05/map_d04.asd05_2020-01-01.nc'
HR_FILE = '/home/oriol/data/WRF/1469893/d05/map_d05_2020-01-01.nc'
MODEL_PATH = '/home/oriol/git/UNET4DownscalingWRF-v2/outputs/20260310_100753/model_state.pth'
OUTPUT_DIR = Path('/home/oriol/git/UNET4DownscalingWRF-v2/outputs/20260310_100753/predictions')
OUTPUT_DIR.mkdir(exist_ok=True)

VARIABLES = ["U", "V", "W", "T", "P", "TKE"]

print("Loading model...")
model = ResUNet(in_channels=6, out_channels=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

print("Loading LR data...")
lr_ds = xr.open_dataset(LR_FILE)
print(f"LR vars: {list(lr_ds.data_vars)}")

# Get data
lr_vars = []
for v in VARIABLES:
    if v in lr_ds:
        lr_vars.append(lr_ds[v].values)
        print(f"  {v}: {lr_ds[v].shape}")

lr_data = np.stack(lr_vars, axis=0)
print(f"LR shape: {lr_data.shape}")

# Normalize
lr_mean = lr_data.mean(axis=(1, 2, 3), keepdims=True)
lr_std = lr_data.std(axis=(1, 2, 3), keepdims=True)
lr_norm = (lr_data - lr_mean) / (lr_std + 1e-8)

# Reshape for model (add batch dim)
# Model expects (B, C, H, W)
lr_input = torch.FloatTensor(lr_norm[:, :, 5, :, :])  # Take level 5
print(f"Input shape: {lr_input.shape}")

print("Running inference...")
with torch.no_grad():
    pred = model(lr_input)

print(f"Pred shape: {pred.shape}")

# Plot first variable
print("Generating plot...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, var_name in enumerate(VARIABLES):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(pred[idx, 0].numpy(), cmap='viridis')
    ax.set_title(f'Pred {var_name}')
    plt.colorbar(im, ax=ax)

plt.suptitle('UNET Predictions - Day 1', fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictions.png', dpi=150)
print(f"Saved {OUTPUT_DIR}/predictions.png")

# Load HR for comparison
print("Loading HR data...")
hr_ds = xr.open_dataset(HR_FILE)
hr_vars = []
for v in VARIABLES:
    if v in hr_ds:
        hr_vars.append(hr_ds[v].values)

hr_data = np.stack(hr_vars, axis=0)
print(f"HR shape: {hr_data.shape}")

# Plot comparison
fig, axes = plt.subplots(2, 6, figsize=(20, 8))

for idx, var_name in enumerate(VARIABLES):
    # LR
    axes[0, idx].imshow(lr_data[idx, 5, :, :], cmap='viridis')
    axes[0, idx].set_title(f'LR {var_name}')
    axes[0, idx].axis('off')
    
    # HR
    axes[1, idx].imshow(hr_data[idx, 5, :, :], cmap='viridis')
    axes[1, idx].set_title(f'HR {var_name}')
    axes[1, idx].axis('off')

plt.suptitle('LR vs HR Comparison', fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'lr_hr_comparison.png', dpi=150)
print(f"Saved {OUTPUT_DIR}/lr_hr_comparison.png")

print("Done!")
