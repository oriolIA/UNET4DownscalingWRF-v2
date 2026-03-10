#!/usr/bin/env python3
"""Inference script for UNET model - Generate predictions and compare with HR"""

import sys
sys.path.insert(0, './src')

import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.resunet import ResUNet

# Config
LR_DIR = Path('/home/oriol/data/WRF/1469893/d04.asd05')
HR_DIR = Path('/home/oriol/data/WRF/1469893/d05')
MODEL_PATH = Path('/home/oriol/git/UNET4DownscalingWRF-v2/outputs/20260310_100753/model_state.pth')
OUTPUT_DIR = Path('/home/oriol/git/UNET4DownscalingWRF-v2/outputs/20260310_100753/predictions')
OUTPUT_DIR.mkdir(exist_ok=True)

# Variables
VARIABLES = ["U", "V", "W", "T", "P", "TKE"]

def load_sample(lr_dir, hr_dir, day_idx=0, n_days=10):
    """Load sample LR and HR data"""
    lr_files = sorted(lr_dir.glob('*.nc'))[day_idx:day_idx+n_days]
    hr_files = sorted(hr_dir.glob('*.nc'))[day_idx:day_idx+n_days]
    
    print(f"Loading {len(lr_files)} LR files...")
    print(f"Loading {len(hr_files)} HR files...")
    
    lr_data = []
    hr_data = []
    
    for lf, hf in zip(lr_files, hr_files):
        lr_ds = xr.open_dataset(lf)
        hr_ds = xr.open_dataset(hf)
        
        # Extract variables
        lr_vars = []
        hr_vars = []
        for v in VARIABLES:
            if v in lr_ds:
                lr_vars.append(lr_ds[v].values)
            if v in hr_ds:
                hr_vars.append(hr_ds[v].values)
        
        lr_data.append(np.stack(lr_vars, axis=0))
        hr_data.append(np.stack(hr_vars, axis=0))
    
    return np.array(lr_data), np.array(hr_data)

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

def denormalize(data, mean, std):
    return data * (std + 1e-8) + mean

def main():
    print("Loading model...")
    model = ResUNet(in_channels=6, out_channels=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    print("Loading data...")
    lr_data, hr_data = load_sample(LR_DIR, HR_DIR, day_idx=0, n_days=10)
    
    print(f"LR shape: {lr_data.shape}")
    print(f"HR shape: {hr_data.shape}")
    
    # Compute mean/std from training (approximate)
    lr_mean = lr_data.mean(axis=(0, 2, 3), keepdims=True)
    lr_std = lr_data.std(axis=(0, 2, 3), keepdims=True)
    hr_mean = hr_data.mean(axis=(0, 2, 3), keepdims=True)
    hr_std = hr_data.std(axis=(0, 2, 3), keepdims=True)
    
    # Normalize
    lr_norm = normalize(lr_data, lr_mean, lr_std)
    
    # Inference
    print("Running inference...")
    predictions = []
    batch_size = 4
    
    with torch.no_grad():
        for i in range(0, len(lr_norm), batch_size):
            batch = torch.FloatTensor(lr_norm[i:i+batch_size])
            pred = model(batch)
            predictions.append(pred.numpy())
    
    pred = np.concatenate(predictions, axis=0)
    
    # Denormalize to HR scale
    pred_denorm = denormalize(pred, hr_mean, hr_std)
    
    print(f"Predictions shape: {pred_denorm.shape}")
    
    # Generate plots for each variable
    print("Generating plots...")
    
    for var_idx, var_name in enumerate(VARIABLES):
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f'{var_name} - Pred vs HR (10 days)', fontsize=16)
        
        for day_idx in range(min(10, pred_denorm.shape[0])):
            row = day_idx // 5
            col = day_idx % 5
            
            # Take middle level/time
            pred_slice = pred_denorm[day_idx, var_idx, 5, :, :]
            hr_slice = hr_data[day_idx, var_idx, 5, :, :]
            
            # Plot
            vmin = min(pred_slice.min(), hr_slice.min())
            vmax = max(pred_slice.max(), hr_slice.max())
            
            axes[row, col].imshow(pred_slice, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row, col].set_title(f'Day {day_idx+1}')
            axes[row, col].axis('off')
        
        # Last row: difference
        for day_idx in range(min(5, pred_denorm.shape[0])):
            pred_slice = pred_denorm[day_idx, var_idx, 5, :, :]
            hr_slice = hr_data[day_idx, var_idx, 5, :, :]
            diff = np.abs(pred_slice - hr_slice)
            
            axes[2, day_idx].imshow(diff, cmap='Reds')
            axes[2, day_idx].set_title(f'Diff Day {day_idx+1}')
            axes[2, day_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{var_name}_comparison.png', dpi=150)
        print(f"Saved {var_name}_comparison.png")
        plt.close()
    
    # Overall comparison
    print("Computing metrics...")
    mse = ((pred_denorm - hr_data) ** 2).mean()
    mae = np.abs(pred_denorm - hr_data).mean()
    
    print(f"Overall MSE: {mse:.6f}")
    print(f"Overall MAE: {mae:.6f}")
    
    # Save metrics
    with open(OUTPUT_DIR / 'metrics.txt', 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        for var_idx, var_name in enumerate(VARIABLES):
            var_mse = ((pred_denorm[:, var_idx] - hr_data[:, var_idx]) ** 2).mean()
            var_mae = np.abs(pred_denorm[:, var_idx] - hr_data[:, var_idx]).mean()
            f.write(f"{var_name} - MSE: {var_mse:.6f}, MAE: {var_mae:.6f}\n")
    
    print(f"\nDone! Results in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
