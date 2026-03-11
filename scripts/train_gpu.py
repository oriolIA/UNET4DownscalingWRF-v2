#!/usr/bin/env python3
"""
Training script for UNET4DownscalingWRF-v2 on GPU with logging
"""
import sys
sys.path.insert(0, "src")

# Patch tensorboardX
import types
tensorboardX = types.ModuleType('tensorboardX')
tensorboardX.SummaryWriter = types.SimpleNamespace()
sys.modules['tensorboardX'] = tensorboardX

from models.resunet import ResUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class WRFDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, start_idx=0, num_samples=180):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.variables = ["U", "V", "W", "T", "P", "TKE"]
        
        lr_files = sorted(self.lr_dir.glob("*.nc"))[start_idx:start_idx+num_samples]
        hr_files = sorted(self.hr_dir.glob("*.nc"))[start_idx:start_idx+num_samples]
        
        self.lr_files = lr_files
        self.hr_files = hr_files
        print(f"Loaded {len(self.lr_files)} samples (start={start_idx})")
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_ds = xr.open_dataset(str(self.lr_files[idx]))
        hr_ds = xr.open_dataset(str(self.hr_files[idx]))
        
        lr_data, hr_data = [], []
        for v in self.variables:
            if v in lr_ds:
                arr = lr_ds[v].values
                if arr.ndim > 2:
                    arr = arr.mean(axis=(0, 1))
                lr_data.append(arr.astype(np.float32))
            if v in hr_ds:
                arr = hr_ds[v].values
                if arr.ndim > 2:
                    arr = arr.mean(axis=(0, 1))
                hr_data.append(arr.astype(np.float32))
        
        lr = np.stack(lr_data, axis=0)
        hr = np.stack(hr_data, axis=0)
        
        lr = (lr - lr.mean()) / (lr.std() + 1e-8)
        hr = (hr - hr.mean()) / (hr.std() + 1e-8)
        
        return torch.from_numpy(lr), torch.from_numpy(hr)

def plot_samples(model, dataset, device, output_path, n_samples=5):
    """Plot LR, prediction, HR samples"""
    model.eval()
    fig, axes = plt.subplots(n_samples, 6, figsize=(20, n_samples*3))
    
    for i in range(n_samples):
        lr, hr = dataset[i]
        lr = lr.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(lr)
            if isinstance(pred, list):
                pred = pred[-1]
            pred_up = F.interpolate(pred, size=hr.shape[1:], mode='bilinear', align_corners=False)
        
        lr_np = lr[0, 0].cpu().numpy()
        pred_np = pred_up[0, 0].cpu().numpy()
        hr_np = hr[0].numpy()
        
        variables = ["U", "V", "W", "T", "P", "TKE"]
        
        for j, var in enumerate(variables):
            if j < 3:  # Show first 3 vars
                axes[i, j].imshow(lr_np)
                axes[i, j].set_title(f'LR {var}')
                axes[i, j].axis('off')
        
        # Prediction vs HR
        axes[i, 3].imshow(pred_np)
        axes[i, 3].set_title('Pred')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(hr_np)
        axes[i, 4].set_title('HR')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(np.abs(pred_np - hr_np))
        axes[i, 5].set_title('Diff')
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Samples saved: {output_path}")

def train(start_idx, num_samples, epochs, output_name):
    print("=" * 60)
    print(f"TRAINING: start={start_idx}, samples={num_samples}, epochs={epochs}")
    print("=" * 60)
    
    device = torch.device('cuda')
    print(f"Device: {device}")
    
    model = ResUNet(in_channels=6, out_channels=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    lr_dir = "/storage/oriol/data/ratioAI.runs/runs/1469893/d04.asd05"
    hr_dir = "/storage/oriol/data/ratioAI.runs/runs/1469893/d05"
    
    dataset = WRFDataset(lr_dir, hr_dir, start_idx, num_samples)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Track losses
    losses = []
    
    print(f"Training with {len(dataset)} samples...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (lr, hr) in enumerate(loader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            out = model(lr)
            if isinstance(out, list):
                out = out[-1]
            
            out_up = F.interpolate(out, size=hr.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(out_up, hr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title(f'Loss Curve (Epoch {epoch+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f'/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/loss_curve.png', dpi=150)
            plt.close()
            
            # Save sample predictions
            try:
                plot_samples(model, dataset, device, 
                           f'/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/samples_epoch{epoch+1}.png')
            except:
                pass
    
    # Final plots
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Final Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/loss_curve.png', dpi=150)
    plt.close()
    
    # Final sample predictions
    try:
        plot_samples(model, dataset, device,
                    '/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/final_samples.png')
    except:
        pass
    
    output_path = f'/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/{output_name}'
    torch.save(model.state_dict(), output_path)
    print(f"Model saved: {output_path}")
    
    # Save losses to file
    with open('/home/oriol/R2-claw/UNET4DownscalingWRF-v2/outputs/losses.txt', 'w') as f:
        for i, l in enumerate(losses):
            f.write(f"Epoch {i+1}: {l:.6f}\n")
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--samples', type=int, default=180)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--output', type=str, default='model.pth')
    args = parser.parse_args()
    
    train(args.start, args.samples, args.epochs, args.output)
