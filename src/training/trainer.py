"""
Training script for UNET4DownscalingWRF-v2
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resunet import ResUNet
from src.utils.metrics import compute_all_metrics


class WRF Dataset(Dataset):
    """Dataset for WRF downscaling."""
    
    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]
    
    def __init__(self, lr_dir: str, hr_dir: str, split: str = "train"):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        
        # Get files
        lr_files = sorted(self.lr_dir.glob("*.nc"))
        hr_files = sorted(self.hr_dir.glob("*.nc"))
        
        # Simple 80/10/10 split
        n = len(lr_files)
        if split == "train":
            files = lr_files[:int(n*0.8)]
            hr_files = hr_files[:int(n*0.8)]
        elif split == "val":
            files = lr_files[int(n*0.8):int(n*0.9)]
            hr_files = hr_files[int(n*0.8):int(n*0.9)]
        else:  # test
            files = lr_files[int(n*0.9):]
            hr_files = hr_files[int(n*0.9):]
        
        self.files = list(zip(files, hr_files))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        lr_path, hr_path = self.files[idx]
        
        # Load LR
        with xr.open_dataset(lr_path) as ds:
            lr = np.stack([
                ds[var].values.mean(axis=0) if len(ds[var].shape) == 3 
                else ds[var].values
                for var in self.VARIABLES
            ], axis=0).astype(np.float32)
        
        # Load HR
        with xr.open_dataset(hr_path) as ds:
            hr = np.stack([
                ds["U"].values.mean(axis=0),
                ds["V"].values.mean(axis=0)
            ], axis=0).astype(np.float32)
        
        # Normalize
        lr = lr / 100.0  # Scale
        hr = hr / 100.0
        
        return torch.from_numpy(lr), torch.from_numpy(hr)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for lr, hr in tqdm(loader, desc="Training"):
        lr = lr.to(device)
        hr = hr.to(device)
        
        optimizer.zero_grad()
        pred = model(lr)
        loss = criterion(pred, hr)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Validating"):
            lr = lr.to(device)
            hr = hr.to(device)
            
            pred = model(lr)
            loss = criterion(pred, hr)
            total_loss += loss.item()
            
            # Metrics
            metrics = compute_all_metrics(pred, hr)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) 
        for k in all_metrics[0].keys()
    }
    
    return total_loss / len(loader), avg_metrics


def main():
    parser = argparse.ArgumentParser(description="UNET4Downscaling Training")
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="resunet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(output_dir / "logs")
    
    # Data
    train_ds = WRF Dataset(args.lr_dir, args.hr_dir, "train")
    val_ds = WRF Dataset(args.lr_dir, args.hr_dir, "val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    if args.model == "resunet":
        model = ResUNet(in_channels=7, out_channels=2)
    else:
        model = ResUNet(in_channels=7, out_channels=2)
    
    model = model.to(device)
    
    # Training
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Train loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.4f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  -> Saved best model")
        
        scheduler.step(val_loss)
    
    writer.close()
    print(f"\nTraining complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
