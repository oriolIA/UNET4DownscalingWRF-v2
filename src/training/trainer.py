"""
Training script for UNET4DownscalingWRF-v2
With Mixed Precision (AMP), Gradient Accumulation, and Advanced Loss Functions
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import xarray as xr
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resunet import ResUNet
from src.models.fpn_decoder import ResUNetFPN
from src.utils.metrics import compute_all_metrics
from src.utils.tta import TestTimeAugmentation


class SSIMLoss(nn.Module):
    """SSIM Loss for better perceptual quality."""
    
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float = 1.5):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int):
        # Create 1D window
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        # Create 2D window
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle deep supervision (list of outputs)
        if isinstance(pred, list):
            # Average SSIM across all outputs (upsampled to match target)
            ssim_vals = []
            for p in pred:
                # Upsample to match target size
                p_up = F.interpolate(p, size=target.shape[2:], mode='bilinear', align_corners=False)
                ssim_vals.append(1 - self._ssim(p_up[:, :1], target[:, :1]))
            return torch.stack(ssim_vals).mean()
        
        return 1 - self._ssim(pred[:, :1], target[:, :1])


class CombinedLoss(nn.Module):
    """Combined MAE + SSIM Loss."""
    
    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha  # Weight for MAE
        self.mae = nn.L1Loss()
        self.ssim = SSIMLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle deep supervision (list of outputs)
        if isinstance(pred, list):
            # Average loss across all outputs
            mae_vals = []
            ssim_vals = []
            for p in pred:
                p_up = F.interpolate(p, size=target.shape[2:], mode='bilinear', align_corners=False)
                mae_vals.append(self.mae(p_up, target))
                ssim_vals.append(self.ssim(p_up, target))
            mae_loss = torch.stack(mae_vals).mean()
            ssim_loss = torch.stack(ssim_vals).mean()
        else:
            mae_loss = self.mae(pred, target)
            ssim_loss = self.ssim(pred, target)
        
        return self.alpha * mae_loss + (1 - self.alpha) * ssim_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using feature differences (simplified)."""
    
    def __init__(self):
        super().__init__()
        # Use simple feature-based loss
        self.l1 = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Multi-scale L1 loss
        loss = 0.0
        
        if isinstance(pred, list):
            # For deep supervision: weight earlier scales more, upsample to match target
            for i, p in enumerate(pred):
                scale_weight = 2 ** (len(pred) - i - 1)
                p_up = F.interpolate(p, size=target.shape[2:], mode='bilinear', align_corners=False)
                loss += scale_weight * self.l1(p_up, target)
            return loss
        else:
            return self.l1(pred, target)


class WRFDataset(Dataset):
    """Dataset for WRF downscaling."""
    
    VARIABLES = ["U", "V", "W", "T", "P", "TKE"]
    
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
        
        # Load LR - mean over time and levels to get 2D spatial
        with xr.open_dataset(lr_path) as ds:
            lr = np.stack([
                ds[var].values.mean(axis=(0,1)) for var in self.VARIABLES
            ], axis=0).astype(np.float32)
        
        # Load HR
        with xr.open_dataset(hr_path) as ds:
            hr = np.stack([
                ds[var].values.mean(axis=(0,1)) for var in self.VARIABLES
            ], axis=0).astype(np.float32)
        
        # Normalize
        lr = lr / 100.0  # Scale
        hr = hr / 100.0
        
        return torch.from_numpy(lr), torch.from_numpy(hr)


def train_epoch(model, loader, optimizer, criterion, device, use_amp: bool = False, 
                scaler: GradScaler = None, gradient_accumulation_steps: int = 1,
                gradient_clip: float = None):
    """Train one epoch with Mixed Precision and Gradient Accumulation."""
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (lr, hr) in enumerate(tqdm(loader, desc="Training")):
        lr = lr.to(device)
        hr = hr.to(device)
        
        if use_amp and scaler is not None:
            # Mixed precision forward pass
            with autocast():
                pred = model(lr)
                if isinstance(pred, list): pred = pred[0]
                
                # Upsample prediction to match target resolution
                if pred.shape[2:] != hr.shape[2:]:
                    pred = F.interpolate(pred, size=hr.shape[2:], mode='bilinear', align_corners=False)
                
                loss = criterion(pred, hr)
                loss = loss / gradient_accumulation_steps
            
            # Scaled backward
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if gradient_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard forward pass
            pred = model(lr)
            if isinstance(pred, list): pred = pred[0]
            
            # Upsample prediction to match target resolution
            if pred.shape[2:] != hr.shape[2:]:
                pred = F.interpolate(pred, size=hr.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(pred, hr)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, use_amp: bool = False, use_tta: bool = False):
    """Validate model with optional mixed precision and TTA."""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    # Setup TTA if enabled
    tta_model = None
    if use_tta:
        tta_model = TestTimeAugmentation(model, flip=True, rotate=True, scale=True, n_scales=3)
    
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Validating"):
            lr = lr.to(device)
            hr = hr.to(device)
            
            if use_amp:
                with autocast():
                    if use_tta and tta_model is not None:
                        pred = tta_model.predict(lr)
                    else:
                        pred = model(lr)
                    if isinstance(pred, list): pred = pred[0]
                    # Upsample prediction to match target resolution
                    if pred.shape[2:] != hr.shape[2:]:
                        pred = F.interpolate(pred, size=hr.shape[2:], mode='bilinear', align_corners=False)
                    loss = criterion(pred, hr)
            else:
                if use_tta and tta_model is not None:
                    pred = tta_model.predict(lr)
                else:
                    pred = model(lr)
                if isinstance(pred, list): pred = pred[0]
                # Upsample prediction to match target resolution
                if pred.shape[2:] != hr.shape[2:]:
                    pred = F.interpolate(pred, size=hr.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(pred, hr)
            
            total_loss += loss.item()
            
            # Metrics - handle deep supervision outputs
            if isinstance(pred, list):
                pred = pred[0]  # Use main output for metrics
            
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
    
    # New training options
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of steps for gradient accumulation (effective batch = batch_size * steps)")
    parser.add_argument("--gradient_clip", type=float, default=None, 
                        help="Gradient clipping value")
    parser.add_argument("--loss", type=str, default="mae", choices=["mae", "ssim", "combined", "perceptual"],
                        help="Loss function type")
    parser.add_argument("--ssim_weight", type=float, default=0.15, 
                        help="Weight of SSIM loss in combined loss (0-1)")
    parser.add_argument("--use_tta", action="store_true", help="Enable Test-Time Augmentation during validation")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(output_dir / "logs")
    
    # Data
    train_ds = WRFDataset(args.lr_dir, args.hr_dir, "train")
    val_ds = WRFDataset(args.lr_dir, args.hr_dir, "val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    if args.model == "resunet":
        model = ResUNet(in_channels=6, out_channels=6)
    else:
        model = ResUNet(in_channels=6, out_channels=6)
    
    model = model.to(device)
    
    # Loss function
    if args.loss == "mae":
        criterion = nn.L1Loss()
    elif args.loss == "ssim":
        criterion = SSIMLoss()
    elif args.loss == "combined":
        criterion = CombinedLoss(alpha=1 - args.ssim_weight)
    elif args.loss == "perceptual":
        criterion = PerceptualLoss()
    else:
        criterion = nn.L1Loss()
    
    print(f"Using loss: {args.loss}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    use_amp = args.use_amp and device.type == "cuda"
    
    if use_amp:
        print(f"Mixed Precision Training enabled (effective batch: {args.batch_size * args.gradient_accumulation_steps})")
    
    if args.gradient_accumulation_steps > 1:
        print(f"Gradient Accumulation: {args.gradient_accumulation_steps} steps (effective batch: {args.batch_size * args.gradient_accumulation_steps})")
    
    # Train loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=use_amp, scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clip=args.gradient_clip
        )
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device, use_amp=use_amp, use_tta=args.use_tta)
        
        # Log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("train/lr", current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  MSE: {metrics['mse']:.6f} | MAE: {metrics['mae']:.4f} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / "best_model.pth")
            print(f"  -> Saved best model")
        
        scheduler.step(val_loss)
    
    writer.close()
    print(f"\nTraining complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
