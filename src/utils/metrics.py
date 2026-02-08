"""
Metrics utilities for WRF downscaling.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error."""
    return F.mse_loss(pred, target).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return F.l1_loss(pred, target).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse_val = F.mse_loss(pred, target)
    if mse_val == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse_val)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Structural Similarity Index."""
    # Simplified SSIM - full implementation in production
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    sigma_pred = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target ** 2
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred * mu_target
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
    
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean().item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }


if __name__ == "__main__":
    # Test
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    metrics = compute_all_metrics(pred, target)
    print("Metrics:", metrics)
