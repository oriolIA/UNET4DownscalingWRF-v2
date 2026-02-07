"""
Metrics utilities for UNET4DownscalingWRF-v2.

Common metrics for evaluating weather downscaling models.
"""

import torch
import torch.nn.functional as F


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
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse_val).item()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Structural Similarity Index (simplified)."""
    # Simplified SSIM - just returning correlation-based similarity
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_std = pred.std()
    target_std = target.std()
    
    if pred_std == 0 or target_std == 0:
        return 1.0 if pred_mean == target_mean else 0.0
    
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    numerator = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2)
    denominator = (pred_mean ** 2 + target_mean ** 2 + c1) * (pred_std ** 2 + target_std ** 2 + c2)
    
    return (numerator / denominator).item()


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all metrics."""
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
