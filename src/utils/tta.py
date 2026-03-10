"""
Test-Time Augmentation (TTA) utilities for WRF downscaling.

TTA applies multiple augmentations at inference time and averages
the results for better predictions.

Reference: https://arxiv.org/abs/1909.13457
"""

import torch
import torch.nn.functional as F
from typing import List, Callable, Optional
import numpy as np


class TestTimeAugmentation:
    """
    Test-Time Augmentation wrapper.
    
    Applies augmentations at inference time and averages predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        augmentations: List[str] = None,
        flip: bool = True,
        rotate: bool = True,
        brightness: bool = False,
        scale: bool = False,
        n_scales: int = 5,
        scale_range: tuple = (0.8, 1.2)
    ):
        """
        Args:
            model: The model to wrap
            augmentations: List of augmentation types to apply
            flip: Apply horizontal/vertical flips
            rotate: Apply 90-degree rotations
            brightness: Apply brightness adjustments
            scale: Apply multi-scale inference
            n_scales: Number of scales for multi-scale inference
            scale_range: Range of scales to test
        """
        self.model = model
        self.flip = flip
        self.rotate = rotate
        self.brightness = brightness
        self.scale = scale
        self.n_scales = n_scales
        self.scale_range = scale_range
        
        # Set model to eval mode
        self.model.eval()
    
    def _flip_augment(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate flip augmentations."""
        augments = [x]  # Original
        
        # Horizontal flip
        augments.append(torch.flip(x, dims=[3]))
        
        # Vertical flip
        augments.append(torch.flip(x, dims=[2]))
        
        # Both flips
        augments.append(torch.flip(x, dims=[2, 3]))
        
        return augments
    
    def _rotate_augment(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate rotation augmentations (90, 180, 270)."""
        augments = [x]
        
        # Rotate 90
        augments.append(torch.rot90(x, k=1, dims=[2, 3]))
        
        # Rotate 180
        augments.append(torch.rot90(x, k=2, dims=[2, 3]))
        
        # Rotate 270
        augments.append(torch.rot90(x, k=3, dims=[2, 3]))
        
        return augments
    
    def _brightness_augment(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate brightness augmentations."""
        augments = [x]
        
        # Slightly brighter
        bright = x * 1.1
        bright = torch.clamp(bright, x.min(), x.max())
        augments.append(bright)
        
        # Slightly darker
        dark = x * 0.9
        dark = torch.clamp(dark, x.min(), x.max())
        augments.append(dark)
        
        return augments
    
    def _scale_augment(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate multi-scale augmentations."""
        b, c, h, w = x.shape
        
        # Generate scales
        scales = np.linspace(self.scale_range[0], self.scale_range[1], self.n_scales)
        
        augments = []
        for s in scales:
            # Scale the input
            new_h, new_w = int(h * s), int(w * s)
            scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Scale back to original size
            scaled = F.interpolate(scaled, size=(h, w), mode='bilinear', align_corners=False)
            augments.append(scaled)
        
        return augments
    
    def predict(self, x: torch.Tensor, use_deep_supervision: bool = False) -> torch.Tensor:
        """
        Apply TTA and return averaged prediction.
        
        Args:
            x: Input tensor [B, C, H, W]
            use_deep_supervision: If True, handle multi-output models
            
        Returns:
            Averaged prediction [B, C, H, W]
        """
        with torch.no_grad():
            all_predictions = []
            
            # Original
            pred = self.model(x)
            if isinstance(pred, list):
                # Handle deep supervision - use first output (full resolution)
                pred = pred[0]
            all_predictions.append(pred)
            
            # Flip augmentations
            if self.flip:
                for aug in self._flip_augment(x):
                    pred = self.model(aug)
                    if isinstance(pred, list):
                        pred = pred[0]
                    # Flip back
                    pred = torch.flip(pred, dims=[3])
                    all_predictions.append(pred)
                    
                    pred = torch.flip(pred, dims=[2])
                    all_predictions.append(pred)
                    
                    pred = torch.flip(pred, dims=[2, 3])
                    all_predictions.append(pred)
            
            # Rotate augmentations
            if self.rotate:
                for k, aug in enumerate(self._rotate_augment(x)):
                    if k == 0:  # Skip original
                        continue
                    pred = self.model(aug)
                    if isinstance(pred, list):
                        pred = pred[0]
                    # Rotate back (inverse rotation)
                    pred = torch.rot90(pred, k=4-k, dims=[2, 3])
                    all_predictions.append(pred)
            
            # Multi-scale inference
            if self.scale:
                for aug in self._scale_augment(x):
                    pred = self.model(aug)
                    if isinstance(pred, list):
                        pred = pred[0]
                    all_predictions.append(pred)
            
            # Average all predictions
            stacked = torch.stack(all_predictions, dim=0)
            avg_pred = stacked.mean(dim=0)
            
            return avg_pred
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with TTA."""
        return self.predict(x)


def apply_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    flip: bool = True,
    rotate: bool = True,
    n_scales: int = 3
) -> torch.Tensor:
    """
    Simple TTA function for quick use.
    
    Args:
        model: Model to use
        x: Input tensor
        flip: Apply flip augmentations
        rotate: Apply rotation augmentations
        n_scales: Number of scales for multi-scale inference
        
    Returns:
        Averaged prediction
    """
    tta = TestTimeAugmentation(
        model=model,
        flip=flip,
        rotate=rotate,
        scale=n_scales > 1,
        n_scales=n_scales
    )
    return tta.predict(x)


if __name__ == "__main__":
    # Test TTA
    from .resunet import ResUNet
    
    print("=" * 50)
    print("Testing TTA")
    
    # Create model
    model = ResUNet(in_channels=7, out_channels=2)
    model.eval()
    
    # Test input
    x = torch.randn(1, 7, 128, 128)
    
    # Standard prediction
    with torch.no_grad():
        pred_std = model(x)
        if isinstance(pred_std, list):
            pred_std = pred_std[0]
        print(f"  Standard output: {pred_std.shape}")
    
    # TTA prediction
    tta = TestTimeAugmentation(model, flip=True, rotate=True, scale=True, n_scales=3)
    pred_tta = tta.predict(x)
    print(f"  TTA output: {pred_tta.shape}")
    
    print("  TTA enabled!")
