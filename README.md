# UNET4DownscalingWRF-v2

**Improved Wind Field Downscaling with Modular UNet Architecture**

A refactored and enhanced version of UNET4downscallinngWRF with:
- **Cleaner architecture** - Modular, composable components
- **Flexible configuration** - Dataclass-based config system
- **Better training pipeline** - Callbacks, metrics, logging
- **Type safety** - Full type hints
- **Enhanced models** - SE blocks, attention gates, deep supervision
- **Pretrained encoders** - Transfer learning with ResNet-18/34

## Quick Start

```bash
# Clone and install
git clone https://github.com/oriolIA/UNET4DownscalingWRF-v2.git
cd UNET4DownscalingWRF-v2

# Install dependencies
pip install -r requirements.txt

# Train
python -m src.main --mode train --config configs/resunet_default.yaml

# Train with pretrained encoder (recommended!)
python -m src.main --mode train --encoder resnet18 --pretrained

# Predict
python -m src.main --mode predict --model outputs/latest/model.pth --input data/
```

## Architecture Highlights

```
Input (C×H×W)
    ↓
Encoder: SE-Residual Blocks + MaxPool (×4)
    OR Pretrained ResNet-18/34
    ↓
Bottleneck: SE-Enhanced Residual
    ↓
Decoder: Upsample + Attention Gates + SE-Residual Blocks (×4)
    ↓
Output (C×H×W) + Multi-scale outputs (deep supervision)
```

## Model Variants

| Model | SE | CBAM | AG | ASPP | Deep Sup | Pretrained | Parameters |
|-------|----|------|----|------|----------|------------|------------|
| ResUNet (SE) | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | 33.5M |
| ResUNet (CBAM) | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | 33.5M |
| ResUNet (none) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 33.3M |
| ResUNet Small | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | 8.4M |
| **ResUNet (ASPP)** | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | **58.2M** |
| **ResUNet (ASPP+CBAM)** | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | **58.2M** |
| UNet Classic | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ~17M |
| **ResNet18 Pretrained** | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ~20M |
| **ResNet34 Pretrained** | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ~30M |

### Creating Models

```python
from src.models.resunet import ResUNet, UNetClassic, create_model
from src.models.pretrained_encoder import UNetPretrained, create_model

# Full-featured model (custom encoder)
model = ResUNet(in_channels=7, out_channels=2, use_se=True, use_deep_supervision=True)

# Pretrained encoder (recommended for transfer learning!)
model = UNetPretrained(
    encoder_name="resnet18",  # or "resnet34"
    pretrained=True,           # ImageNet pretrained weights
    in_channels=7,
    out_channels=2,
    use_attention=True,
    use_deep_supervision=True,
    frozen_stages=2            # Freeze first 2 encoder stages
)

# Factory pattern
model = create_model('resunet_deep')
model = create_model('resunet_se')
model = create_model('unet_classic')
model = create_model('resnet18', in_channels=7, out_channels=2)
```

### Pretrained Encoder Benefits

- **Transfer learning**: Start with ImageNet-pretrained features
- **Faster convergence**: Better initial representations
- **Less data needed**: Leverage pretrained knowledge
- **Frozen stages option**: Fine-tune only decoder for faster training

```python
# Fine-tune only decoder (faster training)
model = UNetPretrained("resnet18", pretrained=True, frozen_stages=2)

# Full fine-tuning
model = UNetPretrained("resnet18", pretrained=True, frozen_stages=-1)
```

## Key Improvements Over v1

| Aspect | v1 | v2 |
|--------|----|----|
| Config | YAML + argparse | Dataclass + hydra |
| Models | Single file | Factory pattern |
| Training | Monolithic loop | Callbacks + hooks |
| Logging | Basic | TensorBoard + wandb |
| Error handling | Minimal | Structured exceptions |
| Type hints | Partial | Full |
| Attention | None | AG-UNet gates |
| Channel Attention | None | Squeeze-and-Excitation |
| Multi-scale | None | Deep Supervision |
| Mixed Precision | None | AMP (torch.cuda.amp) |
| Loss Functions | MAE only | MAE + SSIM + Combined + Perceptual |
| Gradient Accumulation | None | Configurable steps |

## New Features

### Squeeze-and-Excitation (SE) Blocks
Channel attention mechanism that learns to recalibrate feature responses.

### CBAM (Convolutional Block Attention Module)
Combined channel + spatial attention for more focused feature learning.
```python
# Using CBAM attention
model = ResUNet(in_channels=7, out_channels=2, attention='cbam')

# Using SE attention (default)
model = ResUNet(in_channels=7, out_channels=2, attention='se')

# No attention
model = ResUNet(in_channels=7, out_channels=2, attention='none')
```

### Attention Gates (AG-UNET)
Filters skip connections to focus on relevant spatial regions.

### ASPP (Atrous Spatial Pyramid Pooling)
Multi-scale context in the bottleneck using dilated convolutions at different rates (6, 12, 18).
**Recommended for WRF downscaling** - captures atmospheric features at multiple scales.
```python
# Using ASPP (recommended for WRF)
model = ResUNet(in_channels=7, out_channels=2, use_aspp=True)

# ASPP + CBAM (maximum attention)
model = ResUNet(in_channels=7, out_channels=2, attention='cbam', use_aspp=True)

# Factory
model = create_model('resunet_aspp')
```

### Deep Supervision
Multiple output heads at different scales for better gradient flow and intermediate predictions.

### Mixed Precision Training (AMP)
Automatic Mixed Precision training for faster training and reduced memory usage.
```bash
python -m src.main --mode train --use_amp
```

### Advanced Loss Functions
- **SSIM Loss**: Structural Similarity Index for better perceptual quality
- **Combined Loss**: Weighted combination of MAE + SSIM (default)
- **Perceptual Loss**: Multi-scale L1 loss for deep supervision

```bash
# Using different loss functions
python -m src.main --mode train --loss ssim
python -m src.main --mode train --loss combined
python -m src.main --mode train --loss perceptual
```

### Gradient Accumulation
Train with larger effective batch sizes even with limited GPU memory.
```bash
python -m src.main --mode train --gradient_accumulation_steps 4
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt`

## License

MIT
