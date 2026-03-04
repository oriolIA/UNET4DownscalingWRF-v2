# UNET4DownscalingWRF-v2

**Improved Wind Field Downscaling with Modular UNet Architecture**

A refactored and enhanced version of UNET4downscallinngWRF with:
- **Cleaner architecture** - Modular, composable components
- **Flexible configuration** - Dataclass-based config system
- **Better training pipeline** - Callbacks, metrics, logging
- **Type safety** - Full type hints
- **Enhanced models** - SE blocks, attention gates, deep supervision

## Quick Start

```bash
# Clone and install
git clone https://github.com/oriolIA/UNET4DownscalingWRF-v2.git
cd UNET4DownscalingWRF-v2

# Install dependencies
pip install -r requirements.txt

# Train
python -m src.main --mode train --config configs/resunet_default.yaml

# Predict
python -m src.main --mode predict --model outputs/latest/model.pth --input data/
```

## Architecture Highlights

```
Input (C×H×W)
    ↓
Encoder: SE-Residual Blocks + MaxPool (×4)
    ↓
Bottleneck: SE-Enhanced Residual
    ↓
Decoder: Upsample + Attention Gates + SE-Residual Blocks (×4)
    ↓
Output (C×H×W) + Multi-scale outputs (deep supervision)
```

## Model Variants

| Model | SE Blocks | Attention | Deep Supervision | Parameters |
|-------|------------|-----------|------------------|------------|
| ResUNet | ✓ | ✓ | ✓ | 33.5M |
| ResUNet (no SE) | ✗ | ✓ | ✓ | 33.1M |
| ResUNet (basic) | ✗ | ✗ | ✗ | ~31M |
| UNet Classic | ✗ | ✗ | ✗ | ~17M |

### Creating Models

```python
from src.models.resunet import ResUNet, UNetClassic, create_model

# Full-featured model
model = ResUNet(in_channels=7, out_channels=2, use_se=True, use_deep_supervision=True)

# No SE blocks
model = ResUNet(in_channels=7, out_channels=2, use_se=False)

# Factory pattern
model = create_model('resunet_deep')
model = create_model('resunet_se')
model = create_model('unet_classic')
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

## New Features

### Squeeze-and-Excitation (SE) Blocks
Channel attention mechanism that learns to recalibrate feature responses.

### Attention Gates (AG-UNET)
Filters skip connections to focus on relevant spatial regions.

### Deep Supervision
Multiple output heads at different scales for better gradient flow and intermediate predictions.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt`

## License

MIT
