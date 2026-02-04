# UNET4DownscalingWRF-v2

**Improved Wind Field Downscaling with Modular UNet Architecture**

A refactored and enhanced version of UNET4downscallinngWRF with:
- **Cleaner architecture** - Modular, composable components
- **Flexible configuration** - Dataclass-based config system
- **Better training pipeline** - Callbacks, metrics, logging
- **Type safety** - Full type hints
- **Enhanced models** - Improved ResUNet, attention mechanisms

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
Encoder: Residual Blocks + MaxPool (×3)
    ↓
Bottleneck: Dilated/Enhanced Residual (configurable)
    ↓
Decoder: Upsample + Residual Blocks (×3)
    ↓
Output (C×H×W)
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

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt`

## License

MIT
