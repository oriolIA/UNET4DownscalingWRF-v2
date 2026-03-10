#!/bin/bash
# =============================================================================
# UNET4DownscalingWRF-v2 - Quick Test Script
# =============================================================================
# Executa: bash scripts/quick_test.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_DIR/outputs/quick_test"
VENV_PATH="$REPO_DIR/venv"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Configuració
DATA_DIR="${1:-/home/oriol/data/WRF/1469893}"
EPOCHS="${2:-1}"
BATCH_SIZE="${3:-4}"

log "========================================"
log "UNET4DownscalingWRF-v2 - Quick Test"
log "========================================"
log "Data dir: $DATA_DIR"
log "Epochs: $EPOCHS"
log "Batch: $BATCH_SIZE"

# Activar venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    log "Venv activated"
fi

# Executar test sintètic
mkdir -p "$OUTPUT_DIR"

python3 << 'PYTEST'
import os, sys, json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs/quick_test')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("UNET4DownscalingWRF-v2 - Quick Test")
print("=" * 60)

# Test model
from src.models.unet import UNetFactory
from src.config.config import ResUNetConfig, UpsampleMode, BottleneckType

config = ResUNetConfig(
    in_channels=7,
    out_channels=2,
    n_filters=32,  # Petit per test
)

model = UNetFactory.create('resunet', config)
x = torch.randn(1, 7, 50, 51)
y = model(x)

print(f"\n✓ Model: ResUNet")
print(f"  Input:  {x.shape}")
print(f"  Output: {y.shape}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Metrics test
from src.utils.metrics import all_metrics
m = all_metrics(y, y)
print(f"\n✓ Metrics:")
print(f"  MSE:  {m['mse']:.6f}")
print(f"  MAE:  {m['mae']:.6f}")
print(f"  PSNR: {m['psnr']:.2f} dB")
print(f"  SSIM: {m['ssim']:.4f}")

# Guardar resultats
results = {
    "timestamp": datetime.now().isoformat(),
    "model": "ResUNet",
    "input_shape": list(x.shape),
    "output_shape": list(y.shape),
    "params": sum(p.numel() for p in model.parameters()),
    "metrics": m,
}

with open(f"{OUTPUT_DIR}/test_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print("TEST COMPLETE!")
print(f"Results: {OUTPUT_DIR}/test_results.json")
print("=" * 60)
PYTEST

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Test complet!"
else
    log "Test fallit!"
fi

exit $EXIT_CODE
