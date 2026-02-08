#!/usr/bin/env bash
# UNET4DownscalingWRF-v2 - Setup and Training Script

set -e

echo "========================================"
echo "UNET4DownscalingWRF-v2 - Training Setup"
echo "========================================"

# Environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Paths
LR_DIR="/home/oriol/data/WRF/1469893/d02"
HR_DIR="/home/oriol/data/WRF/1469893/d05"
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "LR Directory: $LR_DIR"
echo "HR Directory: $HR_DIR"
echo "Output: $OUTPUT_DIR"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision numpy xarray netCDF4 tqdm tensorboard
fi

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training
python3 -m src.training.trainer \
    --lr_dir "$LR_DIR" \
    --hr_dir "$HR_DIR" \
    --model resunet \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo "========================================"
echo "Training completed!"
echo "TensorBoard: tensorboard --logdir=$OUTPUT_DIR"
echo "========================================"
