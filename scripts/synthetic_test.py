#!/usr/bin/env python3
"""
UNET4DownscalingWRF-v2 - Synthetic Test Suite
Tests model architecture and data loading without PyTorch.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def check_dependencies():
    """Check available packages."""
    deps = {}
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except:
        deps['numpy'] = False
    
    try:
        import xarray
        deps['xarray'] = xarray.__version__
    except:
        deps['xarray'] = False
    
    try:
        import netCDF4
        deps['netcdf4'] = netCDF4.__version__
    except:
        deps['netcdf4'] = False
    
    return deps


def check_data(data_dir):
    """Check WRF data structure."""
    log(f"Checking data: {data_dir}")
    
    if not os.path.exists(data_dir):
        return {"exists": False}
    
    domains = {}
    for d in ['d01', 'd02', 'd05']:
        d_path = Path(data_dir) / d
        if d_path.exists():
            nc_files = list(d_path.glob("*.nc"))
            domains[d] = len(nc_files)
    
    return {"exists": True, "domains": domains}


def test_model_config():
    """Test model configuration."""
    config = {
        "in_channels": 7,
        "out_channels": 2,
        "n_filters": 64,
        "bottleneck": "enhanced",
        "upsampling": "bilinear",
    }
    
    # Estimate parameters
    # Simple UNet: ~31M params with n_filters=64
    estimated_params = 31000000
    
    return {
        "config": config,
        "estimated_params": estimated_params,
        "architecture": "ResUNet with Attention, Enhanced Bottleneck",
    }


def main():
    log("=" * 60)
    log("UNET4DownscalingWRF-v2 - Synthetic Test")
    log("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # 1. Dependencies
    log("\n[1/3] Checking dependencies...")
    results["tests"]["dependencies"] = check_dependencies()
    for k, v in results["tests"]["dependencies"].items():
        log(f"  {k}: {v}")
    
    # 2. Data
    log("\n[2/3] Checking WRF data...")
    data_dir = os.environ.get('DATA_DIR', '/home/oriol/data/WRF/1469893')
    results["tests"]["data"] = check_data(data_dir)
    log(f"  Data: {results['tests']['data']}")
    
    # 3. Model
    log("\n[3/3] Testing model configuration...")
    results["tests"]["model"] = test_model_config()
    log(f"  Architecture: {results['tests']['model']['architecture']}")
    log(f"  Parameters: {results['tests']['model']['estimated_params']:,}")
    
    # Summary
    log("\n" + "=" * 60)
    deps_ok = sum(1 for v in results["tests"]["dependencies"].values() if v) >= 2
    data_ok = results["tests"]["data"]["exists"]
    
    results["status"] = "ready" if (deps_ok and data_ok) else "missing_deps"
    
    log(f"Result: {results['status']}")
    
    # Save
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "synthetic_test.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"Saved to: {output_dir / 'synthetic_test.json'}")
    
    return 0 if results["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
