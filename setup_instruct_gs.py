#!/usr/bin/env python3
"""
Quick setup script for InstructGS.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Quick setup and test."""
    project_root = Path(__file__).parent
    
    print("🚀 InstructGS Quick Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check key directories
    if not (project_root / "gsplat-src").exists():
        print("❌ gsplat-src not found")
        sys.exit(1)
    print("✅ gsplat-src found")
    
    # Check datasets
    datasets_dir = project_root / "datasets"
    if datasets_dir.exists() and any(datasets_dir.iterdir()):
        print("✅ Datasets found")
        # List available datasets
        print("   Available datasets:")
        for dataset in datasets_dir.iterdir():
            if dataset.is_dir():
                print(f"     📁 {dataset.name}")
    else:
        print("⚠️  No datasets found")
        print("   To download Mip-NeRF 360 dataset:")
        print("   cd gsplat-src/examples")
        print("   python datasets/download_dataset.py --dataset mipnerf360 --save-dir ../../datasets")
    
    # Check dependencies
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🎮 CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("   ⚠️  CUDA not available (CPU mode)")
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch")
    
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not installed")
        print("   Install with: pip install pyyaml")
    
    try:
        import rich
        print("✅ Rich available (for interactive interface)")
    except ImportError:
        print("⚠️  Rich not installed (optional)")
        print("   Install with: pip install rich")
    
    print("\n🎯 Quick Start Examples:")
    print("-" * 40)
    
    # Show example commands
    if datasets_dir.exists():
        # Find first available dataset
        for dataset in datasets_dir.iterdir():
            if dataset.is_dir():
                dataset_path = f"datasets/{dataset.name}"
                break
        else:
            dataset_path = "datasets/360_v2/garden"
    else:
        dataset_path = "datasets/360_v2/garden"
    
    print("1. Interactive mode (recommended):")
    print("   python train_instruct_gs.py --interactive")
    print()
    print("2. Direct command line:")
    print(f'   python train_instruct_gs.py --data {dataset_path} --prompt "Turn it into a painting"')
    print()
    print("3. Test gsplat (to make sure everything works):")
    print(f"   cd gsplat-src/examples")
    print(f"   python simple_trainer.py default --disable_viewer --max_steps 100 --data_dir ../../{dataset_path}")
    print()
    
    print("📚 For more information, see the README.md")


if __name__ == "__main__":
    main()