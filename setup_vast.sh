#!/bin/bash
# Setup script for Vast.ai environment
# This script installs dependencies and downloads the Mip-NeRF 360 garden dataset

set -e  # Exit on any error

echo "=========================================="
echo "Vast.ai Setup for 3DGS Scene Editing"
echo "=========================================="
echo ""

# Verify CUDA/PyTorch installation
echo "1. Verifying CUDA and PyTorch..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
echo ""

# Install gsplat
echo "2. Installing gsplat (this may take a few minutes to compile CUDA kernels)..."
pip install gsplat
echo "✓ gsplat installed"
echo ""

# Install gsplat examples requirements
echo "3. Installing gsplat examples requirements..."
pip install -r gsplat-src/examples/requirements.txt
echo "✓ gsplat examples requirements installed"
echo ""

# Install project requirements
echo "4. Installing project requirements..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "✓ Project requirements installed"
else
    echo "⚠ No requirements.txt found, skipping"
fi
echo ""

# Download dataset
echo "5. Downloading Mip-NeRF 360 garden dataset..."
cd gsplat-src/examples
python datasets/download_dataset.py --dataset mipnerf360 --subset garden
cd ../..
echo "✓ Dataset downloaded"
echo ""

# Move dataset to expected location
echo "6. Moving dataset to correct location..."
mkdir -p datasets/360_v2
if [ -d "gsplat-src/examples/data/360_v2/garden" ]; then
    mv gsplat-src/examples/data/360_v2/garden datasets/360_v2/garden 2>/dev/null || echo "⚠ Dataset already in place"
    echo "✓ Dataset moved to datasets/360_v2/garden"
else
    echo "⚠ Dataset source not found, may already be in correct location"
fi
echo ""

# Validate dataset
echo "7. Validating dataset..."
python 00_check_dataset.py --data_root datasets/360_v2/garden --factor 4
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000"
echo ""
echo "Or for a quick test (1000 iterations):"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 1000"
echo ""
echo "Tip: Use tmux to keep training running after disconnect:"
echo "  tmux new -s training"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000"
echo "  # Detach: Ctrl+B, then D"
echo "  # Reattach: tmux attach -t training"
echo ""
