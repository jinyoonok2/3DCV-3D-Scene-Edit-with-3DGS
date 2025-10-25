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

# Clone gsplat repository (not in git since it's in .gitignore)
echo "3. Cloning gsplat repository..."
if [ ! -d "gsplat-src" ]; then
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src
    echo "✓ gsplat repository cloned"
else
    echo "✓ gsplat-src already exists"
fi
echo ""

# Install gsplat examples requirements
echo "4. Installing gsplat examples requirements..."
pip install -r gsplat-src/examples/requirements.txt
echo "✓ gsplat examples requirements installed"
echo ""

# Install project requirements
echo "5. Installing project requirements..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "✓ Project requirements installed"
else
    echo "⚠ No requirements.txt found, skipping"
fi
echo ""

# Install GroundingDINO and SAM2
echo "5.5. Installing GroundingDINO and SAM2..."
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/segment-anything-2.git
echo "✓ GroundingDINO and SAM2 installed"
echo ""

# Clone GroundingDINO repo for config files
echo "5.6. Cloning GroundingDINO repository (for config files)..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    echo "✓ GroundingDINO repository cloned"
else
    echo "✓ GroundingDINO repository already exists"
fi
echo ""

# Download model weights
echo "5.7. Downloading model weights..."
chmod +x download_models.sh
./download_models.sh
echo "✓ Model weights downloaded"
echo ""

# Download dataset
echo "6. Downloading Mip-NeRF 360 dataset (all scenes, ~2.5GB)..."
cd gsplat-src/examples
python datasets/download_dataset.py --dataset mipnerf360
cd ../..
echo "✓ Dataset downloaded"
echo ""

# Move dataset to expected location
echo "7. Moving garden dataset to correct location..."
mkdir -p datasets/360_v2
if [ -d "gsplat-src/examples/data/360_v2/garden" ]; then
    cp -r gsplat-src/examples/data/360_v2/garden datasets/360_v2/garden
    echo "✓ Garden dataset moved to datasets/360_v2/garden"
else
    echo "⚠ Dataset source not found, may already be in correct location"
fi
echo ""

# Validate dataset
echo "8. Validating dataset..."
python 00_check_dataset.py --data_root datasets/360_v2/garden --factor 4
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000"
echo ""
echo "To export as PLY file for visualization:"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000 --save_ply"
echo ""
echo "Tip: Use tmux to keep training running after disconnect:"
echo "  tmux new -s training"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000"
echo "  # Detach: Ctrl+B, then D"
echo "  # Reattach: tmux attach -t training"
echo ""
