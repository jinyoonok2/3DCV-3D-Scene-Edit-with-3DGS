#!/bin/bash
# Vast.ai setup script for 3DGS Scene Editing project
# Optimized for cloud GPU instances with CUDA 12.6+
#
# Usage:
#   ./setup_vastai.sh          - Install dependencies (creates venv if needed)
#   ./setup_vastai.sh --reset  - Remove existing venv and reinstall from scratch

set -e  # Exit on any error

VENV_DIR="venv"
RESET=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --reset|-r)
            RESET=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: ./setup_vastai.sh [--reset]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "3DGS Scene Editing - Vast.ai Setup"
echo "=========================================="
echo ""

# Handle reset flag
if [ "$RESET" = true ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "🔄 Resetting environment..."
        echo "Removing existing virtual environment: $VENV_DIR"
        rm -rf "$VENV_DIR"
        echo "✓ Virtual environment removed"
        echo ""
    else
        echo "⚠ No existing virtual environment found at $VENV_DIR"
        echo ""
    fi
fi

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment already exists at: $VENV_DIR"
    echo "  (Use --reset flag to reinstall from scratch)"
    echo ""
    echo "Activating existing environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "1. Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at: $VENV_DIR"
    echo ""
    
    echo "2. Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "✓ Virtual environment activated"
    echo ""
    
    echo "3. Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    echo "✓ pip upgraded"
    echo ""
fi

# Verify Python and system info
echo "4. Verifying environment..."
python --version
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Install PyTorch with CUDA 12.6
echo "5. Installing PyTorch with CUDA 12.6..."
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed:"
    python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    echo ""
else
    echo "  Installing PyTorch with CUDA 12.6 support..."
    echo "  This may take several minutes..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo "✓ PyTorch installed"
    echo ""
fi

# Verify CUDA after PyTorch installation
echo "6. Verifying CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA is available'); print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA Version: {torch.version.cuda}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

# Install gsplat from PyPI
echo "7. Installing gsplat..."
if python -c "import gsplat" 2>/dev/null; then
    echo "✓ gsplat already installed:"
    python -c "import gsplat; print(f'  Version: {gsplat.__version__}')"
else
    echo "  Compiling CUDA kernels (this may take a few minutes)..."
    pip install gsplat
    echo "✓ gsplat installed"
fi
echo ""

# Clone gsplat repository for examples (needed for requirements.txt)
echo "8. Setting up gsplat examples repository..."
if [ ! -d "gsplat-src" ]; then
    echo "  Cloning gsplat repository..."
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src
    echo "✓ gsplat repository cloned"
else
    echo "✓ gsplat-src already exists"
fi
echo ""

# Install project requirements first (lighter weight dependencies)
echo "9. Installing project requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Project requirements installed"
else
    echo "⚠ No requirements.txt found"
fi
echo ""

# Install GroundingDINO
echo "10. Installing GroundingDINO..."
if python -c "import groundingdino" 2>/dev/null; then
    echo "✓ GroundingDINO already installed"
else
    pip install groundingdino-py
    echo "✓ GroundingDINO installed"
fi
echo ""

# Install SAM2 from GitHub
echo "11. Installing SAM2..."
if python -c "import sam2" 2>/dev/null; then
    echo "✓ SAM2 already installed"
else
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    echo "✓ SAM2 installed"
fi
echo ""

# Download model weights
echo "12. Downloading model weights..."
if [ ! -f "models/groundingdino_swint_ogc.pth" ] || [ ! -f "models/sam2_hiera_large.pt" ]; then
    echo "  Running download_models.sh..."
    chmod +x download_models.sh
    ./download_models.sh
else
    echo "✓ Model weights already downloaded"
    echo "  GroundingDINO: models/groundingdino_swint_ogc.pth"
    echo "  SAM2: models/sam2_hiera_large.pt"
fi
echo ""

# Final verification
echo "13. Final verification..."
echo "  Testing imports..."
python -c "
import torch
import gsplat
import groundingdino
import sam2
import numpy as np
import PIL
print('✓ All core packages imported successfully')
print(f'  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  gsplat: {gsplat.__version__}')
print('  GroundingDINO: OK')
print('  SAM2: OK')
"
echo ""

# Download dataset
echo "14. Checking for dataset..."
if [ ! -d "datasets/360_v2/garden" ]; then
    echo "  Downloading Mip-NeRF 360 garden dataset (~2.5GB)..."
    cd gsplat-src/examples
    python datasets/download_dataset.py --dataset mipnerf360
    cd ../..
    
    echo "  Moving dataset to project directory..."
    mkdir -p datasets/360_v2
    if [ -d "gsplat-src/examples/data/360_v2/garden" ]; then
        cp -r gsplat-src/examples/data/360_v2/garden datasets/360_v2/garden
        echo "✓ Dataset downloaded and moved to datasets/360_v2/garden"
    else
        echo "⚠ Dataset download location unexpected, please check manually"
    fi
else
    echo "✓ Dataset already exists at: datasets/360_v2/garden"
fi
echo ""

echo "=========================================="
echo "Vast.ai Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "✓ Public files downloaded:"
echo "  - Dataset: datasets/360_v2/garden"
echo "  - Models: models/groundingdino_swint_ogc.pth, models/sam2_hiera_large.pt"
echo ""
echo "⚠ You still need to upload YOUR trained files from local machine:"
echo "  1. Checkpoint (trained Gaussian Splatting model):"
echo "     outputs/garden/01_gs_base/ckpt_initial.pt (~500MB)"
echo ""
echo "  2. Masks (generated from Module 03):"
echo "     outputs/garden/round_001/masks_brown_plant/sam_masks/ (~200MB)"
echo ""
echo "Upload using scp or rsync from your local machine:"
echo "  scp -r outputs/garden root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/"
echo ""
echo "Then run Module 04:"
echo "  source activate.sh"
echo "  python 04_lift_masks_to_roi3d.py \\"
echo "    --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \\"
echo "    --masks_root outputs/garden/round_001/masks_brown_plant/sam_masks \\"
echo "    --data_root datasets/360_v2/garden \\"
echo "    --roi_thresh 0.5 \\"
echo "    --sh_degree 3 \\"
echo "    --output_dir outputs/garden/round_001"
echo ""
echo "Download results back to local machine:"
echo "  scp root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs/garden/round_001/roi.pt outputs/garden/round_001/"
echo ""
echo "=========================================="
echo ""
echo "Note: This script skips gsplat examples requirements to avoid"
echo "      build issues. Core functionality is fully available."
echo ""
