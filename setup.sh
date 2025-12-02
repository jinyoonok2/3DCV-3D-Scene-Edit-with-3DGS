#!/bin/bash
# Setup script for 3D Gaussian Splatting Scene Editing
# Requires: CUDA-capable GPU, Python 3.10/3.12, CUDA toolkit matching PyTorch version
# Tested on: Ubuntu 20.04+, Vast.ai GPU instances
#
# Usage:
#   ./setup.sh          - Install dependencies (creates venv if needed)
#   ./setup.sh --reset  - Remove existing venv and reinstall from scratch

set -e  # Exit on error

VENV_DIR="venv"
RESET=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --reset|-r) RESET=true; shift ;;
        *) echo "Usage: ./setup.sh [--reset]"; exit 1 ;;
    esac
done

echo "=========================================="
echo "3DGS Scene Editing - Setup"
echo "=========================================="
echo ""

#=============================================================================
# 1. Python Environment Setup
#=============================================================================
echo "Step 1: Setting up Python environment"

# Find suitable Python version (prefer 3.10, fallback to 3.12)
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
else
    echo "✗ Error: Python 3.10 or 3.12 required"
    echo "  Install: sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
echo "  Using: $PYTHON_CMD"

# Handle reset
if [ "$RESET" = true ] && [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
    echo "  Removed existing venv"
fi

# Create/activate venv
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel -q
    echo "✓ Created new virtual environment"
else
    source "$VENV_DIR/bin/activate"
    echo "✓ Using existing virtual environment"
fi
echo ""

#=============================================================================
# 2. System Dependencies
#=============================================================================
echo "Step 2: Installing system dependencies"
if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq libjpeg-dev zlib1g-dev libtiff-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev \
        libfribidi-dev libxcb1-dev
    echo "✓ System packages installed"
else
    echo "⚠ Skipping system dependencies (no sudo access)"
fi
echo ""

#=============================================================================
# 3. GPU Verification
#=============================================================================
echo "Step 3: Verifying GPU"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    echo "✓ GPU detected"
else
    echo "⚠ nvidia-smi not found - ensure NVIDIA drivers are installed"
fi
echo ""

#=============================================================================
# 4. PyTorch with CUDA 12.6
#=============================================================================
echo "Step 4: Installing PyTorch (CUDA 12.6)"
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'none')")
    if [[ "$TORCH_CUDA" == "12.6" ]]; then
        echo "✓ PyTorch $TORCH_VER (CUDA $TORCH_CUDA) already installed"
    else
        echo "  Reinstalling PyTorch with CUDA 12.6..."
        pip uninstall -y torch torchvision torchaudio
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        echo "✓ PyTorch installed"
    fi
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo "✓ PyTorch installed"
fi

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

#=============================================================================
# 5. gsplat
#=============================================================================
echo "Step 5: Installing gsplat"
if python -c "import gsplat" 2>/dev/null; then
    echo "✓ gsplat already installed: $(python -c 'import gsplat; print(gsplat.__version__)')"
else
    pip install gsplat --no-deps
    pip install numpy jaxtyping typing_extensions ninja rich
    echo "✓ gsplat installed"
fi
echo ""

#=============================================================================
# 6. Clone Required Repositories
#=============================================================================
echo "Step 6: Cloning repositories"

# gsplat-src for examples
if [ ! -d "gsplat-src" ]; then
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src -q
    echo "✓ gsplat-src cloned"
else
    echo "✓ gsplat-src exists"
fi

# GroundingDINO for config files
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git -q
    echo "✓ GroundingDINO cloned"
else
    echo "✓ GroundingDINO exists"
fi

# TripoSR for 3D mesh generation
if [ ! -d "TripoSR" ]; then
    git clone https://github.com/VAST-AI-Research/TripoSR.git -q
    cp -r TripoSR/tsr ./
    touch tsr/__init__.py tsr/models/__init__.py
    echo "✓ TripoSR cloned and configured"
else
    echo "✓ TripoSR exists"
    [ ! -d "tsr" ] && cp -r TripoSR/tsr ./ && touch tsr/__init__.py tsr/models/__init__.py
fi
echo ""

#=============================================================================
# 7. Core Project Requirements
#=============================================================================
echo "Step 7: Installing core project requirements"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Core requirements installed"
fi
echo ""

#=============================================================================
# 8. TripoSR Dependencies
#=============================================================================
echo "Step 8: Installing TripoSR dependencies"
if [ -f "requirements-triposr.txt" ]; then
    pip install -r requirements-triposr.txt
    echo "✓ TripoSR dependencies installed"
fi

# SAM2 (separate install due to potential conflicts)
if ! python -c "import sam2" 2>/dev/null; then
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    echo "✓ SAM2 installed"
fi
echo "✓ Additional dependencies installed"
echo ""

#=============================================================================
# 9. gsplat Dependencies (with CUDA extensions)
#=============================================================================
echo "Step 9: Installing gsplat dependencies"

# Find CUDA toolkit
CUDA_HOME=""
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
    echo "  Found CUDA: $CUDA_HOME"
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    CUDA_HOME="/usr/local/cuda"
    echo "  Found CUDA: $CUDA_HOME"
fi

if [ -z "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "⚠ Warning: nvcc not found"
    echo "  fused-ssim requires CUDA toolkit"
    echo "  Install: conda install -c nvidia cuda-toolkit"
    echo "  Installing gsplat dependencies without fused-ssim..."
    if [ -f "requirements-gsplat.txt" ]; then
        grep -v "fused-ssim" requirements-gsplat.txt | pip install -r /dev/stdin
    fi
else
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export TORCH_CUDA_ARCH_LIST="8.9+PTX"
    export FORCE_CUDA=1
    
    # Install all gsplat dependencies with --no-build-isolation
    if [ -f "requirements-gsplat.txt" ]; then
        pip install --no-build-isolation -r requirements-gsplat.txt || \
        echo "  ⚠ Some gsplat packages failed (may be optional)"
    fi
    
    echo "✓ gsplat dependencies installed"
fi
echo ""

#=============================================================================
# 10. Download Model Weights
#=============================================================================
echo "Step 10: Downloading model weights"
if [ ! -f "models/groundingdino_swint_ogc.pth" ] || [ ! -f "models/sam2_hiera_large.pt" ]; then
    chmod +x download_models.sh
    ./download_models.sh
else
    echo "✓ Model weights already downloaded"
fi
echo ""

#=============================================================================
# 11. Download Dataset
#=============================================================================
echo "Step 11: Downloading dataset (MipNeRF360 garden)"
if [ ! -d "datasets/360_v2/garden" ]; then
    echo "  Downloading dataset (~2.5GB)..."
    mkdir -p datasets/360_v2
    
    # Try gsplat downloader
    if [ -f "gsplat-src/examples/datasets/download_dataset.py" ]; then
        cd gsplat-src/examples
        python datasets/download_dataset.py --dataset mipnerf360 2>/dev/null || true
        cd ../..
    fi
    
    # Copy if download succeeded
    if [ -d "gsplat-src/examples/data/360_v2/garden" ]; then
        cp -r gsplat-src/examples/data/360_v2/garden datasets/360_v2/
        echo "✓ Dataset downloaded"
    else
        echo "⚠ Auto-download failed. Manual download:"
        echo "  wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
        echo "  unzip 360_v2.zip -d datasets/"
    fi
else
    echo "✓ Dataset already exists"
fi
echo ""

#=============================================================================
# 12. Final Verification
#=============================================================================
echo "Step 12: Final verification"
python -c "
import torch, gsplat, sam2, numpy, PIL
print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ gsplat: {gsplat.__version__}')
print('✓ SAM2, NumPy, Pillow: OK')
"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Activate environment:"
echo "  source activate.sh"
echo ""
echo "Next steps:"
echo "  1. Upload trained checkpoint to: outputs/<project>/01_gs_base/ckpt_initial.pt"
echo "  2. Upload masks to: outputs/<project>/round_001/masks_<object>/sam_masks/"
echo "  3. Run pipeline: python 04a_lift_masks_to_roi3d.py --ckpt ... --masks_root ..."
echo ""
