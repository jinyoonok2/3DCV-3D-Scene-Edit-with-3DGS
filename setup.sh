#!/bin/bash
# Setup script for 3D Gaussian Splatting Scene Editing
# Requires: CUDA-capable GPU, Python 3.10/3.12, CUDA toolkit matching PyTorch version
# Tested on: Ubuntu 20.04+, Vast.ai GPU instances
#
# Usage:
#   ./setup.sh          - Install dependencies (creates venv if needed)
#   ./setup.sh --reset  - Remove existing venv and reinstall from scratch

set -e  # Exit on error

# Function to ensure conda environment is active
activate_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

ENV_NAME="3dgs"
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
# 1. Python Environment Setup (Conda)
#=============================================================================
echo "Step 1: Setting up conda environment"

ENV_NAME="3dgs"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "✗ Error: conda not found"
    echo "  Please install Miniconda or Anaconda first"
    echo "  Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "  Using: $(conda --version)"

# Handle reset
if [ "$RESET" = true ]; then
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    echo "  Removed existing conda environment"
fi

# Create/activate conda environment
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    conda create -n "$ENV_NAME" python=3.12 -y -q
    echo "✓ Created new conda environment: $ENV_NAME"
else
    echo "✓ Using existing conda environment: $ENV_NAME"
fi

# Activate environment
activate_conda_env
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
# 4. CUDA Toolkit and PyTorch
#=============================================================================
echo "Step 4: Installing CUDA toolkit and PyTorch"

# Install CUDA toolkit via conda
if ! conda list | grep -q "cuda-toolkit"; then
    echo "  Installing CUDA toolkit..."
    conda install -c nvidia cuda-toolkit=12.6 -y -q
    echo "✓ CUDA toolkit installed"
else
    echo "✓ CUDA toolkit already installed"
fi

# Install PyTorch with CUDA support
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'none')")
    if [[ "$TORCH_CUDA" == "12.6" ]]; then
        echo "✓ PyTorch $TORCH_VER (CUDA $TORCH_CUDA) already installed"
    else
        echo "  Installing PyTorch with CUDA 12.6..."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia -y -q
        echo "✓ PyTorch installed"
    fi
else
    echo "  Installing PyTorch with CUDA 12.6..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia -y -q
    echo "✓ PyTorch installed"
fi

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

#=============================================================================
# 5. gsplat (pip package)
#=============================================================================
echo "Step 5: Installing gsplat from pip"
# Ensure we're in the right conda environment
activate_conda_env
if python -c "import gsplat" 2>/dev/null; then
    echo "✓ gsplat already installed: $(python -c 'import gsplat; print(gsplat.__version__)')"
else
    pip install gsplat
    echo "✓ gsplat installed from pip"
fi
echo ""

#=============================================================================
# 6. Clone Required Repositories
#=============================================================================
echo "Step 6: Setting up external repositories"

# gsplat-src contains examples and dataset parsers (NOT for installation)
echo "✓ gsplat-src examples available (COLMAP parser, utils, fused_ssim)"

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
# Ensure we're in the right conda environment
activate_conda_env
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Core requirements installed"
fi
echo ""

#=============================================================================
# 8. TripoSR Dependencies
#=============================================================================
echo "Step 8: Installing TripoSR dependencies"
# Ensure we're in the right conda environment
activate_conda_env
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
# 9. gsplat-src Example Dependencies (with CUDA extensions)
#=============================================================================
echo "Step 9: Installing gsplat-src example dependencies"
# Ensure we're in the right conda environment
activate_conda_env

# Find CUDA toolkit for fused-ssim compilation
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
    echo "  fused-ssim (from gsplat-src examples) requires CUDA toolkit"
    echo "  Install: conda install -c nvidia cuda-toolkit"
    echo "  Installing gsplat-src example dependencies without fused-ssim..."
    if [ -f "requirements-gsplat.txt" ]; then
        grep -v "fused-ssim" requirements-gsplat.txt | pip install -r /dev/stdin
    fi
else
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export TORCH_CUDA_ARCH_LIST="8.9+PTX"
    export FORCE_CUDA=1
    
    # Install all gsplat-src example dependencies with --no-build-isolation
    if [ -f "requirements-gsplat.txt" ]; then
        pip install --no-build-isolation -r requirements-gsplat.txt || \
        echo "  ⚠ Some gsplat-src example packages failed (may be optional)"
    fi
    
    echo "✓ gsplat-src example dependencies installed"
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
# 11. Final Verification
#=============================================================================
echo "Step 11: Final verification"
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
echo "  OR: conda activate 3dgs"
echo ""
echo "Next steps:"
echo "  1. Upload trained checkpoint to: outputs/<project>/01_gs_base/ckpt_initial.pt"
echo "  2. Upload masks to: outputs/<project>/round_001/masks_<object>/sam_masks/"
echo "  3. Run pipeline: python 04a_lift_masks_to_roi3d.py --ckpt ... --masks_root ..."
echo ""
