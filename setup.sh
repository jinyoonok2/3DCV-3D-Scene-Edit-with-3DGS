#!/bin/bash
# Setup script for 3D Gaussian Splatting Scene Editing
# Requires: CUDA-capable GPU, Python 3.10/3.12, CUDA toolkit matching PyTorch version
# Tested on: Ubuntu 20.04+, Vast.ai GPU instances
#
# Usage:
#   ./setup.sh          - Install dependencies (creates venv if needed)
#   ./setup.sh --reset  - Remove existing venv and reinstall from scratch

set -e  # Exit on error

VENV_NAME="venv"
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
# 1. Python Environment Setup (venv)
#=============================================================================
echo "Step 1: Setting up Python virtual environment"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "✗ Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "  Using Python $PYTHON_VERSION"

# Handle reset
if [ "$RESET" = true ]; then
    rm -rf "$VENV_NAME"
    echo "  Removed existing virtual environment"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    echo "✓ Created virtual environment: $VENV_NAME"
else
    echo "✓ Using existing virtual environment: $VENV_NAME"
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"
echo "✓ Activated virtual environment"
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
# 4. PyTorch with CUDA
#=============================================================================
echo "Step 4: Installing PyTorch with CUDA support"

# Install PyTorch with CUDA support
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'none')")
    echo "✓ PyTorch $TORCH_VER (CUDA: $TORCH_CUDA) already installed"
else
    echo "  Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "✓ PyTorch installed"
fi

# Verify CUDA
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
else
    echo "⚠ CUDA not available - using CPU mode"
fi
echo ""

#=============================================================================
# 5. gsplat (pip package)
#=============================================================================
echo "Step 5: Installing gsplat from pip"
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
if [ -d "gsplat-src" ]; then
    echo "  Removing existing gsplat-src to get original version..."
    rm -rf gsplat-src
fi

echo "  Cloning fresh gsplat-src repository..."
git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src -q
echo "✓ gsplat-src cloned (original version)"

# Create symlink for datasets module access
if [ ! -L "datasets" ]; then
    ln -sf gsplat-src/examples/datasets datasets
    echo "✓ datasets symlink created"
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
# 8. gsplat-src Dependencies (for object removal pipeline)
#=============================================================================
echo "Step 8: Installing gsplat-src dependencies"
if [ -f "requirements-gsplat.txt" ]; then
    echo "  Installing gsplat example dependencies..."
    pip install -r requirements-gsplat.txt
    echo "✓ gsplat-src dependencies installed"
fi

# Install fused-ssim separately (optional performance optimization)
echo "  Installing fused-ssim (performance optimization)..."
if pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5; then
    echo "✓ fused-ssim installed successfully"
else
    echo "⚠ fused-ssim installation failed (CUDA compilation issues) - continuing without it"
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

# Install gsplat dependencies
if [ -f "requirements-gsplat.txt" ]; then
    echo "  Installing gsplat example dependencies..."
    pip install -r requirements-gsplat.txt
    echo "✓ gsplat-src example dependencies installed"
fi

# Install fused-ssim separately (optional performance optimization)
echo "  Installing fused-ssim (performance optimization)..."
if pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5; then
    echo "✓ fused-ssim installed successfully"
else
    echo "⚠ fused-ssim installation failed (CUDA compilation issues) - continuing without it"
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
# 11. Download MipNeRF-360 Dataset (Optional)
#=============================================================================
echo "Step 11: MipNeRF-360 dataset download"
if [ ! -d "datasets/360_v2" ] || [ -z "$(ls -A datasets/360_v2 2>/dev/null)" ]; then
    echo "MipNeRF-360 dataset not found. Would you like to download it?"
    echo "This will download all 7 scenes (~10 GB total)"
    echo ""
    read -p "Download MipNeRF-360 dataset? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        chmod +x download_datasets.sh
        ./download_datasets.sh
    else
        echo "Skipping dataset download. You can download later with:"
        echo "  ./download_datasets.sh"
    fi
else
    echo "✓ MipNeRF-360 dataset already exists"
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
echo "OBJECT REMOVAL SETUP COMPLETE!"
echo "Ready for steps 00-05c (scene editing)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run scripts 00-05c for object removal"
echo "2. After completing removal, run ./setup_replacement.sh"
echo "   to add object generation capabilities (06+)"
echo ""
echo "Activate environment:"
echo "  source activate.sh"
echo "  OR: source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Upload trained checkpoint to: outputs/<project>/01_gs_base/ckpt_initial.pt"
echo "  2. Upload masks to: outputs/<project>/round_001/masks_<object>/sam_masks/"
echo "  3. Run pipeline: python 04a_lift_masks_to_roi3d.py --ckpt ... --masks_root ..."
echo ""
