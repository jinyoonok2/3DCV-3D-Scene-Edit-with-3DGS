#!/bin/bash
# Setup script for 3DGS Scene Editing project
# Requires CUDA 12.6+ (enforced)
# Works on local machines, cloud GPU instances (Vast.ai, etc.)
#
# Usage:
#   ./setup.sh          - Install dependencies (creates venv if needed)
#   ./setup.sh --reset  - Remove existing venv and reinstall from scratch

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
            echo "Usage: ./setup.sh [--reset]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "3DGS Scene Editing - Environment Setup"
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

# Install system dependencies for Pillow
echo "5. Installing system dependencies..."
if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
    echo "  Installing Pillow build dependencies..."
    sudo apt-get update -qq && sudo apt-get install -y -qq \
        libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev \
        liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev libxcb1-dev
    echo "✓ System dependencies installed"
else
    echo "⚠ Skipping system dependencies (no sudo access)"
    echo "  If Pillow installation fails, run:"
    echo "  sudo apt-get install -y libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev libxcb1-dev"
fi
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Install PyTorch with CUDA 12.6
echo "6. Installing PyTorch with CUDA 12.6..."
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
echo "7. Verifying CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ CUDA is available'); print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  CUDA Version: {torch.version.cuda}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

# Install gsplat from PyPI
echo "8. Installing gsplat..."
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
echo "9. Setting up gsplat examples repository..."
if [ ! -d "gsplat-src" ]; then
    echo "  Cloning gsplat repository..."
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src
    echo "✓ gsplat repository cloned"
else
    echo "✓ gsplat-src already exists"
fi
echo ""

# Install project requirements
echo "10. Installing project requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Project requirements installed"
else
    echo "⚠ No requirements.txt found"
fi
echo ""

# Clone GroundingDINO repository for config files
echo "11. Setting up GroundingDINO..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    echo "✓ GroundingDINO repository cloned (for config files)"
else
    echo "✓ GroundingDINO repository already exists"
fi
echo ""

# Install SAM2 from GitHub
echo "12. Installing SAM2..."
if python -c "import sam2" 2>/dev/null; then
    echo "✓ SAM2 already installed"
else
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    echo "✓ SAM2 installed"
fi
echo ""

# Clone TripoSR repository and install dependencies (Module 06)
echo "13. Setting up TripoSR..."
if [ ! -d "TripoSR" ]; then
    echo "  Cloning TripoSR repository..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git
    
    # Copy tsr source to project root for imports
    cp -r TripoSR/tsr ./
    
    # Create __init__.py files for Python module
    touch tsr/__init__.py
    touch tsr/models/__init__.py
    
    # Install TripoSR dependencies (combined for speed)
    echo "  Installing TripoSR dependencies..."
    pip install rembg==2.0.59 xatlas==0.0.9 trimesh onnxruntime einops omegaconf
    pip install git+https://github.com/tatsy/torchmcubes.git
    
    echo "✓ TripoSR repository cloned and dependencies installed"
else
    echo "✓ TripoSR repository already exists"
    if [ ! -d "tsr" ]; then
        echo "  Copying tsr source..."
        cp -r TripoSR/tsr ./
        touch tsr/__init__.py
        touch tsr/models/__init__.py
    fi
    # Create __init__.py if missing
    if [ ! -f "tsr/__init__.py" ]; then
        touch tsr/__init__.py
        touch tsr/models/__init__.py
    fi
    # Verify imports
    if ! python -c "from tsr.system import TSR" 2>/dev/null; then
        echo "  Installing missing dependencies..."
        pip install rembg==2.0.59 xatlas==0.0.9 trimesh onnxruntime einops omegaconf
        pip install git+https://github.com/tatsy/torchmcubes.git
    fi
fi
echo ""

# Download model weights
echo "14. Downloading model weights..."
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
echo "15. Final verification..."
echo "  Testing imports..."
python -c "
import torch
import gsplat
import sam2
import numpy as np
import PIL
print('✓ All core packages imported successfully')
print(f'  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  gsplat: {gsplat.__version__}')
print('  SAM2: OK')
"
echo ""

# Download dataset (optional - can be slow)
echo "16. Checking for dataset..."
if [ ! -d "datasets/360_v2/garden" ]; then
    echo "⚠ Dataset not found. To download (~2.5GB), run:"
    echo "  cd gsplat-src/examples"
    echo "  python datasets/download_dataset.py --dataset mipnerf360"
    echo "  cd ../.."
    echo "  mkdir -p datasets/360_v2"
    echo "  cp -r gsplat-src/examples/data/360_v2/garden datasets/360_v2/garden"
    echo ""
    echo "  Skipping automatic download to save time."
else
    echo "✓ Dataset already exists at: datasets/360_v2/garden"
fi
echo ""

echo "=========================================="
echo "Environment Setup Complete!"
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
echo "Then run Module 04a:"
echo "  source activate.sh"
echo "  python 04a_lift_masks_to_roi3d.py \\"
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
