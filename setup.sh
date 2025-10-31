#!/bin/bash
# Universal setup script for 3DGS Scene Editing project
# Works on any Linux environment (local, WSL, Vast.ai, etc.)
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

# Verify Python and CUDA
echo "4. Verifying environment..."
python --version
echo ""

# Check if torch is installed and verify CUDA
if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed:"
    python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    echo ""
else
    echo "⚠ PyTorch not found. Installing PyTorch with CUDA 12.6 support..."
    echo "  This may take several minutes..."
    pip install torch torchvision torchaudio
    echo "✓ PyTorch installed"
    echo ""
fi

# Install gsplat
echo "5. Installing gsplat..."
if python -c "import gsplat" 2>/dev/null; then
    echo "✓ gsplat already installed"
else
    echo "  Compiling CUDA kernels (this may take a few minutes)..."
    pip install gsplat
    echo "✓ gsplat installed"
fi
echo ""

# Clone gsplat repository for examples
echo "6. Setting up gsplat examples..."
if [ ! -d "gsplat-src" ]; then
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src
    echo "✓ gsplat repository cloned"
else
    echo "✓ gsplat-src already exists"
fi
echo ""

# Install project requirements (includes dependencies for gsplat examples)
echo "7. Installing project requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Project requirements installed"
else
    echo "⚠ No requirements.txt found"
fi
echo ""

# Install gsplat examples requirements (after PyTorch is confirmed)
echo "8. Installing gsplat examples requirements..."
if [ -f "gsplat-src/examples/requirements.txt" ]; then
    # fused-ssim needs torch during build, so use --no-build-isolation
    echo "  Note: Using --no-build-isolation for packages that need torch during build"
    pip install -r gsplat-src/examples/requirements.txt --no-build-isolation
    echo "✓ gsplat examples requirements installed"
fi
echo ""

# Install SAM2 from GitHub
echo "9. Installing SAM2..."
if python -c "import sam2" 2>/dev/null; then
    echo "✓ SAM2 already installed"
else
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    echo "✓ SAM2 installed"
fi
echo ""

# Clone GroundingDINO repository for config files
echo "10. Setting up GroundingDINO..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    echo "✓ GroundingDINO repository cloned (for config files)"
else
    echo "✓ GroundingDINO repository already exists"
fi
echo ""

# Download model weights
echo "11. Downloading model weights..."
if [ ! -f "models/groundingdino_swint_ogc.pth" ] || [ ! -f "models/sam2_hiera_large.pt" ]; then
    chmod +x download_models.sh
    ./download_models.sh
else
    echo "✓ Model weights already downloaded"
    echo "  To re-download, run: ./download_models.sh"
fi
echo ""

# Download dataset (optional)
echo "12. Checking for dataset..."
if [ ! -d "datasets/360_v2/garden" ]; then
    read -p "Download Mip-NeRF 360 garden dataset (~2.5GB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading dataset..."
        cd gsplat-src/examples
        python datasets/download_dataset.py --dataset mipnerf360
        cd ../..
        
        echo "Moving dataset to project directory..."
        mkdir -p datasets/360_v2
        cp -r gsplat-src/examples/data/360_v2/garden datasets/360_v2/garden
        echo "✓ Dataset downloaded and moved"
        
        echo "Validating dataset..."
        python 00_check_dataset.py --data_root datasets/360_v2/garden --factor 4
    else
        echo "⊘ Dataset download skipped"
        echo "  To download later: cd gsplat-src/examples && python datasets/download_dataset.py --dataset mipnerf360"
    fi
else
    echo "✓ Dataset already exists at: datasets/360_v2/garden"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate the environment, run:"
echo "  source activate.sh"
echo ""
echo "To start training:"
echo "  source activate.sh"
echo "  python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000"
echo ""
echo "For help with any module:"
echo "  python <module_name>.py --help"
echo ""
echo "=========================================="
