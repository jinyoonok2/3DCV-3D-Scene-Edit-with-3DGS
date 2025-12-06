#!/bin/bash
# Setup script for 3D Gaussian Splatting Scene Editing
# Requires: CUDA-capable GPU, Python 3.10+, CUDA toolkit
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
# 2. Upgrade pip
#=============================================================================
echo "Step 2: Upgrading pip"
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

#=============================================================================
# 3. GPU Verification
#=============================================================================
echo "Step 3: Verifying GPU"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU Information:"
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
echo "  PyTorch CUDA Status:"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'  ✓ Device: {torch.cuda.get_device_name(0)}'); print(f'  ✓ CUDA: {torch.version.cuda}'); print(f'  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
else
    echo "  ⚠ CUDA not available - using CPU mode"
fi
echo ""

#=============================================================================
# 5. Install Project Dependencies
#=============================================================================
echo "Step 5: Installing project dependencies"
if [ -f "requirements.txt" ]; then
    echo "  This may take a few minutes..."
    pip install -r requirements.txt
    echo "✓ Project dependencies installed"
else
    echo "✗ requirements.txt not found"
    exit 1
fi
echo ""

#=============================================================================
# 6. Clone Required Repositories
#=============================================================================
echo "Step 6: Setting up external repositories"

# gsplat-src contains examples and dataset parsers
if [ -d "gsplat-src" ]; then
    echo "  ✓ gsplat-src already exists"
else
    echo "  Cloning gsplat-src repository..."
    git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src -q
    echo "  ✓ gsplat-src cloned"
fi

# Ensure datasets directory exists
if [ ! -d "datasets" ]; then
    mkdir -p datasets
    echo "  ✓ datasets directory created"
else
    echo "  ✓ datasets directory exists"
fi
echo ""

#=============================================================================
# 7. Download Resources
#=============================================================================
echo "Step 7: Downloading models and datasets"
echo ""

if [ -f "download.sh" ]; then
    chmod +x download.sh
    ./download.sh all
else
    echo "⚠ download.sh not found, skipping downloads"
fi
echo ""

#=============================================================================
# 8. Verify Installation
#=============================================================================
echo "Step 8: Verifying key packages"

python -c "import gsplat; print(f'✓ gsplat: {gsplat.__version__}')" 2>/dev/null || echo "✗ gsplat import failed"
python -c "import torch; print(f'✓ torch: {torch.__version__}')" 2>/dev/null || echo "✗ torch import failed"
python -c "from groundingdino.util.inference import load_model; print('✓ groundingdino')" 2>/dev/null || echo "✗ groundingdino import failed"
python -c "from sam2.build_sam import build_sam2; print('✓ sam2')" 2>/dev/null || echo "✗ sam2 import failed"
python -c "from simple_lama_inpainting import SimpleLama; print('✓ simple-lama-inpainting')" 2>/dev/null || echo "✗ simple-lama-inpainting import failed"
echo ""

#=============================================================================
# 9. Setup Complete
#=============================================================================
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source activate.sh"
echo ""
echo "2. Run the pipeline:"
echo "   python 00_check_dataset.py"
echo "   python 01_train_gs_initial.py"
echo "   ... and so on"
echo ""
echo "Pipeline modules (00-07):"
echo "  00: Dataset validation"
echo "  01: Train initial 3DGS"
echo "  02: Render training views"
echo "  03: Generate masks (GroundingDINO + SAM2)"
echo "  04a: Lift masks to 3D ROI"
echo "  04b: Visualize ROI (optional)"
echo "  05a: Remove object and render holes"
echo "  05b: Inpaint holes (LaMa)"
echo "  05c: Optimize to targets"
echo "  06: Place object at ROI (external Gaussians)"
echo "  07: Final visualization"
echo ""
