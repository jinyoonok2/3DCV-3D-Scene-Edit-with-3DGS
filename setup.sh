#!/bin/bash
# Setup script for 3D Gaussian Splatting Scene Editing using Conda
# Requires: CUDA-capable GPU, conda/mamba, CUDA toolkit
# Tested on: Ubuntu 20.04+, Vast.ai GPU instances
#
# Usage:
#   ./setup.sh          - Install dependencies (creates conda env if needed)
#   ./setup.sh --reset  - Remove existing conda env and reinstall from scratch

set -e  # Exit on error

ENV_NAME="3dgs-scene-edit"
RESET=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --reset|-r) RESET=true; shift ;;
        *) echo "Usage: ./setup.sh [--reset]"; exit 1 ;;
    esac
done

echo "=========================================="
echo "3DGS Scene Editing - Conda Setup"
echo "=========================================="
echo ""

#=============================================================================
# 1. Check for Conda
#=============================================================================
echo "Step 1: Checking for Conda/Mamba"

if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Found mamba (faster than conda)"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✓ Found conda"
else
    echo "✗ Error: Neither conda nor mamba found"
    echo ""
    echo "Please install Miniconda or Anaconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "Or install Mamba (recommended, faster):"
    echo "  conda install -n base -c conda-forge mamba"
    exit 1
fi
echo ""

#=============================================================================
# 2. Handle Reset
#=============================================================================
if [ "$RESET" = true ]; then
    echo "Step 2: Removing existing environment"
    if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        $CONDA_CMD env remove -n "$ENV_NAME" -y
        echo "✓ Removed existing environment: $ENV_NAME"
    else
        echo "  No existing environment to remove"
    fi
    echo ""
fi

#=============================================================================
# 3. Create/Update Conda Environment
#=============================================================================
echo "Step 3: Creating/Updating conda environment from configs/environment.yml"

if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' exists, updating..."
    $CONDA_CMD env update -n "$ENV_NAME" -f configs/environment.yml --prune
    echo "✓ Environment updated"
else
    echo "  Creating new environment '$ENV_NAME'..."
    $CONDA_CMD env create -f configs/environment.yml
    echo "✓ Environment created"
fi
echo ""

#=============================================================================
# 4. Clone Required Repositories
#=============================================================================
echo "Step 4: Setting up external repositories"

# Activate environment for the rest of setup
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate "$ENV_NAME"

# gsplat-src contains examples and dataset parsers (NOT for installation)
if [ -d "gsplat-src" ]; then
    echo "  Removing existing gsplat-src to get original version..."
    rm -rf gsplat-src
fi

echo "  Cloning fresh gsplat-src repository..."
git clone https://github.com/nerfstudio-project/gsplat.git gsplat-src -q
echo "✓ gsplat-src cloned (original version)"

# Ensure datasets directory exists (for MipNeRF-360 data)
if [ ! -d "datasets" ]; then
    mkdir -p datasets
    echo "✓ datasets directory created"
else
    echo "✓ datasets directory exists"
fi
echo ""

#=============================================================================
# 5. Download Model Weights and Datasets
#=============================================================================
echo "Step 5: Downloading required resources"
echo ""

# Run unified download script for both models and datasets
if [ -f "download.sh" ]; then
    chmod +x download.sh
    ./download.sh all
else
    echo "⚠ download.sh not found, skipping downloads"
    echo "  Run './download.sh all' manually to download resources"
fi
echo ""

#=============================================================================
# 6. GPU Verification
#=============================================================================
echo "Step 6: Verifying GPU and PyTorch CUDA"

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    echo ""
fi

echo "  PyTorch CUDA Status:"
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}') if torch.cuda.is_available() else print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

#=============================================================================
# 7. Verify Installation
#=============================================================================
echo "Step 7: Verifying key packages"

python -c "import gsplat; print(f'✓ gsplat: {gsplat.__version__}')" 2>/dev/null || echo "✗ gsplat import failed"
python -c "import torch; print(f'✓ torch: {torch.__version__}')" 2>/dev/null || echo "✗ torch import failed"
python -c "from groundingdino.util.inference import load_model; print('✓ groundingdino')" 2>/dev/null || echo "✗ groundingdino import failed"
python -c "from sam2.build_sam import build_sam2; print('✓ sam2')" 2>/dev/null || echo "✗ sam2 import failed"
python -c "from simple_lama_inpainting import SimpleLama; print('✓ simple-lama-inpainting')" 2>/dev/null || echo "✗ simple-lama-inpainting import failed"
echo ""

#=============================================================================
# 8. Setup Complete
#=============================================================================
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download resources (if not done yet):"
echo "   ./download.sh all          # Models + datasets"
echo "   ./download.sh models       # Models only (~1.5 GB)"
echo "   ./download.sh datasets     # Dataset only (~10 GB)"
echo ""
echo "2. Activate the environment:"
echo "   source activate.sh"
echo "   # or: conda activate $ENV_NAME"
echo ""
echo "3. Run the pipeline:"
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
