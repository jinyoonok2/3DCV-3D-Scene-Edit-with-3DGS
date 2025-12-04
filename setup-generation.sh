#!/bin/bash

# 3D Scene Edit with 3DGS - Object Generation Setup (Phase 2)
# This script creates a separate conda environment for object generation (steps 06+)
# Uses GaussianDreamerPro for high-quality text-to-3D generation
# Follows official GaussianDreamerPro installation with Python 3.8 + PyTorch 2.0.1
# Run this AFTER completing object removal phase (steps 00-05c)

set -e  # Exit on error

CONDA_ENV="gaussiandreamerpro"
GAUSSIANDREAMERPRO_DIR="GaussianDreamerPro"
RESET=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --reset|-r) RESET=true; shift ;;
        *) echo "Usage: ./setup-generation.sh [--reset]"; exit 1 ;;
    esac
done

echo "=========================================="
echo "3D SCENE EDIT - OBJECT GENERATION SETUP"
echo "Phase 2: Object generation with GaussianDreamerPro"
echo "Following official setup: Python 3.8 + PyTorch 2.0.1 + CUDA 11.8"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found!"
    echo "Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

#=============================================================================
# 1. Create Conda Environment (Python 3.8)
#=============================================================================
echo "Step 1: Setting up conda environment"

# Handle reset
if [ "$RESET" = true ]; then
    conda env remove -n "$CONDA_ENV" -y 2>/dev/null || true
    rm -rf "$GAUSSIANDREAMERPRO_DIR"
    echo "  Removed existing environment and repository"
fi

# Create conda environment
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "  Creating conda environment: $CONDA_ENV (Python 3.8)..."
    conda create -n "$CONDA_ENV" python=3.8 -y
    echo "✓ Created conda environment: $CONDA_ENV"
else
    echo "✓ Conda environment already exists: $CONDA_ENV"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate "$CONDA_ENV"
echo "✓ Activated conda environment: $CONDA_ENV"
echo ""

#=============================================================================
# 2. Clone GaussianDreamerPro
#=============================================================================
echo "Step 2: Setting up GaussianDreamerPro repository"

if [ ! -d "$GAUSSIANDREAMERPRO_DIR" ]; then
    git clone https://github.com/hustvl/GaussianDreamerPro.git
    echo "✓ Cloned GaussianDreamerPro"
    
    # Install GLM library (missing from repository)
    cd "$GAUSSIANDREAMERPRO_DIR/submodules/diff-gaussian-rasterization/third_party"
    if [ ! -d "glm" ]; then
        git clone https://github.com/g-truc/glm.git
        echo "✓ Installed GLM library"
    fi
    cd ../../../..
else
    echo "✓ GaussianDreamerPro already exists"
fi
echo ""

#=============================================================================
# 3. Install Python Dependencies (WITHOUT PyTorch)
#=============================================================================
echo "Step 3: Installing Python dependencies (except PyTorch)"

if [ -f "requirements-gaussiandreamerpro.txt" ]; then
    # Install dependencies but skip torch-related packages (install them later)
    grep -v "^torch" requirements-gaussiandreamerpro.txt | pip install -r /dev/stdin
    echo "✓ Python dependencies installed"
else
    echo "❌ requirements-gaussiandreamerpro.txt not found!"
    exit 1
fi
echo ""

#=============================================================================
# 4. Install PyTorch 2.0.1 + CUDA 11.8 (Official Version - LAST)
#=============================================================================
echo "Step 4: Installing PyTorch 2.0.1 + CUDA 11.8 (forcing correct version)"

pip install --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
echo "✓ Installed PyTorch 2.0.1 with CUDA 11.8"

python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

#=============================================================================
# 5. Install PyTorch3D (Pre-built Wheels)
#=============================================================================
echo "Step 5: Installing PyTorch3D from pre-built wheels"

conda install -c iopath iopath -y
conda install -c fvcore -c conda-forge fvcore -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
echo "✓ Installed PyTorch3D"
echo ""

#=============================================================================
# 6. Build CUDA Kernels (No Patches Needed)
#=============================================================================
echo "Step 6: Building CUDA kernels"

cd "$GAUSSIANDREAMERPRO_DIR"

# Build diff-gaussian-rasterization
echo "  Building diff-gaussian-rasterization..."
pip install ./submodules/diff-gaussian-rasterization
echo "  ✓ diff-gaussian-rasterization built"

# Build diff-gaussian-rasterization_2dgs
echo "  Building diff-gaussian-rasterization_2dgs..."
pip install ./submodules/diff-gaussian-rasterization_2dgs
echo "  ✓ diff-gaussian-rasterization_2dgs built"

# Build simple-knn
echo "  Building simple-knn..."
pip install ./submodules/simple-knn
echo "  ✓ simple-knn built"

cd ..
echo ""

#=============================================================================
# 7. Download Shap-E Checkpoint
#=============================================================================
echo "Step 7: Setting up Shap-E checkpoint"

mkdir -p "$GAUSSIANDREAMERPRO_DIR/load"

if [ ! -f "$GAUSSIANDREAMERPRO_DIR/load/shapE_finetuned_with_330kdata.pth" ]; then
    echo "  Downloading finetuned Shap-E model from HuggingFace..."
    echo "  This may take a while (~2GB download)"
    
    cd "$GAUSSIANDREAMERPRO_DIR/load"
    
    # Try wget first, fall back to curl
    if command -v wget &> /dev/null; then
        wget -O shapE_finetuned_with_330kdata.pth \
            "https://huggingface.co/camenduru/GaussianDreamer/resolve/main/shapE_finetuned_with_330kdata.pth"
    elif command -v curl &> /dev/null; then
        curl -L -o shapE_finetuned_with_330kdata.pth \
            "https://huggingface.co/camenduru/GaussianDreamer/resolve/main/shapE_finetuned_with_330kdata.pth"
    else
        echo "  ❌ Neither wget nor curl found!"
        echo "  Please download manually from:"
        echo "  https://huggingface.co/camenduru/GaussianDreamer/resolve/main/shapE_finetuned_with_330kdata.pth"
        echo "  Save to: $GAUSSIANDREAMERPRO_DIR/load/shapE_finetuned_with_330kdata.pth"
        cd ../..
        exit 1
    fi
    
    cd ../..
    echo "  ✓ Downloaded Shap-E checkpoint"
else
    echo "  ✓ Shap-E checkpoint already exists"
fi
echo ""

#=============================================================================
# 8. Verify Installation
#=============================================================================
echo "Step 8: Verifying installation"

python -c "
import torch
import pytorch3d
print('✓ PyTorch:', torch.__version__)
print('✓ PyTorch3D:', pytorch3d.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
"

echo ""
echo "=========================================="
echo "✓ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  conda activate $CONDA_ENV"
echo "  python 06_object_generation.py"
echo ""
echo "Note: This is a separate conda environment from Phase 1 (venv-removal)"
echo "Phase 1 uses Python 3.12 + PyTorch 2.5.1"
echo "Phase 2 uses Python 3.8 + PyTorch 2.0.1 (for GaussianDreamerPro compatibility)"
echo ""
    
    # Try to download (update URL if needed based on actual hosting location)
