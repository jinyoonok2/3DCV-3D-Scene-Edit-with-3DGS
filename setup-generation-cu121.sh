#!/bin/bash

# 3D Scene Edit with 3DGS - Object Generation Setup
# For VastAI PyTorch templates with CUDA 11.8 (e.g., vastai/pytorch:2.0.1-cuda-11.8.0)
# Uses system Python - no conda/venv needed

set -e  # Exit on error

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
echo "Phase 2: GaussianDreamerPro"
echo "For CUDA 12.1+ systems (PyTorch 2.4.1)"
echo "=========================================="
echo ""

# Verify PyTorch is installed
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || {
    echo "❌ Error: PyTorch not found!"
    echo "This script will install PyTorch 2.4.1+cu121."
}

python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 
echo ""

#=============================================================================
# 1. Clone GaussianDreamerPro Repository
#=============================================================================
echo "Step 1: Setting up GaussianDreamerPro repository"# Handle reset
if [ "$RESET" = true ]; then
    rm -rf "$GAUSSIANDREAMERPRO_DIR"
    echo "  Removed existing repository"
fi

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
    
    # Apply compatibility patches for PyTorch 2.4.1
    chmod +x patch_gaussiandreamerpro.sh
    ./patch_gaussiandreamerpro.sh
else
    echo "✓ GaussianDreamerPro already exists"
fi
echo ""

#=============================================================================
# 2. Install PyTorch 2.4.1 + CUDA 12.1 (before dependencies)
#=============================================================================
echo "Step 2: Installing PyTorch 2.4.1 + CUDA 12.1"

pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
echo "✓ Installed PyTorch 2.4.1"

python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
echo ""

#=============================================================================
# 3. Install Python Dependencies (after PyTorch to prevent upgrades)
#=============================================================================
echo "Step 3: Installing Python dependencies"

if [ -f "requirements-gaussiandreamerpro.txt" ]; then
    # Install dependencies but skip torch packages (already installed)
    grep -v "^torch" requirements-gaussiandreamerpro.txt | pip install -r /dev/stdin
    echo "✓ Python dependencies installed"
else
    echo "❌ requirements-gaussiandreamerpro.txt not found!"
    exit 1
fi
echo ""

#=============================================================================
# 4. Install PyTorch3D (build from source - no wheels for Python 3.12)
#=============================================================================
echo "Step 4: Installing PyTorch3D (this may take a few minutes)"

pip install iopath fvcore
pip install pytorch3d  # Build from source

echo "✓ PyTorch3D installed"
echo ""

#=============================================================================
# 5. Build CUDA Kernels
#=============================================================================
echo "Step 5: Building CUDA kernels"

# Set CUDA_HOME if not already set
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "  CUDA_HOME: $CUDA_HOME"
nvcc --version | head -1

cd "$GAUSSIANDREAMERPRO_DIR"

# Build diff-gaussian-rasterization
echo "  Building diff-gaussian-rasterization..."
CUDA_HOME=$CUDA_HOME pip install ./submodules/diff-gaussian-rasterization
echo "  ✓ diff-gaussian-rasterization built"

# Build diff-gaussian-rasterization_2dgs
echo "  Building diff-gaussian-rasterization_2dgs..."
CUDA_HOME=$CUDA_HOME pip install ./submodules/diff-gaussian-rasterization_2dgs
echo "  ✓ diff-gaussian-rasterization_2dgs built"

# Build simple-knn
echo "  Building simple-knn..."
CUDA_HOME=$CUDA_HOME pip install ./submodules/simple-knn
echo "  ✓ simple-knn built"

cd ..
echo ""

#=============================================================================
# 6. Download Shap-E Checkpoint
#=============================================================================
echo "Step 6: Setting up Shap-E checkpoint"

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
# 7. Verify Installation
#=============================================================================
echo "Step 7: Verifying installation"

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
echo "To generate 3D objects:"
echo "  python 06_object_generation.py"
echo ""
echo "Note: Using system Python environment (no conda/venv needed)"
echo ""
