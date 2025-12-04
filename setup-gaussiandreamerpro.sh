#!/bin/bash
# Setup GaussianDreamerPro for text-to-3D generation
# This replaces the threestudio-based GaussianDreamer setup

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_GENERATION="venv-generation"
GAUSSIANDREAMERPRO_DIR="GaussianDreamerPro"

echo "=========================================="
echo "GAUSSIANDREAMERPRO SETUP"
echo "Installing text-to-3D generation framework"
echo "=========================================="
echo ""

# Check if generation environment exists
if [ ! -d "$VENV_GENERATION" ]; then
    echo "❌ Generation environment not found: $VENV_GENERATION"
    echo "Please run ./setup-generation.sh first"
    exit 1
fi

# Activate generation environment
source "$VENV_GENERATION/bin/activate"
echo "✓ Activated generation environment"
echo ""

#=============================================================================
# 1. Clone GaussianDreamerPro
#=============================================================================
echo "Step 1: Cloning GaussianDreamerPro repository"
if [ ! -d "$GAUSSIANDREAMERPRO_DIR" ]; then
    git clone https://github.com/hustvl/GaussianDreamerPro.git
    echo "✓ Cloned GaussianDreamerPro"
else
    echo "✓ GaussianDreamerPro already exists"
fi
echo ""

#=============================================================================
# 2. Check PyTorch (already installed in venv-generation)
#=============================================================================
echo "Step 2: Checking PyTorch installation"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} already installed')"
echo ""

#=============================================================================
# 3. Install PyTorch3D
#=============================================================================
echo "Step 3: Installing PyTorch3D"
# Install dependencies for pytorch3d
pip install iopath fvcore

# Install pytorch3d (compatible with PyTorch 2.5.1)
# Try from conda-forge or build from source
pip install pytorch3d || echo "⚠️  PyTorch3D install may require building from source"

echo "✓ PyTorch3D installation attempted"
echo ""

#=============================================================================
# 4. Install Python dependencies
#=============================================================================
echo "Step 4: Installing Python dependencies"
pip install -r requirements-gaussiandreamerpro.txt
echo "✓ Python dependencies installed"
echo ""

#=============================================================================
# 5. Build CUDA kernels
#=============================================================================
echo "Step 5: Building CUDA kernels"

cd "$GAUSSIANDREAMERPRO_DIR"

# Build diff-gaussian-rasterization
echo "  Building diff-gaussian-rasterization..."
pip install ./submodules/diff-gaussian-rasterization

# Build diff-gaussian-rasterization_2dgs
echo "  Building diff-gaussian-rasterization_2dgs..."
pip install ./submodules/diff-gaussian-rasterization_2dgs

# Build simple-knn
echo "  Building simple-knn..."
pip install ./submodules/simple-knn

cd "$SCRIPT_DIR"
echo "✓ CUDA kernels built"
echo ""

#=============================================================================
# 6. Download Shap-E checkpoint
#=============================================================================
echo "Step 6: Downloading Shap-E checkpoint"

mkdir -p "$GAUSSIANDREAMERPRO_DIR/load"

# Check if checkpoint already exists
if [ ! -f "$GAUSSIANDREAMERPRO_DIR/load/shapE_finetuned_with_330kdata.pth" ]; then
    echo "Downloading finetuned Shap-E model from Cap3D..."
    echo "This may take a while (~2GB download)"
    
    # Download from Hugging Face (if available) or provide instructions
    cd "$GAUSSIANDREAMERPRO_DIR/load"
    
    # Try to download (you may need to update this URL based on actual location)
    wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth || \
    echo "⚠️  Please manually download shapE_finetuned_with_330kdata.pth"
    echo "   Download from: https://huggingface.co/datasets/tiange/Cap3D"
    echo "   Place in: $GAUSSIANDREAMERPRO_DIR/load/"
    
    cd "$SCRIPT_DIR"
else
    echo "✓ Shap-E checkpoint already exists"
fi
echo ""

#=============================================================================
# 7. Verification
#=============================================================================
echo "Step 7: Verifying installation"
python -c "
import torch
import pytorch3d
print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ PyTorch3D: {pytorch3d.__version__}')

try:
    import diff_gaussian_rasterization
    print('✓ diff-gaussian-rasterization: OK')
except:
    print('❌ diff-gaussian-rasterization: Failed')
    
try:
    from simple_knn._C import distCUDA2
    print('✓ simple-knn: OK')
except:
    print('❌ simple-knn: Failed')
"
echo ""

echo "=========================================="
echo "GAUSSIANDREAMERPRO SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To generate 3D assets from text:"
echo "  python 06_object_generation.py --text_prompt 'a coffee mug'"
echo ""
echo "Or edit config.yaml and run:"
echo "  python 06_object_generation.py"
echo ""
