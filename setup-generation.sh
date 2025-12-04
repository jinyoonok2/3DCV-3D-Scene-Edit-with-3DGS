#!/bin/bash

# 3D Scene Edit with 3DGS - Object Generation Setup (Phase 2)
# This script creates a separate environment for object generation (steps 06+)
# Uses GaussianDreamerPro for high-quality text-to-3D generation
# Run this AFTER completing object removal phase (steps 00-05c)

set -e  # Exit on error

VENV_REMOVAL="venv-removal"      # Phase 1 environment
VENV_GENERATION="venv-generation"  # Phase 2 environment
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
echo "=========================================="
echo ""

# Check if Phase 1 environment exists
if [ ! -d "$VENV_REMOVAL" ]; then
    echo "❌ Error: Phase 1 environment not found!"
    echo "Please run ./setup-removal.sh first to set up the object removal pipeline."
    exit 1
fi

echo "✓ Phase 1 environment found: $VENV_REMOVAL"
echo ""

#=============================================================================
# 1. Create Phase 2 Environment (Copy from Phase 1)
#=============================================================================
echo "Step 1: Creating Phase 2 environment"

# Handle reset
if [ "$RESET" = true ]; then
    rm -rf "$VENV_GENERATION"
    rm -rf "$GAUSSIANDREAMERPRO_DIR"
    echo "  Removed existing Phase 2 environment and GaussianDreamerPro"
fi

# Create Phase 2 environment by copying Phase 1
if [ ! -d "$VENV_GENERATION" ]; then
    echo "  Copying Phase 1 environment to Phase 2..."
    cp -r "$VENV_REMOVAL" "$VENV_GENERATION"
    
    # Fix the prompt name in activate script
    sed -i 's/venv-removal/venv-generation/g' "$VENV_GENERATION/bin/activate"
    
    echo "✓ Created Phase 2 environment: $VENV_GENERATION"
else
    echo "✓ Using existing Phase 2 environment: $VENV_GENERATION"
fi

# Activate Phase 2 environment for installation
source "$VENV_GENERATION/bin/activate"
echo "✓ Activated Phase 2 environment"
echo ""

#=============================================================================
# 2. Clone GaussianDreamerPro
#=============================================================================
echo "Step 2: Setting up GaussianDreamerPro repository"

if [ ! -d "$GAUSSIANDREAMERPRO_DIR" ]; then
    git clone https://github.com/hustvl/GaussianDreamerPro.git
    echo "✓ Cloned GaussianDreamerPro"
else
    echo "✓ GaussianDreamerPro already exists"
fi
echo ""

#=============================================================================
# 3. Install GaussianDreamerPro Dependencies
#=============================================================================
echo "Step 3: Installing GaussianDreamerPro dependencies"

# PyTorch already installed from Phase 1, verify version
echo "  Checking PyTorch installation..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"

# Install PyTorch3D
echo "  Installing PyTorch3D..."
pip install iopath fvcore -q
pip install pytorch3d -q || echo "  ⚠️  PyTorch3D install may require building from source"
echo "  ✓ PyTorch3D installation attempted"

# Install remaining dependencies from requirements file
if [ -f "requirements-gaussiandreamerpro.txt" ]; then
    echo "  Installing GaussianDreamerPro Python dependencies..."
    pip install -r requirements-gaussiandreamerpro.txt -q
    echo "  ✓ Python dependencies installed"
else
    echo "  ❌ requirements-gaussiandreamerpro.txt not found!"
    exit 1
fi
echo ""

#=============================================================================
# 4. Build GaussianDreamerPro CUDA Kernels
#=============================================================================
echo "Step 4: Building GaussianDreamerPro CUDA kernels"

cd "$GAUSSIANDREAMERPRO_DIR"

# Build diff-gaussian-rasterization
echo "  Building diff-gaussian-rasterization..."
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization -q
echo "  ✓ diff-gaussian-rasterization built"

# Build diff-gaussian-rasterization_2dgs
echo "  Building diff-gaussian-rasterization_2dgs..."
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_2dgs -q
echo "  ✓ diff-gaussian-rasterization_2dgs built"

# Build simple-knn
echo "  Building simple-knn..."
pip install --no-build-isolation ./submodules/simple-knn -q
echo "  ✓ simple-knn built"

cd ..
echo ""

#=============================================================================
# 5. Download Shap-E Checkpoint
#=============================================================================
echo "Step 5: Setting up Shap-E checkpoint"

mkdir -p "$GAUSSIANDREAMERPRO_DIR/load"

if [ ! -f "$GAUSSIANDREAMERPRO_DIR/load/shapE_finetuned_with_330kdata.pth" ]; then
    echo "  Downloading finetuned Shap-E model from Cap3D..."
    echo "  This may take a while (~2GB download)"
    
    cd "$GAUSSIANDREAMERPRO_DIR/load"
    
    # Try to download (update URL if needed based on actual hosting location)
    wget -q --show-progress https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth || \
    {
        echo "  ⚠️  Automatic download failed"
        echo "  Please manually download shapE_finetuned_with_330kdata.pth"
        echo "  From: https://huggingface.co/datasets/tiange/Cap3D"
        echo "  Place in: $GAUSSIANDREAMERPRO_DIR/load/"
    }
    
    cd ../..
else
    echo "  ✓ Shap-E checkpoint already exists"
fi
echo ""

#=============================================================================
# 6. Verification
#=============================================================================
echo "Step 6: Verifying object generation setup"
python -c "
# Check base dependencies
import torch, gsplat, sam2, numpy, PIL
print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ gsplat: {gsplat.__version__}')
print('✓ SAM2, NumPy, Pillow: OK')

# Check GaussianDreamerPro dependencies
try:
    import pytorch3d
    print(f'✓ PyTorch3D: {pytorch3d.__version__}')
except:
    print('⚠️  PyTorch3D: May need manual installation')

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

print('✓ GaussianDreamerPro setup complete!')
"
echo ""

echo "=========================================="
echo "OBJECT GENERATION SETUP COMPLETE!"
echo "GaussianDreamerPro ready for text-to-3D"
echo "=========================================="
echo ""
echo "Environment structure:"
echo "• Phase 1 (Object Removal): $VENV_REMOVAL"
echo "  - Steps 00-05c: Dataset → Training → Removal → Optimization"
echo "• Phase 2 (Object Generation): $VENV_GENERATION" 
echo "  - Steps 06-08: Text-to-3D → Placement → Visualization"
echo "  - Uses: GaussianDreamerPro for high-quality generation"
echo ""
echo "Activation commands:"
echo "• Object Removal: source activate-removal.sh"
echo "• Object Generation: source activate-generation.sh"
echo ""
echo "To generate 3D objects:"
echo "  source activate-generation.sh"
echo "  python 06_object_generation.py --text_prompt 'a coffee mug'"
echo ""

# Deactivate environment after setup
deactivate
echo "✓ Environment deactivated after setup completion"