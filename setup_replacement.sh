#!/bin/bash

# 3D Scene Edit with 3DGS - Object Generation Setup (Phase 2)
# This script extends the existing environment with TripoSR dependencies
# Run this AFTER completing object removal phase (steps 00-05c)

set -e  # Exit on error

echo "=========================================="
echo "3D SCENE EDIT - OBJECT GENERATION SETUP"
echo "Phase 2: Adding object generation capabilities (steps 06+)"
echo "=========================================="
echo ""

# Check if we're in the right environment
if ! python -c "import gsplat, sam2" 2>/dev/null; then
    echo "❌ Error: Base environment not found!"
    echo "Please run ./setup.sh first to set up the object removal pipeline."
    exit 1
fi

echo "✓ Base environment detected"
echo ""

#=============================================================================
# 1. TripoSR Repository Setup
#=============================================================================
echo "Step 1: Setting up TripoSR repository"

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
# 2. TripoSR Dependencies
#=============================================================================
echo "Step 2: Installing TripoSR dependencies"
if [ -f "requirements-triposr.txt" ]; then
    echo "  This may upgrade/replace some existing packages for compatibility..."
    pip install -r requirements-triposr.txt
    echo "✓ TripoSR dependencies installed"
else
    echo "❌ requirements-triposr.txt not found!"
    exit 1
fi
echo ""

#=============================================================================
# 3. Verification
#=============================================================================
echo "Step 3: Verifying object generation setup"
python -c "
# Check base dependencies still work
import torch, gsplat, sam2, numpy, PIL
print(f'✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ gsplat: {gsplat.__version__}')
print('✓ SAM2, NumPy, Pillow: OK')

# Check TripoSR
try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    print('✓ TripoSR: Available')
except ImportError as e:
    print(f'❌ TripoSR import failed: {e}')
    exit(1)
"
echo ""

echo "=========================================="
echo "OBJECT GENERATION SETUP COMPLETE!"
echo "Full pipeline now ready (steps 00-09)"
echo "=========================================="
echo ""
echo "Available workflows:"
echo "• Object Removal (00-05c): ✓ Ready"  
echo "• Object Generation (06-09): ✓ Ready"
echo ""
echo "You can now run the complete 3D scene editing pipeline!"
echo ""