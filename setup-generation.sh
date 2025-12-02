#!/bin/bash

# 3D Scene Edit with 3DGS - Object Generation Setup (Phase 2)
# This script creates a separate environment for object generation (steps 06+)
# Run this AFTER completing object removal phase (steps 00-05c)

set -e  # Exit on error

VENV_REMOVAL="venv-removal"      # Phase 1 environment
VENV_GENERATION="venv-generation"  # Phase 2 environment
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
echo "Phase 2: Creating separate environment for object generation (steps 06+)"
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
    echo "  Removed existing Phase 2 environment"
fi

# Create Phase 2 environment by copying Phase 1
if [ ! -d "$VENV_GENERATION" ]; then
    echo "  Copying Phase 1 environment to Phase 2..."
    cp -r "$VENV_REMOVAL" "$VENV_GENERATION"
    echo "✓ Created Phase 2 environment: $VENV_GENERATION"
else
    echo "✓ Using existing Phase 2 environment: $VENV_GENERATION"
fi

# Activate Phase 2 environment
source "$VENV_GENERATION/bin/activate"
echo "✓ Activated Phase 2 environment"
echo ""

#=============================================================================
# 2. TripoSR Repository Setup
#=============================================================================
echo "Step 2: Setting up TripoSR repository"

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
# 3. TripoSR Dependencies
#=============================================================================
echo "Step 3: Installing TripoSR dependencies"
if [ -f "requirements-triposr.txt" ]; then
    echo "  Installing TripoSR dependencies in separate environment..."
    pip install -r requirements-triposr.txt
    echo "✓ TripoSR dependencies installed"
else
    echo "❌ requirements-triposr.txt not found!"
    exit 1
fi
echo ""

#=============================================================================
# 4. Verification
#=============================================================================
echo "Step 4: Verifying object generation setup"
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
echo "Two-environment setup ready"
echo "=========================================="
echo ""
echo "Environment structure:"
echo "• Phase 1 (Object Removal): $VENV_REMOVAL"
echo "  - Steps 00-05c: Dataset → Training → Removal → Optimization"
echo "• Phase 2 (Object Generation): $VENV_GENERATION" 
echo "  - Steps 06-09: Generation → Placement → Optimization → Visualization"
echo ""
echo "Activation commands:"
echo "• Object Removal: source $VENV_REMOVAL/bin/activate"
echo "• Object Generation: source $VENV_GENERATION/bin/activate"
echo ""
echo "Or use the activate scripts:"
echo "• source activate-removal.sh      # Phase 1 (object removal)"
echo "• source activate-generation.sh   # Phase 2 (object generation)"
echo ""