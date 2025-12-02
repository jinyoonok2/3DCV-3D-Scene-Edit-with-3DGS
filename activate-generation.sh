#!/bin/bash
# Activate the Phase 2 virtual environment for object generation (steps 06+)
#
# Usage:
#   source activate-generation.sh

VENV_PATH="./venv-gen"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Phase 2 virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please run Phase 2 setup first:"
    echo "  ./setup_replacement.sh"
    return 1 2>/dev/null || exit 1
fi

# Check if activate script exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ Phase 2 virtual environment activate script not found: $VENV_PATH/bin/activate"
    echo ""
    echo "Please run Phase 2 setup first:"
    echo "  ./setup_replacement.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✓ Phase 2 virtual environment activated: $VENV_PATH"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "Available for: Object Generation (steps 06-09)"
echo "  06: Generate objects from text"
echo "  07: Convert mesh to Gaussians"  
echo "  08: Place object at ROI"
echo "  09: Final visualization"
echo ""
echo "To deactivate, run: deactivate"