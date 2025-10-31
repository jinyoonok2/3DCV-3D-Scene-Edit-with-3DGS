#!/bin/bash
# Activate the virtual environment for 3DGS Scene Editing project
#
# Usage:
#   source activate.sh

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "✓ Virtual environment activated: $VENV_DIR"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
