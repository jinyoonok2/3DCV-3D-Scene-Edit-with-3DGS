#!/bin/bash
# Activate the virtual environment for 3DGS Scene Editing project
#
# Usage:
#   source activate.sh

VENV_PATH="./venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Check if activate script exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ Virtual environment activate script not found: $VENV_PATH/bin/activate"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✓ Virtual environment activated: $VENV_PATH"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
