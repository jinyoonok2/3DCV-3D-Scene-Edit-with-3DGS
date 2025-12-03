#!/bin/bash
# Activate the virtual environment for Object Generation Phase (Steps 06-09)
#
# Usage:
#   source activate-generation.sh

VENV_PATH="./venv-generation"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup-generation.sh"
    return 1 2>/dev/null || exit 1
fi

# Check if activate script exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ Virtual environment activate script not found: $VENV_PATH/bin/activate"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup-generation.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✓ Object Generation environment activated: $VENV_PATH"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "📋 Object Generation Pipeline (Steps 06-08):"
echo "  06: python 06_object_generation.py"
echo "  07: python 07_place_object_at_roi.py"
echo "  08: python 08_final_visualization.py"
echo ""
echo "💡 Prerequisites: Complete Object Removal phase (steps 00-05c) first"
echo ""
echo "To deactivate, run: deactivate"