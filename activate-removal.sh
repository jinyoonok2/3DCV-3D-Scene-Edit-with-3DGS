#!/bin/bash
# Activate the virtual environment for Object Removal Phase (Steps 00-05c)
#
# Usage:
#   source activate-removal.sh

VENV_PATH="./venv-removal"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup-removal.sh"
    return 1 2>/dev/null || exit 1
fi

# Check if activate script exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ Virtual environment activate script not found: $VENV_PATH/bin/activate"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup-removal.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✓ Object Removal environment activated: $VENV_PATH"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "📋 Object Removal Pipeline (Steps 00-05c):"
echo "  00: python 00_check_dataset.py"
echo "  01: python 01_train_gs_initial.py"
echo "  02: python 02_render_training_views.py"
echo "  03: python 03_ground_text_to_masks.py"
echo "  04a: python 04a_lift_masks_to_roi3d.py"
echo "  04b: python 04b_visualize_roi.py"
echo "  05a: python 05a_remove_and_render_holes.py"
echo "  05b: python 05b_inpaint_holes.py"
echo "  05c: python 05c_optimize_to_targets.py"
echo ""
echo "To deactivate, run: deactivate"
