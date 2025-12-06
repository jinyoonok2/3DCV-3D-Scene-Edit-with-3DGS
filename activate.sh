#!/bin/bash
# Activate the conda environment for 3DGS Scene Editing
#
# Usage:
#   source activate.sh

ENV_NAME="3dgs-scene-edit"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found"
    echo ""
    echo "Please install Miniconda or Anaconda first"
    return 1 2>/dev/null || exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ Environment '$ENV_NAME' not found"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate environment
conda activate "$ENV_NAME"

echo "✓ Environment activated: $ENV_NAME"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "📋 3DGS Scene Editing Pipeline:"
echo ""
echo "  Object Removal (Steps 00-05c):"
echo "    00: python 00_check_dataset.py"
echo "    01: python 01_train_gs_initial.py"
echo "    02: python 02_render_training_views.py"
echo "    03: python 03_ground_text_to_masks.py"
echo "    04a: python 04a_lift_masks_to_roi3d.py"
echo "    04b: python 04b_visualize_roi.py  # Optional"
echo "    05a: python 05a_remove_and_render_holes.py"
echo "    05b: python 05b_inpaint_holes.py"
echo "    05c: python 05c_optimize_to_targets.py"
echo ""
echo "  Object Placement (Steps 06-07):"
echo "    06: python 06_place_object_at_roi.py"
echo "    07: python 07_final_visualization.py"
echo ""
echo "To deactivate, run: conda deactivate"
