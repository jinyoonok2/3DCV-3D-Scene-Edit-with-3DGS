#!/bin/bash

# Activate conda environment for GaussianDreamerPro (Python 3.8 + PyTorch 2.0.1)
# This environment is separate from Phase 1 (venv-removal)

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found!"
    echo "Please install Miniconda or Anaconda first."
    return 1 2>/dev/null || exit 1
fi

# Initialize conda for current shell
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate gaussiandreamerpro

echo "✓ Activated GaussianDreamerPro environment (Python 3.8 + PyTorch 2.0.1)"
echo ""
echo "Available modules:"
echo "  06 - Object Generation (GaussianDreamerPro text-to-3D)"
echo "  07 - Object Placement (merge into scene)"
echo "  08 - Final Visualization (render results)"
echo ""
echo "Example usage:"
echo "  python 06_object_generation.py"
echo ""
