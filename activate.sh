#!/bin/bash
# Activate the conda environment for 3DGS Scene Editing project
#
# Usage:
#   source activate.sh

ENV_NAME="3dgs"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found"
    echo ""
    echo "Please install Miniconda/Anaconda first"
    return 1 2>/dev/null || exit 1
fi

# Check if environment exists
if ! conda info --envs | grep -q "^$ENV_NAME "; then
    echo "❌ Conda environment not found: $ENV_NAME"
    echo ""
    echo "Please run setup first:"
    echo "  ./setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "✓ Conda environment activated: $ENV_NAME"
echo ""
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
