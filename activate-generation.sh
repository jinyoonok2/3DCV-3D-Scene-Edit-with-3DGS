#!/bin/bash

# Activate GaussianDreamerPro environment
# Uses system Python (no conda/venv needed for VastAI templates)

echo "✓ Using system Python for GaussianDreamerPro"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  (PyTorch not installed yet - run setup-generation.sh first)"
echo ""
echo "Available modules:"
echo "  06 - Object Generation (GaussianDreamerPro text-to-3D)"
echo "  07 - Object Placement (merge into scene)"
echo "  08 - Final Visualization (render results)"
echo ""
echo "Example usage:"
echo "  python 06_object_generation.py"
echo ""
