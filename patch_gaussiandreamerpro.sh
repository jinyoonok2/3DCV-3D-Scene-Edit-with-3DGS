#!/bin/bash

# Patch GaussianDreamerPro CUDA code for PyTorch 2.5+ compatibility
# Replaces deprecated .data<T>() with .data_ptr<T>()

set -e

RASTERIZE_FILE="GaussianDreamerPro/submodules/diff-gaussian-rasterization/rasterize_points.cu"

if [ ! -f "$RASTERIZE_FILE" ]; then
    echo "❌ Error: $RASTERIZE_FILE not found!"
    exit 1
fi

echo "Patching $RASTERIZE_FILE for PyTorch 2.5+ compatibility..."

# Replace all occurrences of .data<TYPE>() with .data_ptr<TYPE>()
sed -i 's/\.data<float>()/\.data_ptr<float>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<int>()/\.data_ptr<int>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<bool>()/\.data_ptr<bool>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<uint32_t>()/\.data_ptr<uint32_t>()/g' "$RASTERIZE_FILE"

echo "✓ Patched $RASTERIZE_FILE"
echo "  Replaced deprecated .data<T>() with .data_ptr<T>()"
