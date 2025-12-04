#!/bin/bash

# Patch GaussianDreamerPro CUDA code for PyTorch 2.5+ and C++17 compatibility
# - Replaces deprecated .data<T>() with .data_ptr<T>()
# - Adds missing <cstdint> header for uint32_t, uint64_t types

set -e

RASTERIZE_FILE="GaussianDreamerPro/submodules/diff-gaussian-rasterization/rasterize_points.cu"
RASTERIZER_IMPL_H="GaussianDreamerPro/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"

echo "Patching GaussianDreamerPro for PyTorch 2.5+ and C++17 compatibility..."

# 1. Fix rasterize_points.cu - Replace deprecated .data<T>() with .data_ptr<T>()
if [ ! -f "$RASTERIZE_FILE" ]; then
    echo "❌ Error: $RASTERIZE_FILE not found!"
    exit 1
fi

sed -i 's/\.data<float>()/\.data_ptr<float>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<int>()/\.data_ptr<int>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<bool>()/\.data_ptr<bool>()/g' "$RASTERIZE_FILE"
sed -i 's/\.data<uint32_t>()/\.data_ptr<uint32_t>()/g' "$RASTERIZE_FILE"
echo "✓ Patched $RASTERIZE_FILE (deprecated API)"

# 2. Fix rasterizer_impl.h - Add missing <cstdint> header
if [ ! -f "$RASTERIZER_IMPL_H" ]; then
    echo "❌ Error: $RASTERIZER_IMPL_H not found!"
    exit 1
fi

# Add #include <cstdint> after the first #include line if not already present
if ! grep -q "#include <cstdint>" "$RASTERIZER_IMPL_H"; then
    sed -i '1a #include <cstdint>' "$RASTERIZER_IMPL_H"
    echo "✓ Patched $RASTERIZER_IMPL_H (added <cstdint>)"
else
    echo "✓ $RASTERIZER_IMPL_H already has <cstdint>"
fi

echo "✓ All patches applied successfully"
