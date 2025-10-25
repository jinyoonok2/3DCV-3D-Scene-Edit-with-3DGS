#!/bin/bash
# Script to download GroundingDINO and SAM2 model weights

echo "Downloading GroundingDINO and SAM2 model weights..."
echo "=================================================="

# Create models directory
mkdir -p models
cd models

echo ""
echo "Downloading GroundingDINO weights..."
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    echo "✓ GroundingDINO SwinT-OGC downloaded"
else
    echo "✓ GroundingDINO already exists"
fi

echo ""
echo "Downloading SAM2 weights (large model)..."
if [ ! -f "sam2_hiera_large.pt" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    echo "✓ SAM2 Large downloaded"
else
    echo "✓ SAM2 Large already exists"
fi

cd ..

echo ""
echo "Cloning GroundingDINO repository (for config files)..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    echo "✓ GroundingDINO repository cloned"
else
    echo "✓ GroundingDINO repository already exists"
fi

echo ""
echo "Optional: Download smaller SAM2 models for faster inference"
echo "Uncomment the following lines if needed:"
echo ""
echo "# SAM2 Tiny (fastest)"
echo "# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
echo ""
echo "# SAM2 Small"
echo "# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
echo ""
echo "# SAM2 Base Plus"
echo "# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo "Model weights: ./models/"
echo "Config files: ./GroundingDINO/"
echo ""
echo "Files:"
ls -lh models/
echo ""
echo "Total size:"
du -sh models/
echo ""
echo "Next: Run 03_ground_text_to_masks.py (no additional setup needed)"
