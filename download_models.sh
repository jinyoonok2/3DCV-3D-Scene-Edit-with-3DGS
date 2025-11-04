#!/bin/bash
# Script to download GroundingDINO and SAM2 model weights
# Can be run independently or called by setup.sh

echo "=========================================="
echo "Downloading Model Weights"
echo "=========================================="
echo ""
echo "This will download:"
echo "  - GroundingDINO SwinT-OGC (~662 MB)"
echo "  - SAM2 Hiera Large (~857 MB)"
echo "  Total: ~1.5 GB"
echo ""
echo "Note: Stable Diffusion models (SDXL-Inpainting) will be"
echo "automatically downloaded by diffusers on first run (~6 GB)"
echo ""

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
echo "Additional models (auto-downloaded on first use):"
echo "  - SDXL-Inpainting: ~/.cache/huggingface/ (~6 GB)"
echo "  - TripoSR: ~/.cache/huggingface/ (~1.5 GB)"
echo "  - Depth Anything v2: ~/.cache/huggingface/ (~1.3 GB)"
echo ""
echo "Total additional downloads on first run: ~9 GB"
echo ""
echo "Next steps:"
echo "  1. Modules 00-04: Use GroundingDINO + SAM2 (ready to use)"
echo "  2. Module 05: Will download SDXL-Inpainting on first run"
echo "  3. Module 06: Will download TripoSR on first run"
echo "  4. Module 07: Will download Depth Anything v2 on first run"
