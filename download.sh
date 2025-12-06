#!/bin/bash
# Unified download script for models, datasets, and GaussianDreamer results
# Usage: ./download.sh [models|datasets|gdrive|all]

set -e

DOWNLOAD_MODELS=false
DOWNLOAD_DATASETS=false
DOWNLOAD_GDRIVE=false

# Parse arguments
if [ $# -eq 0 ] || [ "$1" == "all" ]; then
    DOWNLOAD_MODELS=true
    DOWNLOAD_DATASETS=true
    DOWNLOAD_GDRIVE=true
elif [ "$1" == "models" ]; then
    DOWNLOAD_MODELS=true
elif [ "$1" == "datasets" ]; then
    DOWNLOAD_DATASETS=true
elif [ "$1" == "gdrive" ]; then
    DOWNLOAD_GDRIVE=true
else
    echo "Usage: ./download.sh [models|datasets|gdrive|all]"
    echo ""
    echo "  models    - Download model weights only"
    echo "  datasets  - Download MipNeRF-360 dataset only"
    echo "  gdrive    - Download GaussianDreamerResults from Google Drive"
    echo "  all       - Download everything (default)"
    exit 1
fi

echo "=========================================="
echo "3DGS Scene Editing - Download Resources"
echo "=========================================="
echo ""

#=============================================================================
# Download Model Weights
#=============================================================================
if [ "$DOWNLOAD_MODELS" = true ]; then
    echo "📦 Checking model weights..."
    echo ""
    
    mkdir -p models
    
    # GroundingDINO weights
    if [ -f "models/groundingdino_swint_ogc.pth" ]; then
        echo "  ✓ GroundingDINO already exists"
    else
        echo "  ⬇ Downloading GroundingDINO (~662 MB)..."
        wget -q --show-progress -P models \
            https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
        echo "  ✓ GroundingDINO downloaded"
    fi
    
    # SAM2 weights
    if [ -f "models/sam2_hiera_large.pt" ]; then
        echo "  ✓ SAM2 Large already exists"
    else
        echo "  ⬇ Downloading SAM2 Large (~857 MB)..."
        wget -q --show-progress -P models \
            https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
        echo "  ✓ SAM2 Large downloaded"
    fi
    
    # GroundingDINO repository (for config files)
    if [ -d "GroundingDINO" ]; then
        echo "  ✓ GroundingDINO repo already exists"
    else
        echo "  ⬇ Cloning GroundingDINO repository..."
        git clone -q https://github.com/IDEA-Research/GroundingDINO.git
        echo "  ✓ GroundingDINO repo cloned"
    fi
    
    echo ""
    echo "✅ Model weights ready!"
    echo "   Location: ./models/"
    echo "   Size: $(du -sh models/ 2>/dev/null | cut -f1)"
    echo ""
    echo "   Additional models (auto-download on first use):"
    echo "   - LaMa (~200 MB) → ~/.cache/torch/hub/"
    echo "   - SDXL (~6 GB) → ~/.cache/huggingface/"
    echo ""
fi

#=============================================================================
# Download MipNeRF-360 Dataset
#=============================================================================
if [ "$DOWNLOAD_DATASETS" = true ]; then
    echo "📦 Checking MipNeRF-360 dataset..."
    echo ""
    
    mkdir -p datasets
    
    # Check if any scene exists
    SCENES_EXIST=false
    if [ -d "datasets/360_v2" ]; then
        for scene in bicycle bonsai counter garden kitchen room stump; do
            if [ -d "datasets/360_v2/$scene" ]; then
                SCENES_EXIST=true
                break
            fi
        done
    fi
    
    if [ "$SCENES_EXIST" = true ]; then
        echo "  ✓ Dataset scenes found:"
        for scene in bicycle bonsai counter garden kitchen room stump; do
            if [ -d "datasets/360_v2/$scene" ]; then
                echo "    ✓ $scene"
            else
                echo "    ✗ $scene (missing)"
            fi
        done
        echo ""
        echo "  To re-download, delete datasets/360_v2/ first"
    else
        echo "  ⬇ Downloading MipNeRF-360 dataset (~2.85 GB compressed)..."
        echo "     This may take 5-15 minutes"
        echo ""
        
        cd datasets
        
        # Download
        wget -q --show-progress http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
        
        echo ""
        echo "  📦 Extracting dataset..."
        unzip -q 360_v2.zip
        
        # Organize into 360_v2 directory
        mkdir -p 360_v2
        for scene in bicycle bonsai counter garden kitchen room stump; do
            [ -d "$scene" ] && mv "$scene" 360_v2/ && echo "    ✓ $scene"
        done
        
        rm 360_v2.zip
        cd ..
        
        echo ""
        echo "  ✅ Dataset ready!"
    fi
    
    echo ""
    echo "   Location: ./datasets/360_v2/"
    echo "   Scenes: 7 (bicycle, bonsai, counter, garden, kitchen, room, stump)"
    echo "   Size: $(du -sh datasets/360_v2/ 2>/dev/null | cut -f1 || echo 'N/A')"
    echo ""
fi

#=============================================================================
# Download GaussianDreamerResults from Google Drive
#=============================================================================
if [ "$DOWNLOAD_GDRIVE" = true ]; then
    echo "📦 Checking GaussianDreamerResults from Google Drive..."
    echo ""
    
    FOLDER_ID="1_HPWBbc_t0YOq-kekFm7JoHMNXHxb5AP"
    TARGET_DIR="GaussianDreamerResults"
    
    if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR 2>/dev/null)" ]; then
        echo "  ✓ GaussianDreamerResults already exists"
        echo "    Files: $(find $TARGET_DIR -type f | wc -l)"
    else
        # Check if gdown is installed
        if ! command -v gdown &> /dev/null; then
            echo "  ⬇ Installing gdown..."
            pip install -q gdown
        fi
        
        echo "  ⬇ Downloading GaussianDreamerResults from Google Drive..."
        gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${TARGET_DIR}" --remaining-ok
        
        echo ""
        echo "  ✅ GaussianDreamerResults downloaded!"
    fi
    
    echo ""
    echo "   Location: ./$TARGET_DIR/"
    echo "   Files: $(find $TARGET_DIR -type f 2>/dev/null | wc -l || echo '0')"
    echo ""
fi

#=============================================================================
# Summary
#=============================================================================
echo "=========================================="
echo "✅ Download Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Setup environment: ./setup.sh"
echo "  2. Activate: source activate.sh"
echo "  3. Run pipeline: python 00_check_dataset.py"
echo ""
