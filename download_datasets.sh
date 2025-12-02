#!/bin/bash
# Script to download MipNeRF-360 dataset
# Downloads all 7 scenes from the official dataset

echo "=========================================="
echo "MipNeRF-360 Dataset Downloader"
echo "=========================================="
echo ""
echo "This will download all 7 MipNeRF-360 scenes:"
echo "  - bicycle    (~2.0 GB)"
echo "  - bonsai     (~1.0 GB)"
echo "  - counter    (~1.2 GB)"
echo "  - garden     (~1.1 GB)"
echo "  - kitchen    (~1.3 GB)"
echo "  - room       (~1.4 GB)"
echo "  - stump      (~1.6 GB)"
echo ""
echo "Total download size: ~10 GB"
echo "Estimated time: 10-30 minutes depending on connection"
echo ""

# Ask for confirmation
read -p "Continue with download? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Create datasets directory
echo "Creating datasets directory..."
mkdir -p datasets/360_v2
cd datasets/360_v2

# Base URL for MipNeRF-360 dataset
BASE_URL="http://storage.googleapis.com/gresearch/refraw360/360_v2"

# Dataset scenes
SCENES=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")

# Download each scene
for scene in "${SCENES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Downloading scene: $scene"
    echo "----------------------------------------"
    
    if [ -d "$scene" ]; then
        echo "✓ $scene already exists, skipping..."
        continue
    fi
    
    # Download and extract
    echo "Downloading $scene.zip..."
    if wget -q --show-progress "${BASE_URL}/${scene}.zip"; then
        echo "Extracting $scene.zip..."
        if unzip -q "${scene}.zip"; then
            rm "${scene}.zip"
            echo "✓ $scene downloaded and extracted successfully"
        else
            echo "❌ Failed to extract $scene.zip"
            exit 1
        fi
    else
        echo "❌ Failed to download $scene.zip"
        exit 1
    fi
done

# Return to project root
cd ../..

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "All MipNeRF-360 scenes downloaded to:"
echo "  datasets/360_v2/"
echo ""
echo "Available scenes:"
for scene in "${SCENES[@]}"; do
    if [ -d "datasets/360_v2/$scene" ]; then
        echo "  ✓ $scene"
    else
        echo "  ❌ $scene (failed)"
    fi
done
echo ""
echo "You can now run the pipeline with any scene:"
echo "  python init_project.py --scene garden"
echo "  python 00_check_dataset.py"
echo ""