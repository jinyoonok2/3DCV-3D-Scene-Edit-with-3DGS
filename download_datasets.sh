#!/bin/bash
# Script to download MipNeRF-360 dataset
# Downloads the complete dataset from official Google Research repository

echo "=========================================="
echo "MipNeRF-360 Dataset Downloader"
echo "=========================================="
echo ""
echo "This will download the complete MipNeRF-360 dataset:"
echo "  - All 7 scenes: bicycle, bonsai, counter, garden, kitchen, room, stump"
echo "  - Total download size: ~2.85 GB (compressed)"
echo "  - Extracted size: ~10 GB"
echo ""
echo "Estimated time: 5-15 minutes depending on connection"
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
mkdir -p datasets
cd datasets

# Check if dataset already exists
if [ -d "360_v2" ] && [ "$(ls -A 360_v2 2>/dev/null)" ]; then
    echo "✓ MipNeRF-360 dataset already exists in datasets/360_v2/"
    echo "Scenes found:"
    ls 360_v2/ 2>/dev/null | while read scene; do
        if [ -d "360_v2/$scene" ]; then
            echo "  ✓ $scene"
        fi
    done
    echo ""
    echo "To re-download, delete the datasets/360_v2/ folder first"
    cd ..
    exit 0
fi

# Download the complete dataset
echo "Downloading MipNeRF-360 dataset..."
DATASET_URL="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

if wget -q --show-progress "$DATASET_URL"; then
    echo "✓ Download complete"
    echo ""
    echo "Extracting dataset..."
    if unzip -q "360_v2.zip"; then
        rm "360_v2.zip"
        echo "✓ Extraction complete"
    else
        echo "❌ Failed to extract 360_v2.zip"
        cd ..
        exit 1
    fi
else
    echo "❌ Failed to download dataset"
    cd ..
    exit 1
fi

# Return to project root
cd ..

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "All MipNeRF-360 scenes downloaded to:"
echo "  datasets/360_v2/"
echo ""
echo "Available scenes:"
if [ -d "datasets/360_v2" ]; then
    ls datasets/360_v2/ 2>/dev/null | while read scene; do
        if [ -d "datasets/360_v2/$scene" ]; then
            echo "  ✓ $scene"
        fi
    done
else
    echo "  ❌ No scenes found"
fi
echo ""
echo "You can now run the pipeline with any scene:"
echo "  python init_project.py --scene garden"
echo "  python 00_check_dataset.py"
echo ""