#!/bin/bash
# Script to unzip outputs archive downloaded from Vast.ai

echo "========================================"
echo "Unzipping outputs archive"
echo "========================================"
echo ""

# Check if argument provided
if [ $# -eq 0 ]; then
    # No argument, look for most recent outputs_*.zip
    ZIP_FILE=$(ls -t outputs_*.zip 2>/dev/null | head -n 1)
    
    if [ -z "$ZIP_FILE" ]; then
        echo "ERROR: No outputs_*.zip file found!"
        echo ""
        echo "Usage:"
        echo "  $0                    - Unzip most recent outputs_*.zip"
        echo "  $0 <zip_file>         - Unzip specific file"
        echo ""
        exit 1
    fi
    
    echo "Found most recent archive: ${ZIP_FILE}"
else
    # Argument provided
    ZIP_FILE="$1"
    
    if [ ! -f "$ZIP_FILE" ]; then
        echo "ERROR: File not found: ${ZIP_FILE}"
        exit 1
    fi
fi

echo "Archive: ${ZIP_FILE}"
echo "Size: $(du -h ${ZIP_FILE} | cut -f1)"
echo ""

# Ask for confirmation if outputs folder exists
if [ -d "outputs" ]; then
    echo "⚠️  WARNING: outputs/ folder already exists!"
    echo ""
    read -p "Overwrite existing outputs? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
    echo ""
fi

# Unzip the archive
echo "Extracting..."
unzip -q -o "${ZIP_FILE}"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "Extracted to: outputs/"
echo ""

# Show what was extracted
if [ -d "outputs" ]; then
    echo "Contents:"
    ls -lh outputs/ | tail -n +2
    echo ""
    echo "Total size: $(du -sh outputs/ | cut -f1)"
fi
echo ""
