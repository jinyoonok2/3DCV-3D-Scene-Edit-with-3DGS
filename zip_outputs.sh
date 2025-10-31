#!/bin/bash
# Script to zip outputs folder for download from Vast.ai

echo "========================================"
echo "Zipping outputs folder"
echo "========================================"
echo ""

# Check if outputs folder exists
if [ ! -d "outputs" ]; then
    echo "ERROR: outputs folder not found!"
    exit 1
fi

# Zip filename
ZIP_NAME="outputs.zip"

echo "Creating archive: ${ZIP_NAME}"
echo ""

# Zip the contents of outputs folder (without outputs/ wrapper)
# -r: recursive
# -q: quiet (less verbose)
# Exclude some large/unnecessary files:
#   - .git folders
#   - __pycache__
#   - *.pyc
#   - tb/ (tensorboard logs can be large)
cd outputs
zip -r "../${ZIP_NAME}" . \
    -x "*/.git/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x "*/tb/*" \
    2>&1 | grep -v "adding:" || true
cd ..

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "Archive created: ${ZIP_NAME}"
echo "Size: $(du -h ${ZIP_NAME} | cut -f1)"
echo ""
echo "To download from Vast.ai:"
echo "  scp root@<vast-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/${ZIP_NAME} ."
echo ""
echo "Or use Vast.ai web interface:"
echo "  1. Go to your instance page"
echo "  2. Click 'Files' tab"
echo "  3. Navigate to /workspace/3DCV-3D-Scene-Edit-with-3DGS/"
echo "  4. Download ${ZIP_NAME}"
echo ""
echo "To extract locally:"
echo "  ./unzip_outputs.sh"
echo ""
