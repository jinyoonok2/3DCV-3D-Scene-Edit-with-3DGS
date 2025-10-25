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

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Zip filename (with timestamp to avoid overwriting)
ZIP_NAME="outputs_${TIMESTAMP}.zip"

echo "Creating archive: ${ZIP_NAME}"
echo ""

# Zip the outputs folder
# -r: recursive
# -q: quiet (less verbose)
# Exclude some large/unnecessary files:
#   - .git folders
#   - __pycache__
#   - *.pyc
#   - tb/ (tensorboard logs can be large)
zip -r "${ZIP_NAME}" outputs \
    -x "outputs/*/.git/*" \
    -x "outputs/*/__pycache__/*" \
    -x "outputs/*/*.pyc" \
    -x "outputs/*/tb/*" \
    2>&1 | grep -v "adding:" || true

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
echo ""
echo "Archive created: ${ZIP_NAME}"
echo "Size: $(du -h ${ZIP_NAME} | cut -f1)"
echo ""
echo "To download from Vast.ai:"
echo "  scp root@<vast-ip>:~/3DCV-3D-Scene-Edit-with-3DGS/${ZIP_NAME} ."
echo ""
echo "Or use Vast.ai web interface:"
echo "  1. Go to your instance page"
echo "  2. Click 'Files' tab"
echo "  3. Navigate to /root/3DCV-3D-Scene-Edit-with-3DGS/"
echo "  4. Download ${ZIP_NAME}"
echo ""
