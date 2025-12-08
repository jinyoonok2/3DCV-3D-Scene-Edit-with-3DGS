#!/bin/bash
# Create GIF animations for all comparison grids
# Usage: ./create_animations.sh

set -e  # Exit on error

echo "=========================================="
echo "Creating GIF animations for all scenes"
echo "=========================================="

# Garden Scene
echo -e "\n[Garden] Creating animations..."
mkdir -p outputs/garden_brownplant_removal/animations

echo "  - 07 merged..."
ffmpeg -y -framerate 5 -i 'outputs/garden_brownplant_removal/07_final_visualization/merged/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/garden_brownplant_removal/animations/07_merged.gif

echo "  - 07 comparison..."
ffmpeg -y -framerate 5 -i 'outputs/garden_brownplant_removal/07_final_visualization/comparisons/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/garden_brownplant_removal/animations/07_comparison.gif

echo "  - 05c comparison..."
ffmpeg -y -framerate 5 -i 'outputs/garden_brownplant_removal/05_scene_editing/05c_optimized/comparisons/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/garden_brownplant_removal/animations/05c_comparison.gif

# Kitchen Scene
echo -e "\n[Kitchen] Creating animations..."
mkdir -p outputs/kitchen_yellowtracker_removal/animations

echo "  - 07 merged..."
ffmpeg -y -framerate 5 -i 'outputs/kitchen_yellowtracker_removal/07_final_visualization/merged/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/kitchen_yellowtracker_removal/animations/07_merged.gif

echo "  - 07 comparison..."
ffmpeg -y -framerate 5 -i 'outputs/kitchen_yellowtracker_removal/07_final_visualization/comparisons/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/kitchen_yellowtracker_removal/animations/07_comparison.gif

echo "  - 05c comparison..."
ffmpeg -y -framerate 5 -i 'outputs/kitchen_yellowtracker_removal/05_scene_editing/05c_optimized/comparisons/%05d.png' -vf "select='not(mod(n\,2))',scale=960:-1,setpts=N/TB/5" outputs/kitchen_yellowtracker_removal/animations/05c_comparison.gif

echo -e "\n=========================================="
echo "✓ All animations created successfully!"
echo "=========================================="
echo -e "\nGarden animations:"
echo "  outputs/garden_brownplant_removal/animations/"
echo "    - 07_merged.gif"
echo "    - 07_comparison.gif"
echo "    - 05c_comparison.gif"
echo -e "\nKitchen animations:"
echo "  outputs/kitchen_yellowtracker_removal/animations/"
echo "    - 07_merged.gif"
echo "    - 07_comparison.gif"
echo "    - 05c_comparison.gif"
echo ""
