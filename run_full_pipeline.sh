#!/bin/bash
# Run full pipeline (00-07) with specified config
# Usage: ./run_full_pipeline.sh configs/garden_config.yaml

if [ -z "$1" ]; then
    echo "Error: Config file path required"
    echo "Usage: ./run_full_pipeline.sh <config_path>"
    echo "Example: ./run_full_pipeline.sh configs/garden_config.yaml"
    exit 1
fi

CONFIG=$1

echo "=========================================="
echo "Running full pipeline with config: $CONFIG"
echo "=========================================="

set -e  # Exit on error

echo -e "\n[00] Checking dataset..."
python 00_check_dataset.py --config "$CONFIG"

echo -e "\n[01] Training initial GS..."
python 01_train_gs_initial.py --config "$CONFIG"

echo -e "\n[02] Rendering training views..."
python 02_render_training_views.py --config "$CONFIG"

echo -e "\n[03] Generating masks..."
python 03_ground_text_to_masks.py --config "$CONFIG"

echo -e "\n[04a] Lifting masks to 3D ROI..."
python 04a_lift_masks_to_roi3d.py --config "$CONFIG"

echo -e "\n[04b] Visualizing ROI..."
python 04b_visualize_roi.py --config "$CONFIG"

echo -e "\n[05a] Removing and rendering holes..."
python 05a_remove_and_render_holes.py --config "$CONFIG"

echo -e "\n[05b] Inpainting holes..."
python 05b_inpaint_holes.py --config "$CONFIG"

echo -e "\n[05c] Optimizing to targets..."
python 05c_optimize_to_targets.py --config "$CONFIG"

echo -e "\n[06] Placing object at ROI..."
python 06_place_object_at_roi.py --config "$CONFIG"

echo -e "\n[07] Final visualization..."
python 07_final_visualization.py --config "$CONFIG"

echo -e "\n=========================================="
echo "✓ Pipeline completed successfully!"
echo "=========================================="
