#!/usr/bin/env python3
"""
08_place_object_at_roi.py - Place Generated Object at ROI Location

Goal: Transform and merge the generated Gaussian object into the scene at the ROI location.

This module:
1. Loads the ROI (Region of Interest) from Module 04a
2. Loads the generated object Gaussians from Module 07
3. Loads the scene after object removal from Module 05c
4. Transforms the object to fit the ROI (scale, rotation, translation)
5. Merges object Gaussians with scene Gaussians
6. Saves the combined scene for optimization in Module 09

Inputs:
  --object_gaussians: Path to object Gaussians (.pt from Module 07)
  --roi: Path to ROI file (.pt from Module 04a)
  --scene_gaussians: Path to scene after removal (.pt from Module 05c)
  --output: Path to save merged Gaussians

Outputs:
  - merged_gaussians.pt: Combined scene with new object
  - manifest.json: Placement metadata

Strategy for natural placement:
  - Scale object to fit ROI dimensions
  - Position at ROI center (or bottom for grounded objects)
  - Align object upright (rotation)
  - Preserve object's local structure
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

# Import project config
from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Place generated object at ROI location"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--object_gaussians",
        type=str,
        required=True,
        help="Path to object Gaussians (.pt from Module 07)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        required=True,
        help="Path to ROI file (.pt from Module 04a)",
    )
    parser.add_argument(
        "--scene_gaussians",
        type=str,
        required=True,
        help="Path to scene Gaussians after removal (.pt from Module 05c)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for merged Gaussians (.pt file)",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.8,
        help="Scale factor for object size relative to ROI (default: 0.8). Ignored if --no_scale is set.",
    )
    parser.add_argument(
        "--no_scale",
        action="store_true",
        help="Do not auto-scale object to fit ROI. Object keeps original size (or use --manual_scale).",
    )
    parser.add_argument(
        "--manual_scale",
        type=float,
        default=1.0,
        help="Manual uniform scale multiplier when --no_scale is set (default: 1.0)",
    )
    parser.add_argument(
        "--placement",
        type=str,
        choices=["center", "bottom", "top"],
        default="bottom",
        help="Placement strategy: center (middle of ROI), bottom (on surface), top (hanging)",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.0,
        help="Additional Z-axis offset in meters (default: 0.0)",
    )
    parser.add_argument(
        "--rotation_degrees",
        type=float,
        default=0.0,
        help="Additional rotation around Z-axis in degrees (default: 0.0)",
    )
    
    return parser.parse_args()


def load_checkpoint(path, name="checkpoint"):
    """Load checkpoint file."""
    console.print(f"Loading {name}: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu")
        console.print(f"✓ {name} loaded")
        return ckpt
    except Exception as e:
        console.print(f"[red]Error loading {name}: {e}[/red]")
        sys.exit(1)


def get_bounds(positions):
    """Get bounding box of positions."""
    min_bound = positions.min(dim=0)[0]
    max_bound = positions.max(dim=0)[0]
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    return min_bound, max_bound, center, size


def transform_object_to_roi(object_gaussians, roi_mask, scene_positions, args):
    """Transform object Gaussians to fit ROI location and size."""
    console.print("\nTransforming object to ROI...")
    
    # Extract object positions
    obj_positions = object_gaussians["means"]  # [N, 3]
    
    # Get object bounds
    obj_min, obj_max, obj_center, obj_size = get_bounds(obj_positions)
    console.print(f"Object bounds:")
    console.print(f"  Center: {obj_center.numpy()}")
    console.print(f"  Size: {obj_size.numpy()}")
    
    # Get ROI bounds from scene positions with ROI mask
    roi_positions = scene_positions[roi_mask]  # Positions of ROI Gaussians
    roi_min, roi_max, roi_center, roi_size = get_bounds(roi_positions)
    console.print(f"ROI bounds:")
    console.print(f"  Center: {roi_center.numpy()}")
    console.print(f"  Size: {roi_size.numpy()}")
    
    # 1. Center object at origin
    centered_positions = obj_positions - obj_center
    
    # 2. Scale object
    if args.no_scale:
        # Use manual scale only (no ROI-based auto-scaling)
        scale = args.manual_scale
        console.print(f"Manual scale: {scale:.4f} (no ROI auto-scaling)")
    else:
        # Auto-scale to fit ROI
        obj_max_dim = obj_size.max()
        roi_max_dim = roi_size.max()
        scale = (roi_max_dim / obj_max_dim) * args.scale_factor
        console.print(f"Auto-scale to fit ROI: {scale:.4f}")
    
    scaled_positions = centered_positions * scale
    scaled_scales = object_gaussians["scales"] + np.log(scale)  # Scales are in log space
    
    # 3. Apply rotation (optional, around Z-axis)
    if args.rotation_degrees != 0:
        angle_rad = np.deg2rad(args.rotation_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        scaled_positions = scaled_positions @ rot_matrix.T
        console.print(f"Applied rotation: {args.rotation_degrees}°")
    
    # 4. Translate to ROI position
    if args.placement == "bottom":
        # Place object at bottom of ROI (on surface)
        target_center = roi_center.clone()
        target_center[2] = roi_min[2] + (scaled_positions[:, 2].max() - scaled_positions[:, 2].min()) / 2
    elif args.placement == "top":
        # Place object at top of ROI
        target_center = roi_center.clone()
        target_center[2] = roi_max[2] - (scaled_positions[:, 2].max() - scaled_positions[:, 2].min()) / 2
    else:  # center
        # Place object at center of ROI
        target_center = roi_center
    
    # Apply Z offset
    target_center[2] += args.z_offset
    
    final_positions = scaled_positions + target_center
    
    console.print(f"Placement: {args.placement}")
    console.print(f"Target center: {target_center.numpy()}")
    console.print(f"Z offset: {args.z_offset:.4f}m")
    
    # Create transformed Gaussians
    transformed = {
        "means": final_positions,
        "quats": object_gaussians["quats"],  # Rotations stay same (already aligned)
        "scales": scaled_scales,
        "sh0": object_gaussians["sh0"],
        "shN": object_gaussians["shN"] if "shN" in object_gaussians else torch.zeros_like(object_gaussians["sh0"]),
        "opacities": object_gaussians["opacities"],
    }
    
    console.print(f"✓ Object transformed: {len(final_positions)} Gaussians")
    return transformed


def merge_gaussians(scene_gaussians, object_gaussians):
    """Merge object Gaussians with scene Gaussians."""
    console.print("\nMerging object with scene...")
    
    # Concatenate all parameters
    merged = {}
    for key in ["means", "quats", "scales", "sh0", "shN", "opacities"]:
        if key in scene_gaussians and key in object_gaussians:
            # Handle shN which might be empty
            if key == "shN":
                scene_shN = scene_gaussians[key] if scene_gaussians[key].numel() > 0 else torch.zeros(len(scene_gaussians["means"]), 0, 3)
                object_shN = object_gaussians[key] if object_gaussians[key].numel() > 0 else torch.zeros(len(object_gaussians["means"]), 0, 3)
                
                # Make sure both have same number of SH bands
                if scene_shN.shape[1] != object_shN.shape[1]:
                    max_bands = max(scene_shN.shape[1], object_shN.shape[1])
                    if scene_shN.shape[1] < max_bands:
                        scene_shN = torch.cat([
                            scene_shN,
                            torch.zeros(scene_shN.shape[0], max_bands - scene_shN.shape[1], 3)
                        ], dim=1)
                    if object_shN.shape[1] < max_bands:
                        object_shN = torch.cat([
                            object_shN,
                            torch.zeros(object_shN.shape[0], max_bands - object_shN.shape[1], 3)
                        ], dim=1)
                
                merged[key] = torch.cat([scene_gaussians[key], object_shN], dim=0)
            else:
                merged[key] = torch.cat([scene_gaussians[key], object_gaussians[key]], dim=0)
    
    num_scene = len(scene_gaussians["means"])
    num_object = len(object_gaussians["means"])
    num_total = len(merged["means"])
    
    console.print(f"✓ Merged Gaussians:")
    console.print(f"  Scene: {num_scene} Gaussians")
    console.print(f"  Object: {num_object} Gaussians")
    console.print(f"  Total: {num_total} Gaussians")
    
    return merged


def save_merged_scene(merged_gaussians, output_path, metadata):
    """Save merged Gaussian scene."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\nSaving merged scene to {output_path}")
    
    # Save checkpoint
    checkpoint = {
        "gaussians": merged_gaussians,
        "metadata": metadata,
    }
    torch.save(checkpoint, output_path)
    
    # Save metadata JSON
    manifest_path = output_path.parent / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Object placement complete!")
    console.print(f"Output: {output_path}")
    console.print(f"  - Merged scene: {len(merged_gaussians['means'])} Gaussians")
    console.print(f"  - Manifest: {manifest_path}")


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 08: Place Object at ROI")
    console.print("="*80 + "\n")
    
    # Load checkpoints
    object_ckpt = load_checkpoint(args.object_gaussians, "Object Gaussians")
    roi_mask = load_checkpoint(args.roi, "ROI")  # This is just a tensor from Module 04a
    scene_ckpt = load_checkpoint(args.scene_gaussians, "Scene Gaussians")
    
    # Extract Gaussians from checkpoints
    object_gaussians = object_ckpt.get("gaussians", object_ckpt)
    # Scene checkpoint can have "splats" or "gaussians" key
    scene_gaussians = scene_ckpt.get("splats", scene_ckpt.get("gaussians", scene_ckpt))
    
    # roi_mask might be dict with 'roi_binary' or just a tensor
    if isinstance(roi_mask, dict):
        roi_mask = roi_mask.get("roi_binary", roi_mask.get("roi_mask", roi_mask))
    
    # Transform object to ROI
    transformed_object = transform_object_to_roi(object_gaussians, roi_mask, scene_gaussians["means"], args)
    
    # Merge with scene
    merged_gaussians = merge_gaussians(scene_gaussians, transformed_object)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save in round_001 directory
        config = ProjectConfig(args.config)
        project_name = config.get("project", "name") or "garden"
        round_name = config.get("round_name") or "round_001"
        output_path = Path(f"outputs/{project_name}/{round_name}/08_merged_scene.pt")
    
    # Prepare metadata
    metadata = {
        "module": "08_place_object_at_roi",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "parameters": {
            "object_gaussians": str(args.object_gaussians),
            "roi": str(args.roi),
            "scene_gaussians": str(args.scene_gaussians),
            "scale_factor": args.scale_factor,
            "placement": args.placement,
            "z_offset": args.z_offset,
            "rotation_degrees": args.rotation_degrees,
        },
        "statistics": {
            "num_scene_gaussians": len(scene_gaussians["means"]),
            "num_object_gaussians": len(transformed_object["means"]),
            "num_total_gaussians": len(merged_gaussians["means"]),
        },
    }
    
    # Save merged scene
    save_merged_scene(merged_gaussians, output_path, metadata)
    
    console.print(f"\n[cyan]Next step:[/cyan] Run Module 09 to optimize the merged scene:")
    console.print(f"  python 09_final_optimization.py \\")
    console.print(f"    --init_ckpt {output_path} \\")
    console.print(f"    --data_root datasets/360_v2/garden")


if __name__ == "__main__":
    main()
