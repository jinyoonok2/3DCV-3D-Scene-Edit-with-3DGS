#!/usr/bin/env python3
"""
06_place_object_at_roi.py - Place Generated Object at ROI Location

Goal: Transform and merge the generated Gaussian object into the scene at the ROI location.

This module:
1. Loads the ROI (Region of Interest) from Module 04a
2. Loads the pre-generated object Gaussians (from external notebook/PLY file)
3. Loads the scene after object removal from Module 05c
4. Transforms the object to fit the ROI (scale, rotation, translation)
5. Merges object Gaussians with scene Gaussians
6. Saves the combined scene for visualization in Module 07

Inputs:
  --object_gaussians: Path to object Gaussians (.pt or .ply from external generation)
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
        default="configs/garden_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--object_gaussians",
        type=str,
        help="Path to object Gaussians (.pt from Module 06). If not provided, uses config.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        help="Path to ROI file (.pt from Module 04a). If not provided, uses config.",
    )
    parser.add_argument(
        "--original_scene",
        type=str,
        help="Path to original scene checkpoint for ROI positions (default: infer from config)",
    )
    parser.add_argument(
        "--scene_gaussians",
        type=str,
        help="Path to scene Gaussians after removal (.pt from Module 05a or 05c). If not provided, uses config.",
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
    """Load checkpoint file (.pt or .ply)."""
    console.print(f"Loading {name}: {path}")
    try:
        # Check file extension
        if path.endswith('.ply'):
            # Load PLY file manually
            from plyfile import PlyData
            import numpy as np
            
            plydata = PlyData.read(path)
            vertex = plydata['vertex']
            
            # Extract positions (xyz)
            positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
            
            # Extract other properties if available
            gaussians = {
                "means": torch.from_numpy(positions).float(),
            }
            
            # Try to extract scales (scale_0, scale_1, scale_2)
            if 'scale_0' in vertex:
                scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
                gaussians["scales"] = torch.from_numpy(scales).float()
            
            # Try to extract rotations (rot_0, rot_1, rot_2, rot_3 as quaternions)
            if 'rot_0' in vertex:
                quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
                gaussians["quats"] = torch.from_numpy(quats).float()
            
            # Try to extract opacities
            if 'opacity' in vertex:
                gaussians["opacities"] = torch.from_numpy(np.array(vertex['opacity'])).float().unsqueeze(-1)
            
            # Try to extract spherical harmonics (SH coefficients)
            # Standard format: f_dc_0, f_dc_1, f_dc_2, f_rest_0, f_rest_1, ...
            if 'f_dc_0' in vertex:
                sh0 = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
                gaussians["sh0"] = torch.from_numpy(sh0).float()
                
                # Try to get rest of SH coefficients
                sh_rest_names = [name for name in vertex.data.dtype.names if name.startswith('f_rest_')]
                if sh_rest_names:
                    sh_rest = np.stack([vertex[name] for name in sorted(sh_rest_names)], axis=1)
                    # Reshape to (N, num_sh_rest, 3)
                    num_rest = len(sh_rest_names) // 3
                    gaussians["shN"] = torch.from_numpy(sh_rest).float().reshape(-1, num_rest, 3)
            
            console.print(f"✓ {name} loaded from PLY ({len(positions)} Gaussians)")
            return gaussians
        else:
            # Load PyTorch checkpoint
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            console.print(f"✓ {name} loaded")
            return ckpt
    except Exception as e:
        console.print(f"[red]Error loading {name}: {e}[/red]")
        import traceback
        traceback.print_exc()
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
    
    # Convert ROI mask to boolean if needed
    if roi_mask.dtype not in [torch.bool, torch.uint8]:
        roi_mask = roi_mask.bool()
    
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
    
    # Debug: print shapes
    console.print("Scene Gaussian shapes:")
    for k in ["means", "quats", "scales", "sh0", "shN", "opacities"]:
        if k in scene_gaussians:
            console.print(f"  {k}: {scene_gaussians[k].shape}")
    console.print("Object Gaussian shapes:")
    for k in ["means", "quats", "scales", "sh0", "shN", "opacities"]:
        if k in object_gaussians:
            console.print(f"  {k}: {object_gaussians[k].shape}")
    
    # Concatenate all parameters
    merged = {}
    for key in ["means", "quats", "scales", "sh0", "shN", "opacities"]:
        if key in scene_gaussians and key in object_gaussians:
            scene_val = scene_gaussians[key]
            object_val = object_gaussians[key]
            
            # Fix shape mismatches
            if key == "sh0":
                # Scene: [N, 1, 3], Object might be: [N, 3]
                if scene_val.ndim == 3 and object_val.ndim == 2:
                    object_val = object_val.unsqueeze(1)  # [N, 3] -> [N, 1, 3]
                elif scene_val.ndim == 2 and object_val.ndim == 3:
                    scene_val = scene_val.unsqueeze(1)
            
            elif key == "opacities":
                # Scene might be: [N], Object might be: [N, 1]
                if scene_val.ndim == 1 and object_val.ndim == 2:
                    object_val = object_val.squeeze(-1)  # [N, 1] -> [N]
                elif scene_val.ndim == 2 and object_val.ndim == 1:
                    scene_val = scene_val.squeeze(-1)
            
            # Handle shN which might have different number of bands
            elif key == "shN":
                scene_shN = scene_val if scene_val.numel() > 0 else torch.zeros(len(scene_gaussians["means"]), 0, 3)
                object_shN = object_val if object_val.numel() > 0 else torch.zeros(len(object_gaussians["means"]), 0, 3)
                
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
                
                merged[key] = torch.cat([scene_shN, object_shN], dim=0)
                continue
            
            merged[key] = torch.cat([scene_val, object_val], dim=0)
    
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
    
    console.print(f"\n[green]✓[/green] Object placement complete!")
    console.print(f"Output: {output_path}")
    console.print(f"  - Merged scene: {len(merged_gaussians['means'])} Gaussians")


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 06: Place Object at ROI")
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    placement_config = config.config.get('replacement', {}).get('placement', {})
    project_name = config.get("project", "name")
    
    # Get paths from config or CLI
    if args.object_gaussians:
        object_path = args.object_gaussians
    else:
        object_path_config = placement_config.get('object_gaussians')
        if object_path_config:
            object_path = str(object_path_config).replace('${project.name}', project_name)
            console.print(f"[cyan]Using object_gaussians from config:[/cyan] {object_path}")
        else:
            console.print("[red]Error: No object_gaussians path found![/red]")
            console.print("Provide: --object_gaussians or set replacement.placement.object_gaussians in config")
            sys.exit(1)
    
    if args.roi:
        roi_path = args.roi
    else:
        roi_path_config = placement_config.get('roi_file')
        if roi_path_config:
            roi_path = str(roi_path_config).replace('${project.name}', project_name)
            console.print(f"[cyan]Using roi from config:[/cyan] {roi_path}")
        else:
            console.print("[red]Error: No ROI path found![/red]")
            console.print("Provide: --roi or set replacement.placement.roi_file in config")
            sys.exit(1)
    
    if args.scene_gaussians:
        scene_path = args.scene_gaussians
    else:
        scene_path_config = placement_config.get('scene_gaussians')
        if scene_path_config:
            scene_path = str(scene_path_config).replace('${project.name}', project_name)
            console.print(f"[cyan]Using scene_gaussians from config:[/cyan] {scene_path}")
        else:
            console.print("[red]Error: No scene_gaussians path found![/red]")
            console.print("Provide: --scene_gaussians or set replacement.placement.scene_gaussians in config")
            sys.exit(1)
    
    # Load checkpoints
    object_ckpt = load_checkpoint(object_path, "Object Gaussians")
    roi_mask = load_checkpoint(roi_path, "ROI")  # This is just a tensor from Module 04a
    scene_ckpt = load_checkpoint(scene_path, "Scene Gaussians")
    
    # Load original scene for ROI positions if provided
    if args.original_scene:
        original_ckpt = load_checkpoint(args.original_scene, "Original Scene (for ROI positions)")
        original_gaussians = original_ckpt.get("splats", original_ckpt.get("gaussians", original_ckpt))
    else:
        # Use config default
        original_path = config.get_path('initial_training') / 'ckpt_initial.pt'
        console.print(f"[cyan]Using original scene from config:[/cyan] {original_path}")
        original_ckpt = load_checkpoint(str(original_path), "Original Scene (for ROI positions)")
        original_gaussians = original_ckpt.get("splats", original_ckpt.get("gaussians", original_ckpt))
    
    # Extract Gaussians from checkpoints
    # Object can be PLY (dict from load_ply_as_gaussians) or checkpoint dict
    if object_path.endswith('.ply'):
        object_gaussians = object_ckpt  # Already in correct format from load_ply_as_gaussians
    else:
        object_gaussians = object_ckpt.get("gaussians", object_ckpt)
    
    # Scene checkpoint can have "splats" or "gaussians" key
    scene_gaussians = scene_ckpt.get("splats", scene_ckpt.get("gaussians", scene_ckpt))
    
    # roi_mask might be dict with 'roi_binary' or just a tensor
    if isinstance(roi_mask, dict):
        roi_mask = roi_mask.get("roi_binary", roi_mask.get("roi_mask", roi_mask))
    
    # Transform object to ROI (using original scene positions for ROI bounds)
    transformed_object = transform_object_to_roi(object_gaussians, roi_mask, original_gaussians["means"], args)
    
    # Merge with scene
    merged_gaussians = merge_gaussians(scene_gaussians, transformed_object)
    
    # Determine output path using unified structure
    if args.output:
        output_path = Path(args.output)
    else:
        # Use config value
        output_config = placement_config.get('output_merged')
        if output_config:
            output_path = Path(str(output_config).replace('${project.name}', project_name))
            console.print(f"[cyan]Saving to:[/cyan] {output_path}")
        else:
            output_path = config.get_path('scene_placement') / 'merged_gaussians.pt'
    
    # Prepare metadata
    metadata = {
        "module": "06_place_object_at_roi",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "object_gaussians": str(args.object_gaussians),
            "roi": str(args.roi),
            "scene_gaussians": str(args.scene_gaussians),
            "original_scene": str(args.original_scene) if args.original_scene else "inferred from config",
        },
        "parameters": {
            "scale_factor": args.scale_factor,
            "placement": args.placement,
            "z_offset": args.z_offset,
            "rotation_degrees": args.rotation_degrees,
            "no_scale": args.no_scale,
            "manual_scale": args.manual_scale,
        },
        "results": {
            "num_scene_gaussians": len(scene_gaussians["means"]),
            "num_object_gaussians": len(transformed_object["means"]),
            "num_total_gaussians": len(merged_gaussians["means"]),
        },
        "outputs": {
            "merged_checkpoint": str(output_path),
        },
    }
    
    # Save merged scene
    save_merged_scene(merged_gaussians, output_path, metadata)
    
    # Save manifest using unified system
    config = ProjectConfig(args.config)
    config.save_manifest("06_place_object_at_roi", metadata)
    
    console.print(f"\n[cyan]Next step:[/cyan] Run Module 07 to visualize the merged scene:")
    console.print(f"  python 07_final_visualization.py \\")
    console.print(f"    --merged_ckpt {output_path} \\")
    console.print(f"    --data_root datasets/360_v2/garden")


if __name__ == "__main__":
    main()
