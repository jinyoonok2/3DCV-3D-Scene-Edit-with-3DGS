#!/usr/bin/env python3
"""
00_check_dataset.py - Validate Dataset Inputs

Goal: Confirm dataset paths, list images/poses, and preview a few frames to ensure
      the target object (e.g., plant pot) is visible.

Inputs:
  --data_root: Path to the dataset (e.g., datasets/360_v2/garden/)
  --output_dir: Directory to save validation outputs (default: outputs/<dataset_name>/00_dataset/)
  --n_thumbs: Number of thumbnail images to save (default: 6)
  --factor: Downsample factor for dataset loading (default: 4)

Outputs (saved in output_dir):
  - summary.txt: Dataset statistics (image counts, intrinsics, focal lengths, etc.)
  - thumbs/: Sampled thumbnail images
  - manifest.json: Provenance information (paths, timestamp, parameters)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

# Add gsplat examples to path
sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))
from datasets.colmap import Parser

# Import project config (no name conflict now)
from project_utils.config import ProjectConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset inputs")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the dataset directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for validation results (overrides config)",
    )
    parser.add_argument(
        "--n_thumbs",
        type=int,
        default=6,
        help="Number of thumbnail images to save",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="Downsample factor for loading dataset (overrides config)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize world space (same as training)",
    )
    return parser.parse_args()


def compute_dataset_stats(parser_obj: Parser):
    """Compute statistics about the dataset."""
    stats = {}
    
    # Basic counts
    stats["num_images"] = len(parser_obj.image_names)
    stats["num_points"] = len(parser_obj.points)
    stats["num_cameras"] = len(set(parser_obj.camera_ids))
    
    # Camera intrinsics statistics
    focal_lengths = []
    image_sizes = []
    for camera_id in parser_obj.Ks_dict.keys():
        K = parser_obj.Ks_dict[camera_id]
        fx, fy = K[0, 0], K[1, 1]
        focal_lengths.append([fx, fy])
        width, height = parser_obj.imsize_dict[camera_id]
        image_sizes.append([width, height])
    
    focal_lengths = np.array(focal_lengths)
    image_sizes = np.array(image_sizes)
    
    stats["focal_length_min"] = focal_lengths.min(axis=0).tolist()
    stats["focal_length_max"] = focal_lengths.max(axis=0).tolist()
    stats["focal_length_mean"] = focal_lengths.mean(axis=0).tolist()
    stats["image_sizes"] = image_sizes.tolist()
    
    # Camera pose statistics
    positions = parser_obj.camtoworlds[:, :3, 3]  # Extract positions
    stats["camera_positions_min"] = positions.min(axis=0).tolist()
    stats["camera_positions_max"] = positions.max(axis=0).tolist()
    stats["camera_positions_mean"] = positions.mean(axis=0).tolist()
    stats["camera_positions_std"] = positions.std(axis=0).tolist()
    
    # Scene bounds
    if hasattr(parser_obj, "bounds") and parser_obj.bounds is not None:
        stats["bounds"] = parser_obj.bounds.tolist()
    
    # Point cloud statistics
    stats["points_min"] = parser_obj.points.min(axis=0).tolist()
    stats["points_max"] = parser_obj.points.max(axis=0).tolist()
    stats["points_mean"] = parser_obj.points.mean(axis=0).tolist()
    stats["points_std"] = parser_obj.points.std(axis=0).tolist()
    
    if hasattr(parser_obj, "points_err"):
        stats["points_error_mean"] = float(parser_obj.points_err.mean())
        stats["points_error_std"] = float(parser_obj.points_err.std())
    
    return stats


def save_thumbnails(parser_obj: Parser, output_dir: Path, n_thumbs: int):
    """Save evenly-spaced thumbnail images from the dataset."""
    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = len(parser_obj.image_paths)
    # Select evenly-spaced indices
    indices = np.linspace(0, num_images - 1, n_thumbs, dtype=int)
    
    saved_paths = []
    for idx in indices:
        image_path = parser_obj.image_paths[idx]
        image_name = parser_obj.image_names[idx]
        
        # Load image
        image = imageio.imread(image_path)[..., :3]
        
        # Save thumbnail
        thumb_name = f"thumb_{idx:03d}_{Path(image_name).stem}.png"
        thumb_path = thumbs_dir / thumb_name
        imageio.imwrite(thumb_path, image)
        
        saved_paths.append(str(thumb_path))
        print(f"  Saved thumbnail {idx+1}/{n_thumbs}: {thumb_name}")
    
    return saved_paths


def write_summary(output_dir: Path, stats: dict, parser_obj: Parser):
    """Write human-readable summary file."""
    summary_path = output_dir / "summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASIC INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Number of images: {stats['num_images']}\n")
        f.write(f"  Number of cameras: {stats['num_cameras']}\n")
        f.write(f"  Number of 3D points: {stats['num_points']}\n")
        
        if "bounds" in stats:
            f.write(f"  Scene bounds (near, far): {stats['bounds']}\n")
        f.write("\n")
        
        f.write("IMAGE PROPERTIES\n")
        f.write("-" * 80 + "\n")
        unique_sizes = set(tuple(s) for s in stats['image_sizes'])
        for width, height in unique_sizes:
            count = sum(1 for s in stats['image_sizes'] if s[0] == width and s[1] == height)
            f.write(f"  Size {width}x{height}: {count} images\n")
        f.write("\n")
        
        f.write("CAMERA INTRINSICS\n")
        f.write("-" * 80 + "\n")
        fx_min, fy_min = stats['focal_length_min']
        fx_max, fy_max = stats['focal_length_max']
        fx_mean, fy_mean = stats['focal_length_mean']
        f.write(f"  Focal length (fx): min={fx_min:.2f}, max={fx_max:.2f}, mean={fx_mean:.2f}\n")
        f.write(f"  Focal length (fy): min={fy_min:.2f}, max={fy_max:.2f}, mean={fy_mean:.2f}\n")
        f.write("\n")
        
        f.write("CAMERA EXTRINSICS (Positions)\n")
        f.write("-" * 80 + "\n")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            f.write(f"  {axis}-axis: min={stats['camera_positions_min'][i]:.3f}, "
                   f"max={stats['camera_positions_max'][i]:.3f}, "
                   f"mean={stats['camera_positions_mean'][i]:.3f}, "
                   f"std={stats['camera_positions_std'][i]:.3f}\n")
        f.write("\n")
        
        f.write("POINT CLOUD STATISTICS\n")
        f.write("-" * 80 + "\n")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            f.write(f"  {axis}-axis: min={stats['points_min'][i]:.3f}, "
                   f"max={stats['points_max'][i]:.3f}, "
                   f"mean={stats['points_mean'][i]:.3f}, "
                   f"std={stats['points_std'][i]:.3f}\n")
        
        if 'points_error_mean' in stats:
            f.write(f"\n  Reprojection error: mean={stats['points_error_mean']:.4f}, "
                   f"std={stats['points_error_std']:.4f}\n")
        f.write("\n")
        
        f.write("IMAGE SAMPLES\n")
        f.write("-" * 80 + "\n")
        f.write(f"  First image: {parser_obj.image_names[0]}\n")
        f.write(f"  Last image:  {parser_obj.image_names[-1]}\n")
        f.write("\n")
        
        f.write("VERIFICATION CHECKS\n")
        f.write("-" * 80 + "\n")
        
        # Checks
        checks = []
        if stats['num_images'] > 50:
            checks.append("✓ Sufficient images for training (>50)")
        else:
            checks.append("✗ Warning: Low image count (<50)")
        
        if stats['num_points'] > 1000:
            checks.append("✓ Sufficient 3D points (>1000)")
        else:
            checks.append("✗ Warning: Low point count (<1000)")
        
        if stats['num_cameras'] > 0:
            checks.append(f"✓ Camera models loaded ({stats['num_cameras']} unique)")
        
        for check in checks:
            f.write(f"  {check}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSummary written to: {summary_path}")
    return summary_path


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    data_root = args.data_root if args.data_root else str(config.get_path('dataset_root'))
    output_dir = args.output_dir if args.output_dir else str(config.get_path('dataset_check'))
    factor = args.factor if args.factor is not None else config.config['dataset']['factor']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Dataset Validation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Downsample factor: {factor}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(data_root):
        print(f"ERROR: Data directory does not exist: {data_root}")
        sys.exit(1)
    
    # Load dataset using COLMAP parser
    print("Loading dataset...")
    try:
        parser_obj = Parser(
            data_dir=data_root,
            factor=factor,
            normalize=args.normalize,
            test_every=config.config['dataset'].get('test_every', 8),
        )
        print(f"✓ Successfully loaded dataset")
        print(f"  - {len(parser_obj.image_names)} images")
        print(f"  - {len(parser_obj.points)} 3D points")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute statistics
    print("Computing dataset statistics...")
    stats = compute_dataset_stats(parser_obj)
    print("✓ Statistics computed")
    print()
    
    # Save thumbnails
    print(f"Saving {args.n_thumbs} thumbnail images...")
    thumbnail_paths = save_thumbnails(parser_obj, output_dir, args.n_thumbs)
    print("✓ Thumbnails saved")
    print()
    
    # Write summary
    print("Writing summary...")
    summary_path = write_summary(output_dir, stats, parser_obj)
    print("✓ Summary written")
    print()
    
    # Create manifest
    manifest = {
        "module": "00_check_dataset",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "data_root": data_root,
        "output_dir": str(output_dir),
        "parameters": {
            "factor": factor,
            "normalize": args.normalize,
            "n_thumbs": args.n_thumbs,
        },
        "statistics": stats,
        "outputs": {
            "summary": str(summary_path),
            "thumbnails": thumbnail_paths,
        },
    }
    
    config.save_manifest("00_check_dataset", manifest)
    
    print(f"Manifest saved to: {manifest_path}")
    print()
    
    # Final verification message
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"✓ Dataset loaded successfully: {stats['num_images']} images")
    print(f"✓ Thumbnails saved: {len(thumbnail_paths)} images")
    print(f"✓ Summary available at: {summary_path}")
    print()
    print("Next step: Review the thumbnails to verify that the target object")
    print("           (e.g., plant pot) is visible in the sampled views.")
    print("=" * 80)


if __name__ == "__main__":
    main()
