#!/usr/bin/env python3
"""
04b_visualize_roi.py - Visualize ROI by Rendering Full Scene with Highlighted ROI

Goal: Render full 3D scene with ROI Gaussians highlighted for validation.

Approach:
  1. Load 3DGS checkpoint and ROI weights from Module 04
  2. For each training view, render FULL scene (all Gaussians)
  3. Use ROI weights as colors to highlight ROI regions
  4. Compare with original SAM masks and compute IoU

This is separate from 04_lift_masks_to_roi3d.py because:
  - ROI computation is fast (5 seconds for 6.6M Gaussians)
  - Full-scene rendering is memory-intensive and may fail on large scenes
  - Visualization is optional and doesn't block the pipeline

Inputs:
  --roi: Path to ROI weights (e.g., outputs/garden/round_001/roi.pt)
  --ckpt: Path to 3DGS checkpoint (e.g., outputs/garden/01_gs_base/ckpt_29999.pt)
  --masks_root: Directory with 2D masks for IoU comparison
  --data_root: Dataset root for camera poses
  --output_dir: Output directory (default: same as roi.pt directory)
  --num_views: Number of views to visualize (default: 10)

Outputs (saved in output_dir/proj_masks/):
  - roi_proj_view_XXX.png: Projected ROI vs SAM mask comparison
  - visualization_metrics.json: Mean IoU between projected ROI and SAM masks
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add gsplat examples to path
sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))

try:
    from datasets.colmap import Dataset, Parser
    from utils import set_random_seed
except ImportError:
    print("ERROR: Could not import from gsplat examples.")
    print("Make sure gsplat-src/examples is available.")
    sys.exit(1)

try:
    from gsplat import rasterization
except ImportError:
    print("ERROR: gsplat not installed. Run: pip install gsplat")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ROI by rendering full scene")
    parser.add_argument(
        "--roi",
        type=str,
        required=True,
        help="Path to ROI weights tensor (roi.pt)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to 3DGS checkpoint",
    )
    parser.add_argument(
        "--masks_root",
        type=str,
        required=True,
        help="Directory containing 2D masks for IoU comparison",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root for camera poses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as roi.pt directory)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=10,
        help="Number of views to visualize",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Dataset downsampling factor",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Test split frequency",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Spherical harmonics degree",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path, device="cuda"):
    """Load 3DGS checkpoint and apply transformations (same as step 02)."""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if "splats" in ckpt:
        ckpt = ckpt["splats"]
    
    splats = {}
    for key in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
        if key in ckpt:
            splats[key] = ckpt[key].to(device)
    
    # Apply same transformations as step 02
    splats["scales"] = torch.exp(splats["scales"])  # Exp transform
    splats["opacities"] = torch.sigmoid(splats["opacities"])  # Sigmoid transform
    
    # Combine SH0 and SHN into colors
    if "sh0" in splats and "shN" in splats:
        splats["colors"] = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    
    num_gaussians = splats["means"].shape[0]
    print(f"✓ Loaded {num_gaussians:,} Gaussians (with exp/sigmoid transforms)")
    return splats


def load_masks(masks_root, image_names):
    """Load 2D masks from disk."""
    masks_path = Path(masks_root)
    masks = {}
    
    print(f"Loading masks from {masks_path}...")
    
    for img_name in image_names:
        # Try different filename patterns and extensions
        for prefix in ["mask_", ""]:
            for ext in [".npy", ".png"]:
                mask_file = masks_path / f"{prefix}{img_name}{ext}"
                if mask_file.exists():
                    if ext == ".npy":
                        mask = np.load(mask_file)
                    else:
                        mask = np.array(Image.open(mask_file))
                        if mask.ndim == 3:
                            mask = mask[:, :, 0]  # Grayscale
                        mask = mask.astype(np.float32) / 255.0
                    
                    masks[img_name] = mask
                    break
            if img_name in masks:
                break
    
    print(f"✓ Loaded {len(masks)} masks")
    return masks


def project_roi_to_masks(splats, roi_weights, dataset, sh_degree=3, device="cuda"):
    """
    Project ROI weights back to 2D by rendering FULL scene.
    ROI Gaussians are highlighted by using roi_weights as colors.
    """
    projected_masks = {}
    
    torch.cuda.empty_cache()
    
    num_gaussians = splats["means"].shape[0]
    num_roi = (roi_weights > 0.01).sum().item()
    
    print(f"Projecting ROI to 2D views...")
    print(f"  Rendering FULL scene ({num_gaussians:,} Gaussians)")
    print(f"  ROI Gaussians: {num_roi:,} ({100*num_roi/num_gaussians:.2f}%)")
    
    for idx in tqdm(range(len(dataset)), desc="Rendering views"):
        data = dataset[idx]
        img_name = f"view_{idx:03d}"
        
        camtoworld = data["camtoworld"].unsqueeze(0).to(device)
        K = data["K"].unsqueeze(0).to(device)
        height, width = data["image"].shape[:2]
        
        try:
            with torch.no_grad():
                # Use ROI weights as grayscale color for ALL Gaussians
                # Non-ROI Gaussians will render as black (roi_weight=0)
                roi_colors = roi_weights.unsqueeze(-1).unsqueeze(-1).expand(num_gaussians, 1, 3)
                
                # Render FULL scene (like step 02)
                render_roi, _, _ = rasterization(
                    means=splats["means"],
                    quats=splats["quats"],
                    scales=splats["scales"],
                    opacities=splats["opacities"],
                    colors=roi_colors,
                    viewmats=torch.linalg.inv(camtoworld),
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=0,  # DC only (constant colors)
                    render_mode="RGB",
                )
                
                # Extract projected ROI mask
                proj_mask = render_roi[0, :, :, 0].cpu().numpy()
                projected_masks[img_name] = proj_mask
                
                del render_roi
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  GPU OOM at view {idx}.")
                print(f"   Scene too large ({num_gaussians:,} Gaussians).")
                print(f"   Processed {len(projected_masks)}/{len(dataset)} views.")
                break
            else:
                raise
    
    return projected_masks


def visualize_roi_projection(proj_mask, sam_mask, output_path):
    """Create side-by-side comparison of projected ROI vs SAM mask."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Projected ROI
    axes[0].imshow(proj_mask, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title("Projected ROI (from 3D)")
    axes[0].axis('off')
    
    # SAM mask
    axes[1].imshow(sam_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("SAM Mask (2D)")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(sam_mask, cmap='gray', vmin=0, vmax=1, alpha=0.5)
    axes[2].imshow(proj_mask, cmap='jet', vmin=0, vmax=1, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("ROI VISUALIZATION (Full Scene Rendering)")
    print("=" * 80)
    print()
    
    # Setup output directory
    roi_path = Path(args.roi)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = roi_path.parent
    
    proj_masks_dir = output_dir / "proj_masks"
    proj_masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Projection masks: {proj_masks_dir}")
    print()
    
    # Load ROI weights
    print(f"Loading ROI weights from {roi_path}...")
    roi_weights = torch.load(roi_path, map_location=device)
    num_gaussians = roi_weights.shape[0]
    num_roi = (roi_weights > 0.01).sum().item()
    print(f"✓ Loaded ROI weights for {num_gaussians:,} Gaussians")
    print(f"  ROI Gaussians (>0.01): {num_roi:,} ({100*num_roi/num_gaussians:.2f}%)")
    print()
    
    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    parser = Parser(
        data_dir=Path(args.data_root),
        factor=args.factor,
        normalize=True,
        test_every=args.test_every,
    )
    dataset = Dataset(parser, split="train")
    print(f"✓ Loaded {len(dataset)} training views")
    print()
    
    # Load checkpoint
    splats = load_checkpoint(args.ckpt, device=device)
    print()
    
    # Load masks
    image_names = [f"view_{i:03d}" for i in range(len(dataset))]
    masks = load_masks(args.masks_root, image_names)
    print()
    
    # Limit number of views
    num_views_to_render = min(args.num_views, len(dataset))
    dataset_subset = torch.utils.data.Subset(dataset, range(num_views_to_render))
    
    print(f"Rendering {num_views_to_render} views...")
    print()
    
    # Project ROI to 2D
    projected_masks = project_roi_to_masks(
        splats, roi_weights, dataset_subset, args.sh_degree, device
    )
    print()
    
    # Compute IoU with SAM masks
    ious = []
    for idx in range(num_views_to_render):
        img_name = f"view_{idx:03d}"
        
        if img_name not in masks or img_name not in projected_masks:
            continue
        
        sam_mask = masks[img_name]
        proj_mask = projected_masks[img_name]
        
        # Resize masks to match if needed
        if sam_mask.shape != proj_mask.shape:
            sam_mask = cv2.resize(sam_mask, (proj_mask.shape[1], proj_mask.shape[0]))
        
        # Threshold both for binary comparison
        sam_binary = sam_mask > 0.5
        proj_binary = proj_mask > 0.5
        
        iou = compute_iou(proj_binary, sam_binary)
        ious.append(iou)
        
        # Save visualization
        vis_path = proj_masks_dir / f"roi_proj_{img_name}.png"
        visualize_roi_projection(proj_mask, sam_mask, vis_path)
    
    mean_iou = np.mean(ious) if ious else 0.0
    print(f"Mean IoU (projected ROI vs SAM masks): {mean_iou:.3f}")
    print(f"✓ Saved {len(ious)} visualizations to {proj_masks_dir}")
    print()
    
    # Save metrics
    metrics = {
        "num_views_rendered": len(projected_masks),
        "num_views_with_iou": len(ious),
        "mean_iou": float(mean_iou),
        "iou_per_view": {f"view_{i:03d}": float(iou) for i, iou in enumerate(ious)},
        "roi_stats": {
            "num_gaussians": num_gaussians,
            "num_roi_gaussians": int(num_roi),
            "roi_percentage": float(100 * num_roi / num_gaussians),
        }
    }
    
    metrics_path = output_dir / "visualization_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics to {metrics_path}")
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"✓ Visualizations: {proj_masks_dir}")
    print(f"✓ Mean IoU: {mean_iou:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
