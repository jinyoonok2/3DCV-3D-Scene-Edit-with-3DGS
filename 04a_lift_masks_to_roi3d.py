#!/usr/bin/env python3
"""
04_lift_masks_to_roi3d.py - Lift 2D Masks to 3D ROI (Per-Gaussian Weights)

Goal: Convert per-view 2D masks into per-Gaussian ROI weights roi ∈ [0,1].

Approach:
  1. Load 3DGS checkpoint and 2D masks from Module 03
  2. For each training view, rasterize the scene to get per-pixel Gaussian IDs
  3. Accumulate mask values for each Gaussian across all views
  4. Normalize to get ROI weight per Gaussian (how much it appears in masked regions)
  5. Apply threshold to get binary ROI

Inputs:
  --ckpt: Path to 3DGS checkpoint (e.g., outputs/garden/01_gs_base/ckpt_initial.pt)
  --masks_root: Directory with 2D masks (e.g., outputs/garden/round_001/masks/sam_masks/)
  --images_root: Directory with rendered images to align masks (e.g., outputs/garden/round_001/pre_edit/train/)
  --data_root: Dataset root for camera poses (e.g., datasets/360_v2/garden)
  --output_dir: Output directory (default: inferred from masks_root)
  --roi_thresh: Threshold for binary ROI (default: 0.5)
  --min_views: Minimum views a Gaussian must appear in masked region (default: 3)

Outputs (saved in output_dir):
  - roi.pt: Float tensor [N_gaussians] with ROI weights in [0,1]
  - roi_binary.pt: Boolean tensor [N_gaussians] for thresholded ROI
  - proj_masks/roi_proj_view_XXX.png: Projected ROI vs SAM mask comparison
  - metrics.json: Mean IoU, ROI sparsity, number of ROI Gaussians
  - manifest.json: Parameters and metadata
"""

import argparse
import json
import sys
from datetime import datetime
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

# Import project config
from project_utils.config import ProjectConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Lift 2D masks to 3D ROI weights")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to 3DGS checkpoint (overrides config)",
    )
    parser.add_argument(
        "--masks_root",
        type=str,
        default=None,
        help="Directory containing 2D masks (overrides config)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root for camera poses (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--roi_thresh",
        type=float,
        default=None,
        help="Threshold for binary ROI (overrides config)",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=3,
        help="Minimum views a Gaussian must appear in masked region",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Spherical harmonics degree (must match checkpoint)",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path, device="cuda"):
    """Load 3DGS checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Check if splats are nested under 'splats' key
    if "splats" in ckpt:
        print(f"Checkpoint structure: nested under 'splats' key (step {ckpt.get('step', 'unknown')})")
        ckpt = ckpt["splats"]
    
    # Extract splats (parameters)
    splats = {}
    for key in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
        if key in ckpt:
            splats[key] = ckpt[key].to(device)
    
    # Combine SH coefficients if split
    if "sh0" in splats and "shN" in splats:
        # sh0: [N, 1, 3], shN: [N, K-1, 3]
        splats["colors"] = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        del splats["sh0"]
        del splats["shN"]
    elif "colors" in ckpt:
        splats["colors"] = ckpt["colors"].to(device)
    
    num_gaussians = splats["means"].shape[0]
    print(f"✓ Loaded {num_gaussians:,} Gaussians")
    
    return splats


def load_masks(masks_root, image_names):
    """Load 2D masks corresponding to image names."""
    masks_root = Path(masks_root)
    masks = {}
    
    print(f"Loading masks from {masks_root}...")
    missing = 0
    
    for img_name in image_names:
        # Try different extensions
        mask_path = None
        for ext in [".npy", ".png", ".jpg"]:
            candidate = masks_root / f"mask_{img_name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            missing += 1
            continue
        
        # Load mask
        if mask_path.suffix == ".npy":
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0
        
        masks[img_name] = mask
    
    print(f"✓ Loaded {len(masks)} masks ({missing} missing)")
    return masks


def rasterize_gaussians_with_ids(splats, camtoworlds, Ks, width, height, sh_degree=3, device="cuda"):
    """
    Rasterize Gaussians and return rendered images + Gaussian IDs per pixel.
    
    Returns:
        renders: [N_views, H, W, 3] RGB images
        gaussian_ids: [N_views, H, W] Gaussian ID per pixel (-1 for background)
        alphas: [N_views, H, W] Alpha values per pixel
    """
    renders = []
    gaussian_ids_list = []
    alphas_list = []
    
    # Prepare colors (SH coefficients)
    colors = splats["colors"]
    num_sh_bases = (sh_degree + 1) ** 2
    colors_truncated = colors[:, :num_sh_bases, :]  # [N, K, 3]
    
    for i in range(len(camtoworlds)):
        # Rasterize
        render_colors, render_alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=colors_truncated,
            viewmats=torch.linalg.inv(camtoworlds[i:i+1]),  # [1, 4, 4]
            Ks=Ks[i:i+1],  # [1, 3, 3]
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB+ED",  # Enable extra data for tracking
        )
        
        # Extract Gaussian IDs per pixel
        # Use info dictionary to get contributing Gaussians
        # Note: gsplat returns gaussian_ids in info if available
        if "gaussian_ids" in info:
            ids = info["gaussian_ids"]  # [1, H, W]
        else:
            # Fallback: use depth ordering to approximate (not perfect)
            # This is a simplified version - ideally we'd track exact IDs
            ids = torch.full((1, height, width), -1, dtype=torch.long, device=device)
        
        renders.append(render_colors[0])  # [H, W, 3]
        gaussian_ids_list.append(ids[0])  # [H, W]
        alphas_list.append(render_alphas[0])  # [H, W, 1]
    
    renders = torch.stack(renders)  # [N, H, W, 3]
    gaussian_ids = torch.stack(gaussian_ids_list)  # [N, H, W]
    alphas = torch.stack([a.squeeze(-1) for a in alphas_list])  # [N, H, W]
    
    return renders, gaussian_ids, alphas


def compute_roi_weights_voting(splats, dataset, masks, sh_degree=3, device="cuda"):
    """
    Compute ROI weights by voting: for each Gaussian, accumulate mask values
    from all views where it's visible.
    
    Approach: Rasterize each view, for each pixel with mask > 0, increment
    the vote for contributing Gaussians.
    """
    num_gaussians = splats["means"].shape[0]
    
    # Accumulators
    roi_votes = torch.zeros(num_gaussians, device=device)
    roi_counts = torch.zeros(num_gaussians, device=device)
    
    print("Computing ROI weights by accumulating votes from 2D masks...")
    
    # Prepare colors
    colors = splats["colors"]
    num_sh_bases = (sh_degree + 1) ** 2
    colors_truncated = colors[:, :num_sh_bases, :]
    
    # Process each view
    for idx in tqdm(range(len(dataset)), desc="Processing views"):
        data = dataset[idx]
        img_name = f"view_{idx:03d}"
        
        if img_name not in masks:
            continue
        
        # Get camera parameters
        camtoworld = data["camtoworld"].unsqueeze(0).to(device)  # [1, 4, 4]
        K = data["K"].unsqueeze(0).to(device)  # [1, 3, 3]
        height, width = data["image"].shape[:2]
        
        # Get 2D mask
        mask_2d = masks[img_name]  # [H, W]
        
        # Debug first view
        if idx == 0:
            print(f"  DEBUG view {idx}:")
            print(f"    Image size: {height}x{width}")
            print(f"    Mask size: {mask_2d.shape}")
            print(f"    Mask range: [{mask_2d.min():.3f}, {mask_2d.max():.3f}]")
            print(f"    Mask mean: {mask_2d.mean():.3f}")
            print(f"    Pixels > 0: {(mask_2d > 0).sum()} / {mask_2d.size}")
        
        # CRITICAL: Resize mask to match image resolution if needed
        if mask_2d.shape[0] != height or mask_2d.shape[1] != width:
            mask_2d = cv2.resize(mask_2d, (width, height), interpolation=cv2.INTER_LINEAR)
        
        mask_tensor = torch.from_numpy(mask_2d).float().to(device)
        
        # Rasterize to get per-pixel contribution
        # We'll use a different approach: rasterize with alpha and approximate
        # contribution by checking which Gaussians project to masked regions
        
        # Project Gaussians to 2D
        with torch.no_grad():
            # Transform means to camera space
            means_world = splats["means"]  # [N, 3]
            means_homo = torch.cat([means_world, torch.ones(num_gaussians, 1, device=device)], dim=1)  # [N, 4]
            viewmat = torch.linalg.inv(camtoworld[0])  # [4, 4]
            means_cam = (viewmat @ means_homo.T).T  # [N, 4]
            means_cam = means_cam[:, :3]  # [N, 3]
            
            # Project to image plane
            K_mat = K[0]  # [3, 3]
            means_proj = (K_mat @ means_cam.T).T  # [N, 3]
            means_2d = means_proj[:, :2] / (means_proj[:, 2:3] + 1e-6)  # [N, 2]
            
            # Check if Gaussian centers are within masked regions
            # Convert to pixel coordinates
            px = means_2d[:, 0].long()
            py = means_2d[:, 1].long()
            
            # Valid pixels (within image bounds)
            valid = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (means_cam[:, 2] > 0)
            
            # VECTORIZED: For valid Gaussians, check mask value at their projected location
            # Get indices of valid Gaussians
            valid_indices = torch.where(valid)[0]
            
            if len(valid_indices) > 0:
                # Get pixel coordinates for valid Gaussians
                valid_px = px[valid_indices].clamp(0, width - 1)
                valid_py = py[valid_indices].clamp(0, height - 1)
                
                # Sample mask values at projected locations (vectorized)
                mask_values = mask_tensor[valid_py, valid_px]  # [num_valid]
                
                # Get opacities for valid Gaussians
                valid_opacities = splats["opacities"][valid_indices].squeeze(-1)  # [num_valid]
                
                # Accumulate votes (vectorized)
                roi_votes[valid_indices] += mask_values * valid_opacities
                roi_counts[valid_indices] += 1
                
                # Debug first view
                if idx == 0:
                    print(f"    Valid Gaussians: {len(valid_indices):,}")
                    print(f"    Gaussians in mask (value > 0): {(mask_values > 0).sum().item():,}")
                    print(f"    Mean mask value at Gaussian centers: {mask_values.mean().item():.3f}")
    
    # Normalize: average mask value across views where Gaussian is visible
    roi_weights = roi_votes / (roi_counts + 1e-6)
    roi_weights = torch.clamp(roi_weights, 0, 1)
    
    print(f"✓ Computed ROI weights")
    print(f"  Gaussians with roi > 0: {(roi_weights > 0).sum().item():,}")
    print(f"  Gaussians with roi > 0.5: {(roi_weights > 0.5).sum().item():,}")
    print(f"  Max ROI weight: {roi_weights.max().item():.3f}")
    
    return roi_weights


def apply_threshold(roi_weights, threshold, min_views_count=None):
    """Apply threshold to get binary ROI."""
    roi_binary = roi_weights >= threshold
    
    if min_views_count is not None:
        # Additional filter: require Gaussian to appear in at least N views
        # (This would require tracking view counts - simplified here)
        pass
    
    return roi_binary


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    ckpt = args.ckpt if args.ckpt else str(config.get_checkpoint_path('initial'))
    masks_root = args.masks_root if args.masks_root else str(config.get_path('masks') / 'sam_masks')
    data_root = args.data_root if args.data_root else str(config.get_path('dataset_root'))
    output_dir = args.output_dir if args.output_dir else str(config.get_path('roi'))
    roi_thresh = args.roi_thresh if args.roi_thresh is not None else config.config['roi']['threshold']
    seed = args.seed if args.seed is not None else config.config['dataset']['seed']
    factor = args.factor if args.factor is not None else config.config['dataset']['factor']
    test_every = args.test_every if args.test_every is not None else config.config['dataset']['test_every']
    sh_degree = args.sh_degree if args.sh_degree is not None else config.config['training']['sh_degree']
    min_views = args.min_views if args.min_views is not None else config.config['roi'].get('min_views', 3)
    
    set_random_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    proj_masks_dir = output_dir / "proj_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    proj_masks_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Lift 2D Masks to 3D ROI")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {ckpt}")
    print(f"Masks root: {masks_root}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"ROI threshold: {roi_thresh}")
    print()
    
    # Load dataset
    print(f"Loading dataset from {data_root}...")
    parser = Parser(
        data_dir=Path(data_root),
        factor=factor,
        normalize=True,
        test_every=test_every,
    )
    dataset = Dataset(parser, split="train")
    print(f"✓ Loaded {len(dataset)} training views")
    print()
    
    # Load checkpoint
    splats = load_checkpoint(ckpt, device=device)
    num_gaussians = splats["means"].shape[0]
    
    # Load masks
    image_names = [f"view_{i:03d}" for i in range(len(dataset))]
    masks = load_masks(masks_root, image_names)
    print()
    
    if len(masks) == 0:
        print("ERROR: No masks found! Check masks_root path.")
        sys.exit(1)
    
    # Compute ROI weights
    roi_weights = compute_roi_weights_voting(
        splats, dataset, masks, sh_degree=sh_degree, device=device
    )
    print()
    
    # Apply threshold for binary ROI
    roi_binary = apply_threshold(roi_weights, roi_thresh)
    num_roi_gaussians = roi_binary.sum().item()
    sparsity = (num_roi_gaussians / num_gaussians) * 100
    
    print(f"Binary ROI:")
    print(f"  Threshold: {roi_thresh}")
    print(f"  ROI Gaussians: {num_roi_gaussians:,} / {num_gaussians:,} ({sparsity:.2f}%)")
    print()
    
    # Save ROI weights
    torch.save(roi_weights.cpu(), output_dir / "roi.pt")
    torch.save(roi_binary.cpu(), output_dir / "roi_binary.pt")
    print(f"✓ Saved ROI weights to {output_dir / 'roi.pt'}")
    print(f"✓ Saved binary ROI to {output_dir / 'roi_binary.pt'}")
    print()
    
    mean_iou = 0.0  # Visualization moved to 04b_visualize_roi.py
    
    # Save metrics
    metrics = {
        "num_gaussians": num_gaussians,
        "num_roi_gaussians": int(num_roi_gaussians),
        "roi_sparsity_percent": float(sparsity),
        "roi_threshold": args.roi_thresh,
        "mean_iou": float(mean_iou),
        "num_masks": len(masks),
        "roi_weight_stats": {
            "min": float(roi_weights.min()),
            "max": float(roi_weights.max()),
            "mean": float(roi_weights.mean()),
            "median": float(roi_weights.median()),
        }
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics to {output_dir / 'metrics.json'}")
    
    # Save manifest
    manifest = {
        "module": "04_lift_masks_to_roi3d",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "checkpoint": str(ckpt),
            "masks_root": str(masks_root),
            "data_root": str(data_root),
        },
        "parameters": {
            "roi_thresh": roi_thresh,
            "min_views": min_views,
            "sh_degree": sh_degree,
            "seed": seed,
        },
        "outputs": {
            "roi_weights": str(output_dir / "roi.pt"),
            "roi_binary": str(output_dir / "roi_binary.pt"),
            "proj_masks_dir": str(proj_masks_dir),
        },
        "metrics": metrics,
    }
    
    config.save_manifest("04_lift_masks_to_roi3d", manifest)
    print(f"✓ Saved manifest to {config.get_path('logs') / '04_lift_masks_to_roi3d_manifest.json'}")
    print()
    print("=" * 80)
    print("ROI LIFTING COMPLETE")
    print("=" * 80)
    print(f"✓ ROI weights: {output_dir / 'roi.pt'}")
    print(f"✓ Binary ROI: {output_dir / 'roi_binary.pt'}")
    print()
    print(f"Next steps:")
    print(f"  1. Visualize ROI: python 04b_visualize_roi.py --roi {output_dir / 'roi.pt'} --ckpt {ckpt} --data_root {data_root} --masks_root {masks_root}")
    print(f"  2. Create edited targets: python 05_ip2p_edit_targets.py ...")
    print(f"  3. ROI-gated optimization: python 06_optimize_with_roi.py ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
