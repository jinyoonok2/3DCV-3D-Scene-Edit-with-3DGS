#!/usr/bin/env python3
"""
05a_remove_and_render_holes.py - Remove ROI Gaussians and Render Holed Scene

Goal: Delete Gaussians in ROI and render the scene with holes from all training views.

Approach:
  1. Load 3DGS checkpoint and ROI weights
  2. Delete Gaussians where ROI > threshold
  3. Render holed scene from all training views
  4. Save holed renders and hole masks for inpainting

Inputs:
  --ckpt: Path to initial 3DGS checkpoint
  --roi: Path to ROI weights (.pt file)
  --data_root: Dataset root for camera poses
  --output_dir: Output directory (default: inferred from ROI parent)
  --roi_thresh: ROI threshold for deletion (default: 0.7)
  --factor: Image downsample factor (default: 4)

Outputs (saved in output_dir/05a_holed/):
  - ckpt_holed.pt: Checkpoint with ROI Gaussians removed
  - renders/train/*.png: Rendered images with holes
  - masks/train/*.png: Binary masks (white = hole)
  - manifest.json: Metadata
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from rich.console import Console

from project_utils.config import ProjectConfig

console = Console()

# Add gsplat examples to path
sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))

try:
    from datasets.colmap import Dataset, Parser
    from utils import set_random_seed
except ImportError:
    console.print("[red]ERROR: Could not import from gsplat examples.[/red]")
    sys.exit(1)

try:
    from gsplat import rasterization
except ImportError:
    console.print("[red]ERROR: gsplat not installed.[/red]")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Remove ROI and render holes")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to 3DGS checkpoint (overrides config)")
    parser.add_argument("--roi", type=str, default=None, help="Path to ROI weights (overrides config)")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--roi_thresh", type=float, default=None, help="ROI threshold (overrides config)")
    parser.add_argument("--factor", type=int, default=None, choices=[1,2,4,8], help="Downsample factor (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def delete_roi_gaussians(params, roi_weights, roi_thresh):
    """Delete Gaussians in ROI and return new parameters"""
    mask_keep = roi_weights <= roi_thresh
    n_original = len(roi_weights)
    n_keep = mask_keep.sum().item()
    n_deleted = n_original - n_keep
    
    console.print(f"\n[yellow]Deleting Gaussians in ROI:[/yellow]")
    console.print(f"  Original: {n_original:,}")
    console.print(f"  Deleted:  {n_deleted:,} ({100*n_deleted/n_original:.1f}%)")
    console.print(f"  Keeping:  {n_keep:,} ({100*n_keep/n_original:.1f}%)")
    
    new_params = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor) and len(value) == n_original:
            new_params[key] = value[mask_keep]
        else:
            new_params[key] = value
    
    return new_params, mask_keep, n_deleted


def render_view(means, quats, scales, opacities, sh0, shN, viewmat, K, width, height, sh_degree=3):
    """Render a single view"""
    # Concatenate SH coefficients
    num_sh_bases = (sh_degree + 1) ** 2
    colors = torch.cat([sh0, shN], 1)[:, :num_sh_bases, :]  # [N, K, 3]
    
    # Apply activation functions
    scales_act = torch.exp(scales)
    opacities_act = torch.sigmoid(opacities)
    
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales_act,
        opacities=opacities_act,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
    )
    return renders[0], alphas[0], info


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    ckpt = args.ckpt if args.ckpt else str(config.get_checkpoint_path('initial'))
    roi_path = args.roi if args.roi else str(config.get_path('roi') / 'roi.pt')
    data_root = args.data_root if args.data_root else str(config.get_path('dataset_root'))
    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05a_holed')
    roi_thresh = args.roi_thresh if args.roi_thresh is not None else config.config['roi']['threshold']
    factor = args.factor if args.factor is not None else config.config['dataset']['factor']
    seed = args.seed if args.seed is not None else config.config['dataset']['seed']
    
    device = torch.device(args.device)
    set_random_seed(seed)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05a - Remove ROI and Render Holes[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    ckpt_path = Path(ckpt)
    roi_path = Path(roi_path)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    renders_dir = output_dir / "renders" / "train"
    masks_dir = output_dir / "masks" / "train"
    renders_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]ROI threshold:[/cyan] {roi_thresh}\n")
    
    # Load checkpoint
    console.print("[cyan]Loading 3DGS checkpoint...[/cyan]")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt["splats"]
    console.print(f"[green]✓ Loaded {len(params['means']):,} Gaussians[/green]")
    
    # Load ROI
    console.print("[cyan]Loading ROI weights...[/cyan]")
    roi_weights = torch.load(roi_path, map_location=device)
    console.print(f"[green]✓ Loaded ROI (mean: {roi_weights.mean():.3f})[/green]")
    
    # Delete ROI Gaussians
    params_holed, mask_keep, n_deleted = delete_roi_gaussians(params, roi_weights, roi_thresh)
    
    # Load dataset
    console.print("\n[cyan]Loading dataset...[/cyan]")
    parser = Parser(data_root, factor=factor, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train")
    console.print(f"[green]✓ Loaded {len(dataset)} training views[/green]")
    
    # Render holed scene
    console.print("\n[yellow]Rendering holed scene...[/yellow]")
    
    for idx in tqdm(range(len(dataset)), desc="Rendering"):
        data = dataset[idx]
        
        # Get dimensions from image tensor
        image_tensor = data["image"]  # [H, W, 3]
        height, width = image_tensor.shape[:2]
        
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        worldtoview = torch.inverse(camtoworld)
        
        # Render
        render, alpha, _ = render_view(
            means=params_holed["means"],
            quats=params_holed["quats"],
            scales=params_holed["scales"],
            opacities=params_holed["opacities"],
            sh0=params_holed["sh0"],
            shN=params_holed["shN"],
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
            sh_degree=3,
        )
        
        # Create hole mask (low alpha = hole)
        mask = (alpha < 0.5).float()
        
        # Save render
        img_np = (render.cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        cv2.imwrite(str(renders_dir / f"{idx:05d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_dir / f"{idx:05d}.png"), mask_np)
    
    console.print(f"[green]✓ Saved renders to {renders_dir}[/green]")
    console.print(f"[green]✓ Saved masks to {masks_dir}[/green]")
    
    # Save holed checkpoint
    holed_ckpt = {"splats": params_holed}
    torch.save(holed_ckpt, output_dir / "ckpt_holed.pt")
    console.print(f"[green]✓ Saved holed checkpoint[/green]")
    
    # Save manifest
    manifest = {
        "module": "05a_remove_and_render_holes",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "checkpoint": str(ckpt_path),
            "roi": str(roi_path),
            "data_root": str(data_root),
        },
        "parameters": {
            "roi_thresh": roi_thresh,
            "factor": factor,
            "seed": seed,
        },
        "results": {
            "n_views": len(dataset),
            "n_original_gaussians": len(roi_weights),
            "n_deleted_gaussians": n_deleted,
            "n_remaining_gaussians": len(params_holed["means"]),
        },
    }
    
    config.save_manifest("05a_remove_and_render_holes", manifest)
    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05a_remove_and_render_holes_manifest.json'}[/green]")
    
    console.print("\n[bold green]✓ Module 05a Complete![/bold green]")
    console.print(f"[bold green]Next: Run 05b to inpaint the holes[/bold green]\n")


if __name__ == "__main__":
    main()
