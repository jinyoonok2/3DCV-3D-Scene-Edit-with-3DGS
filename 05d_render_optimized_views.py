#!/usr/bin/env python3
"""
05d_render_optimized_views.py - Render Optimized 3DGS Checkpoint

Goal: Render all training views from the optimized checkpoint (after inpainting optimization)
      to visualize the final result with brown plant removed.

Inputs:
  --ckpt: Path to optimized checkpoint (e.g., outputs/garden/round_001/05c_optimized/ckpt_patched.pt)
  --data_root: Path to the dataset directory
  --output_dir: Output directory (default: sibling of ckpt)
  --views: Which views to render ("all", "train", "val")
  --factor: Downsample factor (default: 4)

Outputs (saved in output_dir/renders/):
  - view_000.png, view_001.png, ...
  - manifest.json: Metadata about rendering
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm
from rich.console import Console

from project_utils.config import ProjectConfig

console = Console()

sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))
from datasets.colmap import Dataset, Parser
from utils import set_random_seed
from gsplat.rendering import rasterization


def parse_args():
    parser = argparse.ArgumentParser(description="Render optimized checkpoint views")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to optimized checkpoint (ckpt_patched.pt)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same dir as ckpt)",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="train",
        help="Which views: 'all', 'train', 'val'",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="Downsample factor (overrides config)",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=None,
        help="Test split interval (overrides config)",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Spherical harmonics degree",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path, device="cuda"):
    """Load optimized checkpoint."""
    console.print(f"[cyan]Loading checkpoint:[/cyan] {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    splats = checkpoint["splats"]
    splats_dict = {}
    for key, value in splats.items():
        if isinstance(value, torch.Tensor):
            splats_dict[key] = value.to(device)
        else:
            splats_dict[key] = torch.tensor(value).to(device)
    
    n_gaussians = len(splats_dict["means"])
    console.print(f"[green]✓[/green] Loaded {n_gaussians:,} Gaussians")
    return splats_dict


def render_view(splats, camtoworld, K, width, height, sh_degree=3, device="cuda"):
    """Render a single view from the 3DGS model."""
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3] - apply activation
    opacities = torch.sigmoid(splats["opacities"])  # [N,] - apply activation
    
    # Concatenate SH coefficients
    num_sh_bases = (sh_degree + 1) ** 2
    sh0 = splats["sh0"]  # [N, 1, 3]
    shN = splats["shN"]  # [N, K-1, 3]
    colors = torch.cat([sh0, shN], dim=1)[:, :num_sh_bases, :]  # [N, K, 3]
    
    # Prepare camera parameters
    camtoworld = camtoworld.to(device)
    K = K.to(device)
    
    # Rasterize
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld)[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        sh_degree=sh_degree,
    )
    
    return renders[0]  # [H, W, 3]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    cfg = ProjectConfig(args.config)
    data_root = args.data_root or cfg.config['paths']['dataset_root']
    factor = args.factor if args.factor is not None else cfg.config['dataset']['factor']
    test_every = args.test_every if args.test_every is not None else cfg.config['dataset']['test_every']
    
    set_random_seed(args.seed)
    
    # Setup output directory
    ckpt_path = Path(args.ckpt)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / "renders_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    
    # Load checkpoint
    splats = load_checkpoint(args.ckpt, device)
    
    # Load dataset for camera poses
    console.print(f"[cyan]Loading dataset:[/cyan] {data_root}")
    parser = Parser(
        data_dir=data_root,
        factor=factor,
        normalize=True,
        test_every=test_every,
    )
    dataset = Dataset(parser, split="train")
    
    console.print(f"[green]✓[/green] Dataset loaded: {len(dataset)} training views")
    
    # Determine which views to render
    if args.views == "all":
        indices = list(range(len(dataset)))
    elif args.views == "train":
        indices = list(range(len(dataset)))
    elif args.views == "val":
        val_dataset = Dataset(parser, split="val")
        dataset = val_dataset
        indices = list(range(len(val_dataset)))
    else:
        indices = [int(x) for x in args.views.split(",")]
    
    console.print(f"[cyan]Rendering {len(indices)} views...[/cyan]")
    
    # Render views
    metadata = {
        "checkpoint": str(ckpt_path),
        "data_root": str(data_root),
        "factor": factor,
        "sh_degree": args.sh_degree,
        "n_gaussians": len(splats["means"]),
        "n_views": len(indices),
        "timestamp": datetime.now().isoformat(),
    }
    
    for idx in tqdm(indices, desc="Rendering"):
        data = dataset[idx]
        camtoworld = data["camtoworld"]
        K = data["K"]
        
        # Get image dimensions from the image tensor
        image_tensor = data["image"]  # [H, W, 3]
        height, width = image_tensor.shape[:2]
        
        # Render
        render = render_view(
            splats, camtoworld, K, width, height,
            sh_degree=args.sh_degree, device=device
        )
        
        # Convert to uint8
        render_np = (render.cpu().numpy() * 255).astype(np.uint8)
        
        # Save
        output_path = output_dir / f"view_{idx:03d}.png"
        imageio.imwrite(output_path, render_np)
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"[green]✓[/green] Rendered {len(indices)} views to: {output_dir}")
    console.print(f"[green]✓[/green] Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
