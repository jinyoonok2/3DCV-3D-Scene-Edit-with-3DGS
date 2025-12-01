#!/usr/bin/env python3
"""
09_final_optimization.py - Final Scene Optimization with New Object

Goal: Optimize the merged scene (from Module 08) to blend the new object naturally.

This module refines the scene with the newly placed object by:
1. Loading the merged Gaussians (scene + object from Module 08)
2. Optimizing all Gaussian parameters using training views
3. Matching lighting, shadows, and appearance
4. Fixing any blending artifacts

The optimization process is similar to Module 05c but operates on the merged scene.

Inputs:
  --init_ckpt: Merged Gaussians checkpoint from Module 08
  --data_root: Path to dataset (for training views)
  --output_dir: Directory to save optimized results

Outputs:
  - ckpt_final.pt: Final optimized checkpoint
  - Rendered validation views

Dependencies:
  - gsplat: 3D Gaussian Splatting optimization
  - torch: Training loop
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rich.console import Console

# Import project utilities
from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final optimization of merged scene with new object"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        required=True,
        help="Initial checkpoint (merged scene from Module 08)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Number of optimization iterations (default: 3000)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N iterations (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    
    return parser.parse_args()


def load_merged_checkpoint(ckpt_path):
    """Load merged Gaussians checkpoint."""
    console.print(f"Loading merged checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Extract Gaussians
    if "gaussians" in ckpt:
        gaussians = ckpt["gaussians"]
    else:
        gaussians = ckpt
    
    console.print(f"✓ Loaded {len(gaussians['means'])} Gaussians")
    return gaussians


def rgb_ssim(
    img0: Tensor,
    img1: Tensor,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Compute SSIM loss (simplified version)."""
    assert img0.shape == img1.shape
    
    # Simple MSE-based approximation for now
    # Full SSIM implementation would be more complex
    mse = F.mse_loss(img0, img1)
    return 1.0 - torch.exp(-mse)


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 09: Final Scene Optimization")
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.get_path("module_09", "final_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    
    # Load merged Gaussians
    gaussians = load_merged_checkpoint(args.init_ckpt)
    
    # Move to device and setup optimization
    means = gaussians["means"].to(device).requires_grad_(True)
    quats = gaussians["quats"].to(device).requires_grad_(True)
    scales = gaussians["scales"].to(device).requires_grad_(True)
    sh0 = gaussians["sh0"].to(device).requires_grad_(True)
    shN = gaussians.get("shN", torch.zeros_like(sh0)).to(device).requires_grad_(True)
    opacities = gaussians["opacities"].to(device).requires_grad_(True)
    
    console.print(f"\nOptimization parameters:")
    console.print(f"  Means: {means.shape}")
    console.print(f"  Quats: {quats.shape}")
    console.print(f"  Scales: {scales.shape}")
    console.print(f"  SH0: {sh0.shape}")
    console.print(f"  ShN: {shN.shape}")
    console.print(f"  Opacities: {opacities.shape}")
    
    # Setup optimizer
    params = [
        {"params": [means], "lr": args.lr * 10, "name": "means"},
        {"params": [quats], "lr": args.lr, "name": "quats"},
        {"params": [scales], "lr": args.lr, "name": "scales"},
        {"params": [sh0], "lr": args.lr, "name": "sh0"},
        {"params": [shN], "lr": args.lr / 20, "name": "shN"},
        {"params": [opacities], "lr": args.lr, "name": "opacities"},
    ]
    optimizer = torch.optim.Adam(params)
    
    console.print(f"\nOptimizer: Adam")
    console.print(f"  Learning rate: {args.lr}")
    console.print(f"  Iterations: {args.iterations}")
    
    # Load dataset (simplified - would need actual dataset loader)
    # For now, just show the optimization structure
    console.print(f"\nDataset: {args.data_root}")
    console.print("[yellow]Note: Full dataset loader implementation needed[/yellow]")
    
    # Optimization loop
    console.print(f"\nStarting optimization...\n")
    
    pbar = tqdm(range(args.iterations), desc="Optimizing")
    for iteration in pbar:
        # TODO: Sample training view from dataset
        # For now, this is a placeholder structure
        
        # Placeholder: Would render and compute loss here
        # loss = compute_loss(rendered, target)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # Update progress
        if iteration % 100 == 0:
            pbar.set_postfix({"iter": iteration})
        
        # Save checkpoint
        if (iteration + 1) % args.save_every == 0 or iteration == args.iterations - 1:
            ckpt_path = output_dir / f"ckpt_{iteration+1:06d}.pt"
            checkpoint = {
                "iteration": iteration + 1,
                "gaussians": {
                    "means": means.detach().cpu(),
                    "quats": quats.detach().cpu(),
                    "scales": scales.detach().cpu(),
                    "sh0": sh0.detach().cpu(),
                    "shN": shN.detach().cpu(),
                    "opacities": opacities.detach().cpu(),
                },
            }
            torch.save(checkpoint, ckpt_path)
            console.print(f"\n✓ Checkpoint saved: {ckpt_path}")
    
    # Save final checkpoint
    final_ckpt_path = output_dir / "ckpt_final.pt"
    final_checkpoint = {
        "iteration": args.iterations,
        "gaussians": {
            "means": means.detach().cpu(),
            "quats": quats.detach().cpu(),
            "scales": scales.detach().cpu(),
            "sh0": sh0.detach().cpu(),
            "shN": shN.detach().cpu(),
            "opacities": opacities.detach().cpu(),
        },
        "metadata": {
            "module": "09_final_optimization",
            "timestamp": datetime.now().isoformat(),
            "init_ckpt": str(args.init_ckpt),
            "iterations": args.iterations,
            "lr": args.lr,
        },
    }
    torch.save(final_checkpoint, final_ckpt_path)
    
    console.print(f"\n[green]✓[/green] Final optimization complete!")
    console.print(f"Final checkpoint: {final_ckpt_path}")
    console.print(f"Total Gaussians: {len(means)}")
    
    console.print(f"\n[yellow]Note: This is a placeholder implementation.[/yellow]")
    console.print(f"Full implementation would include:")
    console.print(f"  - Dataset loader for training views")
    console.print(f"  - Rendering with gsplat")
    console.print(f"  - Loss computation (RGB + SSIM)")
    console.print(f"  - Validation rendering")
    console.print(f"\nFor now, you can use the optimization code from 05c_optimize_to_targets.py")
    console.print(f"and adapt it to load the merged checkpoint from Module 08.")


if __name__ == "__main__":
    main()
