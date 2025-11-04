#!/usr/bin/env python3
"""
05c_optimize_to_targets.py - Optimize 3DGS to Match Inpainted Targets

Goal: Fine-tune 3D Gaussians to match the inpainted 2D target views.

Approach:
  1. Load holed checkpoint from 05a
  2. Load inpainted targets from 05b
  3. Optimize Gaussian parameters to minimize L1 loss between renders and targets
  4. Use densification strategy to grow new Gaussians in hole regions
  5. Save optimized checkpoint

Inputs:
  --ckpt: Path to holed checkpoint (from 05a)
  --targets_dir: Directory with inpainted targets (from 05b)
  --data_root: Dataset root for camera poses
  --output_dir: Output directory (default: sibling of targets_dir)
  --iters: Optimization iterations (default: 1000)
  --lr_*: Learning rates for different parameters
  --densify_from_iter: Start densification (default: 100)
  --densify_until_iter: Stop densification (default: 800)

Outputs (saved in output_dir/05c_optimized/):
  - ckpt_patched.pt: Final optimized checkpoint
  - renders/train/*.png: Final rendered views
  - loss_curve.png: Training loss plot
  - manifest.json: Metadata
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from rich.console import Console

console = Console()

sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))

try:
    from datasets.colmap import Dataset, Parser
    from utils import set_random_seed
except ImportError:
    console.print("[red]ERROR: Could not import from gsplat examples.[/red]")
    sys.exit(1)

try:
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
except ImportError:
    console.print("[red]ERROR: gsplat not installed.[/red]")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize 3DGS to match targets")
    parser.add_argument("--ckpt", type=str, required=True, help="Holed checkpoint from 05a")
    parser.add_argument("--targets_dir", type=str, required=True, help="Targets from 05b")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--iters", type=int, default=1000, help="Optimization iterations")
    parser.add_argument("--lr_means", type=float, default=1.6e-4, help="LR for positions")
    parser.add_argument("--lr_scales", type=float, default=5e-3, help="LR for scales")
    parser.add_argument("--lr_quats", type=float, default=1e-3, help="LR for rotations")
    parser.add_argument("--lr_opacities", type=float, default=5e-2, help="LR for opacities")
    parser.add_argument("--lr_sh0", type=float, default=2.5e-3, help="LR for SH features")
    parser.add_argument("--densify_from_iter", type=int, default=100, help="Start densification")
    parser.add_argument("--densify_until_iter", type=int, default=800, help="Stop densification")
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0002, help="Densify threshold")
    parser.add_argument("--factor", type=int, default=4, choices=[1,2,4,8], help="Downsample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def render_view(means, quats, scales, opacities, colors, viewmat, K, width, height):
    """Render a single view"""
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        packed=False,
        sh_degree=None,
        render_mode="RGB",
    )
    return renders[0], alphas[0], info


def load_targets(targets_dir, device):
    """Load all target images"""
    targets = []
    img_files = sorted((targets_dir / "train").glob("*.png"))
    
    console.print(f"[cyan]Loading {len(img_files)} target images...[/cyan]")
    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).to(device)
        targets.append(img_tensor)
    
    console.print(f"[green]✓ Loaded {len(targets)} targets[/green]")
    return targets


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_random_seed(args.seed)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05c - Optimize 3DGS to Match Targets[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    ckpt_path = Path(args.ckpt)
    targets_dir = Path(args.targets_dir)
    data_root = Path(args.data_root)
    
    if args.output_dir is None:
        output_dir = targets_dir.parent / "05c_optimized"
    else:
        output_dir = Path(args.output_dir)
    
    renders_dir = output_dir / "renders" / "train"
    renders_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Checkpoint:[/cyan] {ckpt_path}")
    console.print(f"[cyan]Targets:[/cyan] {targets_dir}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print(f"[cyan]Iterations:[/cyan] {args.iters}\n")
    
    # Load checkpoint
    console.print("[cyan]Loading holed checkpoint...[/cyan]")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt["splats"]
    console.print(f"[green]✓ Loaded {len(params['means']):,} Gaussians[/green]")
    
    # Load dataset
    console.print("[cyan]Loading dataset...[/cyan]")
    parser = Parser(data_root, factor=args.factor, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train")
    console.print(f"[green]✓ Loaded {len(dataset)} training views[/green]")
    
    # Load targets
    targets = load_targets(targets_dir, device)
    
    if len(targets) != len(dataset):
        console.print(f"[red]ERROR: Number of targets ({len(targets)}) != dataset size ({len(dataset)})[/red]")
        sys.exit(1)
    
    # Setup optimizable parameters
    means = torch.nn.Parameter(params["means"].clone())
    quats = torch.nn.Parameter(params["quats"].clone())
    scales = torch.nn.Parameter(params["scales"].clone())
    opacities = torch.nn.Parameter(params["opacities"].clone())
    sh0 = torch.nn.Parameter(params["sh0"].clone())
    
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": args.lr_means, "name": "means"},
        {"params": [quats], "lr": args.lr_quats, "name": "quats"},
        {"params": [scales], "lr": args.lr_scales, "name": "scales"},
        {"params": [opacities], "lr": args.lr_opacities, "name": "opacities"},
        {"params": [sh0], "lr": args.lr_sh0, "name": "sh0"},
    ])
    
    # Setup densification strategy
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=args.densify_grad_thresh,
        grow_scale3d=0.01,
        grow_scale2d=0.01,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
        refine_scale2d_stop_iter=args.densify_until_iter,
        reset_every=3000,
        refine_every=100,
    )
    
    strategy_state = strategy.initialize_state()
    
    # Optimization loop
    console.print("\n[yellow]Optimizing...[/yellow]")
    losses = []
    
    for iter_idx in tqdm(range(args.iters), desc="Training"):
        # Random view
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        width = data["width"]
        height = data["height"]
        worldtoview = torch.inverse(camtoworld)
        target = targets[idx]
        
        # Render
        render, alpha, info = render_view(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh0,
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
        )
        
        # L1 loss
        loss = torch.nn.functional.l1_loss(render, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Store gradients for densification
        if args.densify_from_iter <= iter_idx < args.densify_until_iter:
            if info["means2d"].grad is not None:
                strategy_state = strategy.collect_statistics(
                    iter=iter_idx,
                    state=strategy_state,
                    params={"means2d": info["means2d"]},
                    optimizers={"means2d": optimizer},
                )
        
        optimizer.step()
        losses.append(loss.item())
        
        # Densification
        if args.densify_from_iter <= iter_idx < args.densify_until_iter:
            if iter_idx % 100 == 0:
                n_before = len(means)
                
                # Apply densification
                strategy_state = strategy.step(
                    iter=iter_idx,
                    state=strategy_state,
                    params={
                        "means": means,
                        "quats": quats,
                        "scales": scales,
                        "opacities": opacities,
                        "sh0": sh0,
                    },
                    optimizers={"all": optimizer},
                )
                
                n_after = len(means)
                if n_after != n_before:
                    console.print(f"  Iter {iter_idx}: Densified {n_before:,} → {n_after:,} Gaussians")
        
        # Log
        if (iter_idx + 1) % 100 == 0:
            console.print(f"  Iter {iter_idx+1}/{args.iters}: Loss = {loss.item():.4f}, N = {len(means):,}")
    
    console.print(f"[green]✓ Optimization complete (final loss: {losses[-1]:.4f})[/green]")
    console.print(f"[green]✓ Final Gaussian count: {len(means):,}[/green]")
    
    # Save checkpoint
    console.print("\n[yellow]Saving results...[/yellow]")
    
    final_params = {
        "means": means.detach(),
        "quats": quats.detach(),
        "scales": scales.detach(),
        "opacities": opacities.detach(),
        "sh0": sh0.detach(),
    }
    
    final_ckpt = {"splats": final_params}
    torch.save(final_ckpt, output_dir / "ckpt_patched.pt")
    console.print(f"[green]✓ Saved checkpoint[/green]")
    
    # Render final views
    console.print("[cyan]Rendering final views...[/cyan]")
    for idx in tqdm(range(len(dataset)), desc="Rendering"):
        data = dataset[idx]
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        width = data["width"]
        height = data["height"]
        worldtoview = torch.inverse(camtoworld)
        
        render, _, _ = render_view(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh0,
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
        )
        
        img_np = (render.detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(renders_dir / f"{idx:05d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    console.print(f"[green]✓ Saved final renders[/green]")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("L1 Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ Saved loss curve[/green]")
    
    # Save manifest
    manifest = {
        "module": "05c_optimize_to_targets",
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "checkpoint": str(ckpt_path),
            "targets_dir": str(targets_dir),
            "data_root": str(data_root),
        },
        "parameters": {
            "iters": args.iters,
            "lr_means": args.lr_means,
            "lr_scales": args.lr_scales,
            "lr_quats": args.lr_quats,
            "lr_opacities": args.lr_opacities,
            "lr_sh0": args.lr_sh0,
            "densify_from_iter": args.densify_from_iter,
            "densify_until_iter": args.densify_until_iter,
            "factor": args.factor,
            "seed": args.seed,
        },
        "results": {
            "n_views": len(dataset),
            "n_initial_gaussians": len(params["means"]),
            "n_final_gaussians": len(means),
            "final_loss": float(losses[-1]),
            "mean_loss": float(np.mean(losses[-100:])),
        },
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    console.print("\n[bold green]" + "="*80 + "[/bold green]")
    console.print("[bold green]✓ Module 05c Complete![/bold green]")
    console.print(f"[bold green]Output: {output_dir / 'ckpt_patched.pt'}[/bold green]")
    console.print("[bold green]" + "="*80 + "[/bold green]\n")


if __name__ == "__main__":
    main()
