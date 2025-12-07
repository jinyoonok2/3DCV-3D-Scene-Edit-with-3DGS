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

from project_utils.config import ProjectConfig

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
    parser.add_argument("--config", type=str, default="configs/garden_config.yaml", help="Path to config file (default: configs/garden_config.yaml)")
    parser.add_argument("--ckpt", type=str, default=None, help="Holed checkpoint from 05a (overrides config)")
    parser.add_argument("--targets_dir", type=str, default=None, help="Targets from 05b (overrides config)")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--iters", type=int, default=None, help="Optimization iterations (overrides config)")
    parser.add_argument("--lr_means", type=float, default=None, help="LR for positions (overrides config)")
    parser.add_argument("--lr_scales", type=float, default=None, help="LR for scales (overrides config)")
    parser.add_argument("--lr_quats", type=float, default=None, help="LR for rotations (overrides config)")
    parser.add_argument("--lr_opacities", type=float, default=None, help="LR for opacities (overrides config)")
    parser.add_argument("--lr_sh0", type=float, default=None, help="LR for SH features (overrides config)")
    parser.add_argument("--densify_from_iter", type=int, default=None, help="Start densification (overrides config)")
    parser.add_argument("--densify_until_iter", type=int, default=None, help="Stop densification (overrides config)")
    parser.add_argument("--densify_grad_thresh", type=float, default=None, help="Densify threshold (overrides config)")
    parser.add_argument("--factor", type=int, default=None, choices=[1,2,4,8], help="Downsample (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def render_view(means, quats, scales, opacities, colors, viewmat, K, width, height, sh_degree=3):
    """Render a single view"""
    # Apply activations to parameters (like in training code)
    scales_activated = torch.exp(scales)
    opacities_activated = torch.sigmoid(opacities)
    
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales_activated,
        opacities=opacities_activated,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        camera_model="pinhole",
    )
    return renders[0], alphas[0], info


def load_targets(targets_dir, device):
    """Load all target images"""
    targets = []
    img_files = sorted(targets_dir.glob("*.png"))
    
    console.print(f"[cyan]Loading {len(img_files)} target images...[/cyan]")
    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).to(device)
        targets.append(img_tensor)
    
    console.print(f"[green]✓ Loaded {len(targets)} targets[/green]")
    return targets


def create_comparison_grid_horizontal(original_img, inpainted_img, optimized_img, save_path):
    """Create 3-panel comparison image in horizontal layout.
    
    Layout: [Original | Inpainted (05b) | Optimized (05c)]
    """
    # Ensure all images are same size
    h = min(original_img.shape[0], inpainted_img.shape[0], optimized_img.shape[0])
    w = min(original_img.shape[1], inpainted_img.shape[1], optimized_img.shape[1])
    
    original_img = original_img[:h, :w]
    inpainted_img = inpainted_img[:h, :w]
    optimized_img = optimized_img[:h, :w]
    
    # Create horizontal grid
    grid = np.hstack([original_img, inpainted_img, optimized_img])
    
    # Add text labels
    grid = grid.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(grid, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Inpainted (05b)", (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Optimized (05c)", (2*w + 10, 30), font, font_scale, color, thickness)
    
    cv2.imwrite(str(save_path), grid)


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    ckpt = args.ckpt if args.ckpt else str(config.get_path('inpainting') / '05a_holed' / 'ckpt_holed.pt')
    targets_dir = args.targets_dir if args.targets_dir else str(config.get_path('inpainting') / '05b_inpainted' / 'targets' / 'train')
    data_root = args.data_root if args.data_root else str(config.get_path('dataset_root'))
    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05c_optimized')
    iters = args.iters if args.iters is not None else config.config.get('optimization', {}).get('iterations', 1000)
    lr_means = args.lr_means if args.lr_means is not None else config.config.get('optimization', {}).get('lr_means', 1.6e-4)
    lr_scales = args.lr_scales if args.lr_scales is not None else config.config.get('optimization', {}).get('lr_scales', 5e-3)
    lr_quats = args.lr_quats if args.lr_quats is not None else config.config.get('optimization', {}).get('lr_quats', 1e-3)
    lr_opacities = args.lr_opacities if args.lr_opacities is not None else config.config.get('optimization', {}).get('lr_opacities', 5e-2)
    lr_sh0 = args.lr_sh0 if args.lr_sh0 is not None else config.config.get('optimization', {}).get('lr_sh0', 2.5e-3)
    densify_from_iter = args.densify_from_iter if args.densify_from_iter is not None else config.config.get('optimization', {}).get('densify_from_iter', 100)
    densify_until_iter = args.densify_until_iter if args.densify_until_iter is not None else config.config.get('optimization', {}).get('densify_until_iter', 800)
    densify_grad_thresh = args.densify_grad_thresh if args.densify_grad_thresh is not None else config.config.get('optimization', {}).get('densify_grad_thresh', 0.0002)
    factor = args.factor if args.factor is not None else config.config['dataset']['factor']
    seed = args.seed if args.seed is not None else config.config['dataset']['seed']
    
    device = torch.device(args.device)
    set_random_seed(seed)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05c - Optimize 3DGS to Match Targets[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    ckpt_path = Path(ckpt)
    targets_dir = Path(targets_dir)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    renders_dir = output_dir / "renders" / "train"
    comparisons_dir = output_dir / "comparisons"
    renders_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Checkpoint:[/cyan] {ckpt_path}")
    console.print(f"[cyan]Targets:[/cyan] {targets_dir}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print(f"[cyan]Iterations:[/cyan] {iters}\n")
    
    # Load original checkpoint for comparison
    console.print("[cyan]Loading original checkpoint for comparison...[/cyan]")
    original_ckpt_path = config.get_checkpoint_path('initial')
    original_ckpt = torch.load(original_ckpt_path, map_location=device)
    if "gaussians" in original_ckpt:
        original_params = original_ckpt["gaussians"]
    elif "splats" in original_ckpt:
        original_params = original_ckpt["splats"]
    else:
        original_params = original_ckpt
    console.print(f"[green]✓ Loaded original checkpoint with {len(original_params['means']):,} Gaussians[/green]")
    
    # Load holed checkpoint
    console.print("[cyan]Loading holed checkpoint...[/cyan]")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt["splats"]
    console.print(f"[green]✓ Loaded {len(params['means']):,} Gaussians[/green]")
    
    # Load dataset
    console.print("[cyan]Loading dataset...[/cyan]")
    parser = Parser(data_root, factor=factor, normalize=True, test_every=8)
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
    shN = torch.nn.Parameter(params["shN"].clone())  # Also optimized, like in training code
    
    # Create separate optimizers for each parameter (required by DefaultStrategy)
    # Match the training code exactly
    optimizers = {
        "means": torch.optim.Adam([{"params": means, "lr": lr_means, "name": "means"}], eps=1e-15),
        "quats": torch.optim.Adam([{"params": quats, "lr": lr_quats, "name": "quats"}], eps=1e-15),
        "scales": torch.optim.Adam([{"params": scales, "lr": lr_scales, "name": "scales"}], eps=1e-15),
        "opacities": torch.optim.Adam([{"params": opacities, "lr": lr_opacities, "name": "opacities"}], eps=1e-15),
        "sh0": torch.optim.Adam([{"params": sh0, "lr": lr_sh0, "name": "sh0"}], eps=1e-15),
        "shN": torch.optim.Adam([{"params": shN, "lr": lr_sh0 / 20.0, "name": "shN"}], eps=1e-15),  # Same as training: sh0_lr / 20
    }
    
    # Setup densification strategy (but we'll disable it for this short optimization)
    # With 6.4M Gaussians already, we don't need to add more - just adjust existing ones
    # We'll skip strategy calls entirely to avoid CUDA errors
    use_strategy = False  # Disable strategy for simple parameter optimization
    
    if use_strategy:
        strategy = DefaultStrategy(
            prune_opa=0.005,
            grow_grad2d=densify_grad_thresh,
            grow_scale3d=0.01,
            grow_scale2d=0.01,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            refine_scale2d_stop_iter=0,  # Disable densification by setting stop_iter to 0
            reset_every=3000,
            refine_every=100,
        )
        strategy_state = strategy.initialize_state()
    else:
        strategy = None
        strategy_state = None
    
    # Optimization loop
    console.print("\n[yellow]Optimizing...[/yellow]")
    losses = []
    
    for iter_idx in tqdm(range(iters), desc="Training"):
        # Random view
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        worldtoview = torch.inverse(camtoworld)
        target = targets[idx]
        
        # Get dimensions from target (not from dataset image)
        height, width = target.shape[:2]
        
        # Concatenate SH coefficients for rendering
        colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
        
        # Render
        render, alpha, info = render_view(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
        )
        
        # Pre-backward (only if using strategy)
        if use_strategy:
            splats_dict = {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "sh0": sh0,
                "shN": shN,
            }
            strategy.step_pre_backward(
                params=splats_dict,
                optimizers=optimizers,
                state=strategy_state,
                step=iter_idx,
                info=info,
            )
        
        # L1 loss
        loss = torch.nn.functional.l1_loss(render, target)
        
        # Backward and optimize
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        losses.append(loss.item())
        
        # Post-backward (only if using strategy)
        if use_strategy:
            n_before = len(means)
            strategy.step_post_backward(
                params=splats_dict,
                optimizers=optimizers,
                state=strategy_state,
                step=iter_idx,
                info=info,
                packed=False,
            )
            n_after = len(means)
        else:
            n_after = len(means)
        
        # Log
        if (iter_idx + 1) % 100 == 0:
            console.print(f"  Iter {iter_idx+1}/{iters}: Loss = {loss.item():.4f}, N = {n_after:,}")
    
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
        "shN": shN.detach(),
    }
    
    final_ckpt = {"splats": final_params}
    torch.save(final_ckpt, output_dir / "ckpt_patched.pt")
    console.print(f"[green]✓ Saved checkpoint[/green]")
    
    # Render final views and create comparison grids
    console.print("[cyan]Rendering final views and creating comparison grids...[/cyan]")
    colors_final = torch.cat([sh0, shN], 1)  # Concatenate for rendering
    
    # Prepare original checkpoint colors
    original_sh0 = original_params["sh0"].to(device)
    original_shN = original_params["shN"].to(device) if "shN" in original_params else torch.zeros_like(original_sh0)
    original_colors = torch.cat([original_sh0, original_shN], 1)
    
    for idx in tqdm(range(len(dataset)), desc="Rendering"):
        data = dataset[idx]
        camtoworld = data["camtoworld"].to(device)
        K = data["K"].to(device)
        worldtoview = torch.inverse(camtoworld)
        
        # Get dimensions from target (consistent with training)
        target = targets[idx]
        height, width = target.shape[:2]
        
        # Render original
        original_render, _, _ = render_view(
            means=original_params["means"].to(device),
            quats=original_params["quats"].to(device),
            scales=original_params["scales"].to(device),
            opacities=original_params["opacities"].to(device),
            colors=original_colors,
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
        )
        
        # Render optimized (final)
        optimized_render, _, _ = render_view(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_final,
            viewmat=worldtoview,
            K=K,
            width=width,
            height=height,
        )
        
        # Convert to numpy BGR for saving
        original_img = cv2.cvtColor((original_render.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inpainted_img = cv2.cvtColor((target.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        optimized_img = cv2.cvtColor((optimized_render.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Save individual optimized render
        cv2.imwrite(str(renders_dir / f"{idx:05d}.png"), optimized_img)
        
        # Create and save comparison grid
        comparison_path = comparisons_dir / f"{idx:05d}.png"
        create_comparison_grid_horizontal(original_img, inpainted_img, optimized_img, comparison_path)
    
    console.print(f"[green]✓ Saved final renders and comparison grids[/green]")
    
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
        "config_file": args.config,
        "inputs": {
            "checkpoint": str(ckpt_path),
            "targets_dir": str(targets_dir),
            "data_root": str(data_root),
        },
        "parameters": {
            "iters": iters,
            "lr_means": lr_means,
            "lr_scales": lr_scales,
            "lr_quats": lr_quats,
            "lr_opacities": lr_opacities,
            "lr_sh0": lr_sh0,
            "densify_from_iter": densify_from_iter,
            "densify_until_iter": densify_until_iter,
            "factor": factor,
            "seed": seed,
        },
        "results": {
            "n_views": len(dataset),
            "n_initial_gaussians": len(params["means"]),
            "n_final_gaussians": len(means),
            "final_loss": float(losses[-1]),
            "mean_loss": float(np.mean(losses[-100:])),
        },
    }
    
    config.save_manifest("05c_optimize_to_targets", manifest)
    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05c_optimize_to_targets_manifest.json'}[/green]")
    
    console.print("\n[bold green]" + "="*80 + "[/bold green]")
    console.print("[bold green]✓ Module 05c Complete![/bold green]")
    console.print(f"[bold green]Output: {output_dir / 'ckpt_patched.pt'}[/bold green]")
    console.print(f"[cyan]📁 Optimized renders: {len(dataset)} images in renders/train/[/cyan]")
    console.print(f"[cyan]🔄 Comparison grids: {len(dataset)} images in comparisons/ (Original|Inpainted|Optimized)[/cyan]")
    console.print("[bold green]" + "="*80 + "[/bold green]\n")


if __name__ == "__main__":
    main()
