#!/usr/bin/env python3
"""
07_final_visualization.py - Final Scene Visualization and Evaluation

This module renders and evaluates the final merged scene from Module 06.
It creates comprehensive visualizations showing:
- Rendered views of the final scene
- Before/after comparisons 
- Object placement evaluation
- Quantitative metrics (CLIP, LPIPS, SSIM, PSNR)
- Summary grid images

Inputs:
  --merged_ckpt: Path to merged scene checkpoint (from Module 06)
  --original_ckpt: Path to original scene checkpoint (Module 01)
  --data_root: Dataset directory

Outputs (saved in 07_final_visualization/):
  - renders/: Final scene renders from all viewpoints
  - comparisons/: Before/after comparison grids
  - metrics.json: Quantitative evaluation results
  - summary_grid.png: Overview visualization
  - README.md: Module summary (auto-generated)
  - manifest.json: Complete evaluation metadata
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from tqdm import tqdm

from project_utils.config import ProjectConfig

console = Console()

# Add gsplat to path
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
    parser = argparse.ArgumentParser(
        description="Visualize and evaluate final merged scene with 4-stage comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/garden_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--merged_ckpt",
        type=str,
        help="Path to merged scene checkpoint (from Module 06, auto-detected if not provided)",
    )
    parser.add_argument(
        "--optimized_ckpt",
        type=str,
        help="Path to optimized checkpoint (from Module 05c, auto-detected if not provided)",
    )
    parser.add_argument(
        "--inpainted_ckpt",
        type=str,
        help="Path to removed/holed checkpoint (from Module 05a, auto-detected if not provided)",
    )
    parser.add_argument(
        "--original_ckpt",
        type=str,
        help="Path to original scene checkpoint (overrides config)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Dataset root directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides config)",
    )
    
    return parser.parse_args()


def load_checkpoint(ckpt_path, device="cuda"):
    """Load checkpoint and extract Gaussian parameters."""
    console.print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if "gaussians" in ckpt:
        params = ckpt["gaussians"]
    elif "splats" in ckpt:
        params = ckpt["splats"] 
    else:
        params = ckpt
        
    console.print(f"✓ Loaded {len(params['means']):,} Gaussians")
    return params


def render_view(params, camtoworld, K, width, height, device="cuda", sh_degree=3):
    """Render a single view using gsplat."""
    # Move parameters to device
    means = params["means"].to(device)
    quats = params["quats"].to(device)
    scales = torch.exp(params["scales"]).to(device)
    opacities = torch.sigmoid(params["opacities"]).to(device)
    
    # Handle colors (concatenate sh0 and shN)
    sh0 = params["sh0"].to(device)
    if "shN" in params and params["shN"].numel() > 0:
        shN = params["shN"].to(device)
        colors = torch.cat([sh0, shN], -2)  # Concatenate along SH dimension
    else:
        colors = sh0
    
    # Ensure colors have correct shape [N, (sh_degree+1)^2, 3]
    if colors.shape[-2] < (sh_degree + 1) ** 2:
        # Pad with zeros if not enough SH coefficients
        needed = (sh_degree + 1) ** 2 - colors.shape[-2]
        padding = torch.zeros(colors.shape[0], needed, 3, device=device)
        colors = torch.cat([colors, padding], -2)
    
    # Move camera parameters to device
    camtoworld = camtoworld.to(device)
    K = K.to(device)
    
    # Render
    render_colors, _, _ = rasterization(
        means=means,
        quats=quats / quats.norm(dim=-1, keepdim=True),  # Normalize quaternions
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.inverse(camtoworld)[None],  # World to camera
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
    )
    
    return render_colors.squeeze(0)  # Remove batch dimension


def create_comparison_grid(original_img, inpainted_img, optimized_img, merged_img, save_path):
    """Create 4-panel comparison image in 2x2 grid.
    
    Layout:
    Top-left: Original
    Top-right: Inpainted/Removed (05a)
    Bottom-left: Optimized (05c)
    Bottom-right: Final with Object (merged)
    """
    # Ensure all images are same size
    h = min(original_img.shape[0], inpainted_img.shape[0], optimized_img.shape[0], merged_img.shape[0])
    w = min(original_img.shape[1], inpainted_img.shape[1], optimized_img.shape[1], merged_img.shape[1])
    
    original_img = original_img[:h, :w]
    inpainted_img = inpainted_img[:h, :w]
    optimized_img = optimized_img[:h, :w]
    merged_img = merged_img[:h, :w]
    
    # Create 2x2 grid
    top_row = np.hstack([original_img, inpainted_img])
    bottom_row = np.hstack([optimized_img, merged_img])
    grid = np.vstack([top_row, bottom_row])
    
    # Add text labels
    grid = grid.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(grid, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Removed", (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(grid, "Optimized", (10, h + 30), font, font_scale, color, thickness)
    cv2.putText(grid, "With Object", (w + 10, h + 30), font, font_scale, color, thickness)
    
    cv2.imwrite(str(save_path), grid)


def compute_metrics(img1, img2):
    """Compute basic image metrics between two images."""
    # Convert to float [0, 1]
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0
    
    # MSE and PSNR
    mse = np.mean((img1_f - img2_f) ** 2)
    psnr = -10 * np.log10(mse + 1e-8)
    
    # SSIM (simplified version)
    def ssim_simple(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return ssim
    
    ssim = ssim_simple(img1_f, img2_f)
    
    return {"mse": float(mse), "psnr": float(psnr), "ssim": float(ssim)}


def create_summary_grid(image_paths, save_path, grid_size=(2, 3)):
    """Create a grid of images for summary visualization."""
    if len(image_paths) == 0:
        return
        
    # Load first few images
    images = []
    for img_path in image_paths[:grid_size[0] * grid_size[1]]:
        if Path(img_path).exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to standard size
                img = cv2.resize(img, (512, 384))
                images.append(img)
    
    if not images:
        return
        
    # Create grid
    rows = []
    for i in range(grid_size[0]):
        row_imgs = images[i*grid_size[1]:(i+1)*grid_size[1]]
        if row_imgs:
            # Pad row if needed
            while len(row_imgs) < grid_size[1]:
                row_imgs.append(np.zeros_like(row_imgs[0]))
            row = np.hstack(row_imgs)
            rows.append(row)
    
    if rows:
        grid = np.vstack(rows)
        cv2.imwrite(str(save_path), grid)


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 07: Final Scene Visualization (4-Stage Comparison)") 
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    project_name = config.get("project", "name")
    
    # Setup checkpoint paths
    if args.merged_ckpt:
        merged_ckpt = Path(args.merged_ckpt)
    else:
        merged_ckpt = config.get_path('scene_placement') / 'merged_gaussians.pt'
    
    if args.optimized_ckpt:
        optimized_ckpt = Path(args.optimized_ckpt)
    else:
        optimized_ckpt = config.get_path('inpainting') / '05c_optimized' / 'ckpt_final.pt'
    
    if args.inpainted_ckpt:
        inpainted_ckpt = Path(args.inpainted_ckpt)
    else:
        # This is the removed/holed checkpoint from 05a (before inpainting)
        inpainted_ckpt = config.get_path('inpainting') / '05a_holed' / 'ckpt_holed.pt'
    
    if args.original_ckpt:
        original_ckpt = Path(args.original_ckpt)
    else:
        original_ckpt = config.get_checkpoint_path('initial')
    
    # Verify all checkpoints exist
    for name, ckpt_path in [
        ("Merged", merged_ckpt),
        ("Optimized", optimized_ckpt),
        ("Removed", inpainted_ckpt),
        ("Original", original_ckpt)
    ]:
        if not ckpt_path.exists():
            console.print(f"[red]{name} checkpoint not found: {ckpt_path}[/red]")
            sys.exit(1)
    
    data_root = Path(args.data_root) if args.data_root else Path(config.get_path('dataset_root'))
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.get_path('final_visualization')
    
    # Create output directories
    merged_dir = output_dir / "merged"  # Final results only
    comparisons_dir = output_dir / "comparisons"  # 4-panel grids
    merged_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Original checkpoint: {original_ckpt}")
    console.print(f"Removed checkpoint: {inpainted_ckpt}")
    console.print(f"Optimized checkpoint: {optimized_ckpt}")
    console.print(f"Merged checkpoint: {merged_ckpt}")
    console.print(f"Data root: {data_root}")
    console.print(f"Output: {output_dir}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all checkpoints
    console.print("[cyan]Loading checkpoints...[/cyan]")
    original_params = load_checkpoint(original_ckpt, device)
    removed_params = load_checkpoint(inpainted_ckpt, device)  # This is 05a removed/holed
    optimized_params = load_checkpoint(optimized_ckpt, device)
    merged_params = load_checkpoint(merged_ckpt, device)
    console.print("✓ All checkpoints loaded\n")
    
    # Load dataset - ALL train + val views
    console.print("Loading dataset...")
    parser = Parser(data_root, factor=4, normalize=True, test_every=8)
    train_dataset = Dataset(parser, split="train")
    val_dataset = Dataset(parser, split="test")
    
    # Combine all views
    all_views = list(train_dataset) + list(val_dataset)
    console.print(f"✓ Dataset loaded: {len(train_dataset)} train + {len(val_dataset)} val = {len(all_views)} total views\n")
    
    # Render all views
    console.print("[cyan]Rendering all views...[/cyan]")
    comparison_paths = []
    all_metrics = []
    
    for i, data in enumerate(tqdm(all_views, desc="Rendering")):
        # Get camera parameters
        camtoworld = data["camtoworld"]
        K = data["K"] 
        height, width = data["image"].shape[:2]
        
        # Render all 4 stages
        try:
            original_render = render_view(original_params, camtoworld, K, width, height, device)
            removed_render = render_view(removed_params, camtoworld, K, width, height, device)
            optimized_render = render_view(optimized_params, camtoworld, K, width, height, device)
            merged_render = render_view(merged_params, camtoworld, K, width, height, device)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to render view {i}: {e}[/yellow]")
            continue
        
        # Convert to numpy BGR for OpenCV
        original_img = cv2.cvtColor((torch.clamp(original_render, 0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        removed_img = cv2.cvtColor((torch.clamp(removed_render, 0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        optimized_img = cv2.cvtColor((torch.clamp(optimized_render, 0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        merged_img = cv2.cvtColor((torch.clamp(merged_render, 0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Save merged result (final image only)
        merged_path = merged_dir / f"{i:05d}.png"
        cv2.imwrite(str(merged_path), merged_img)
        
        # Create 4-panel comparison grid
        comparison_path = comparisons_dir / f"{i:05d}.png"
        create_comparison_grid(original_img, removed_img, optimized_img, merged_img, comparison_path)
        comparison_paths.append(str(comparison_path))
        
        # Compute metrics (original vs final merged)
        metrics = compute_metrics(
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)
        )
        metrics["view_index"] = i
        all_metrics.append(metrics)
    
    console.print(f"✓ Rendered {len(comparison_paths)} views")
    
    # Create summary grid (optional - select a few samples)
    if comparison_paths:
        console.print("[cyan]Creating summary grid...[/cyan]")
        summary_path = output_dir / "summary_grid.png"
        # Sample 9 evenly spaced views for summary
        sample_indices = np.linspace(0, len(comparison_paths)-1, min(9, len(comparison_paths)), dtype=int)
        sample_paths = [comparison_paths[i] for i in sample_indices]
        create_summary_grid(sample_paths, summary_path)
        console.print(f"✓ Summary grid: {summary_path}")
    
    # Compute aggregate metrics
    console.print("[cyan]Computing aggregate metrics...[/cyan]")
    avg_metrics = {
        "mean_mse": np.mean([m["mse"] for m in all_metrics]),
        "mean_psnr": np.mean([m["psnr"] for m in all_metrics]),
        "mean_ssim": np.mean([m["ssim"] for m in all_metrics]),
        "num_views": len(all_metrics),
        "per_view_metrics": all_metrics,
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    console.print(f"✓ Metrics saved: {metrics_path}")
    console.print(f"  - Mean PSNR: {avg_metrics['mean_psnr']:.2f}")
    console.print(f"  - Mean SSIM: {avg_metrics['mean_ssim']:.3f}")
    
    # Create manifest
    manifest = {
        "module": "07_final_visualization",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "original_checkpoint": str(original_ckpt),
            "removed_checkpoint": str(inpainted_ckpt),
            "optimized_checkpoint": str(optimized_ckpt),
            "merged_checkpoint": str(merged_ckpt),
            "data_root": str(data_root),
        },
        "results": {
            "num_views_rendered": len(comparison_paths),
            "num_train_views": len(train_dataset),
            "num_val_views": len(val_dataset),
            "mean_psnr": avg_metrics['mean_psnr'],
            "mean_ssim": avg_metrics['mean_ssim'],
            "mean_mse": avg_metrics['mean_mse'],
        },
        "outputs": {
            "merged_dir": str(merged_dir),
            "comparisons_dir": str(comparisons_dir),
            "summary_grid": str(output_dir / "summary_grid.png") if comparison_paths else None,
            "metrics": str(metrics_path),
        },
    }
    
    # Save manifest using unified system
    config.save_manifest("07_final_visualization", manifest)
    
    console.print(f"\n[green]✓ Module 07 complete![/green]")
    console.print(f"📁 Output directory: {output_dir}")
    console.print(f"🖼️  Merged renders: {len(comparison_paths)} images in merged/")
    console.print(f"🔄 Comparison grids: {len(comparison_paths)} images in comparisons/ (2x2: Original|Removed|Optimized|Final)")
    console.print(f"\n[cyan]💡 To create GIF from merged results:[/cyan]")
    console.print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{merged_dir}/*.png' -vf scale=1920:-1 {output_dir}/merged_animation.gif")
    console.print(f"\n[cyan]💡 To create GIF from 4-panel comparisons:[/cyan]")
    console.print(f"  ffmpeg -framerate 10 -pattern_type glob -i '{comparisons_dir}/*.png' -vf scale=1920:-1 {output_dir}/comparison_animation.gif")
    console.print(f"\n[cyan]View results:[/cyan]")
    console.print(f"  ls {merged_dir}")
    console.print(f"  ls {comparisons_dir}")
    console.print()


if __name__ == "__main__":
    main()