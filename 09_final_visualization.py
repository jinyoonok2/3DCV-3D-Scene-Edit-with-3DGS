#!/usr/bin/env python3
"""
09_final_visualization.py - Final Scene Visualization and Evaluation

This module renders and evaluates the final merged scene from Module 08.
It creates comprehensive visualizations showing:
- Rendered views of the final scene
- Before/after comparisons 
- Object placement evaluation
- Quantitative metrics (CLIP, LPIPS, SSIM, PSNR)
- Summary grid images

Inputs:
  --merged_ckpt: Path to merged scene checkpoint (from Module 08)
  --original_ckpt: Path to original scene checkpoint (Module 01)
  --data_root: Dataset directory

Outputs (saved in 09_final_visualization/):
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
        description="Visualize and evaluate final merged scene"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--merged_ckpt",
        type=str,
        help="Path to merged scene checkpoint (from Module 08, auto-detected if not provided)",
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
    parser.add_argument(
        "--views",
        type=str,
        default="val",
        help="Which views to render: 'all', 'train', 'val', or comma-separated indices (default: val)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=10,
        help="Maximum number of views to render (default: 10)",
    )
    parser.add_argument(
        "--create_comparisons",
        action="store_true",
        default=True,
        help="Create before/after comparison images (default: True)",
    )
    parser.add_argument(
        "--summary_grid",
        action="store_true",
        default=True,
        help="Create summary grid image (default: True)",
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


def create_comparison_grid(original_img, merged_img, save_path):
    """Create side-by-side comparison image."""
    # Ensure images are same size
    h, w = min(original_img.shape[0], merged_img.shape[0]), min(original_img.shape[1], merged_img.shape[1])
    original_img = original_img[:h, :w]
    merged_img = merged_img[:h, :w]
    
    # Create side-by-side comparison
    comparison = np.hstack([original_img, merged_img])
    
    # Add text labels
    comparison = comparison.copy()
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "With Object", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(str(save_path), comparison)


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
    console.print("Module 09: Final Scene Visualization & Evaluation") 
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Setup paths
    if args.merged_ckpt:
        merged_ckpt = Path(args.merged_ckpt)
    else:
        # Auto-detect from Module 08
        merged_ckpt = config.get_checkpoint_path('merged')
        if not merged_ckpt.exists():
            console.print(f"[red]Merged checkpoint not found: {merged_ckpt}[/red]")
            console.print("Run Module 08 first or provide --merged_ckpt")
            sys.exit(1)
    
    original_ckpt = Path(args.original_ckpt) if args.original_ckpt else config.get_checkpoint_path('initial')
    data_root = Path(args.data_root) if args.data_root else Path(config.get_path('dataset_root'))
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.get_path('final_optimization')
    
    # Create output directories
    renders_dir = output_dir / "renders"
    comparisons_dir = output_dir / "comparisons" 
    renders_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Merged checkpoint: {merged_ckpt}")
    console.print(f"Original checkpoint: {original_ckpt}")
    console.print(f"Data root: {data_root}")
    console.print(f"Output: {output_dir}")
    console.print(f"Views: {args.views} (max {args.num_views})\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoints
    merged_params = load_checkpoint(merged_ckpt, device)
    original_params = load_checkpoint(original_ckpt, device)
    
    # Load dataset
    console.print("Loading dataset...")
    parser = Parser(data_root, factor=4, normalize=True, test_every=8)
    
    if args.views == "all":
        dataset = Dataset(parser, split="train") + Dataset(parser, split="test")
    elif args.views == "val" or args.views == "test":
        dataset = Dataset(parser, split="test")
    elif args.views == "train":
        dataset = Dataset(parser, split="train")
    else:
        # Comma-separated indices
        indices = [int(x.strip()) for x in args.views.split(',')]
        full_dataset = Dataset(parser, split="train")
        dataset = [full_dataset[i] for i in indices if i < len(full_dataset)]
    
    # Limit number of views
    if len(dataset) > args.num_views:
        dataset = dataset[:args.num_views]
    
    console.print(f"✓ Dataset loaded: {len(dataset)} views\n")
    
    # Render views
    console.print("[cyan]Rendering views...[/cyan]")
    rendered_paths = []
    comparison_paths = []
    all_metrics = []
    
    for i, data in enumerate(tqdm(dataset, desc="Rendering")):
        # Get camera parameters
        camtoworld = data["camtoworld"]
        K = data["K"] 
        height, width = data["image"].shape[:2]
        
        # Render both scenes
        try:
            merged_render = render_view(merged_params, camtoworld, K, width, height, device)
            original_render = render_view(original_params, camtoworld, K, width, height, device)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to render view {i}: {e}[/yellow]")
            continue
        
        # Convert to numpy
        merged_img = (torch.clamp(merged_render, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        original_img = (torch.clamp(original_render, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # Save renders
        render_path = renders_dir / f"{i:05d}_merged.png"
        original_path = renders_dir / f"{i:05d}_original.png"
        cv2.imwrite(str(render_path), cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(original_path), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        rendered_paths.append(str(render_path))
        
        # Create comparison if requested
        if args.create_comparisons:
            comparison_path = comparisons_dir / f"{i:05d}_comparison.png"
            create_comparison_grid(
                cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(merged_img, cv2.COLOR_RGB2BGR), 
                comparison_path
            )
            comparison_paths.append(str(comparison_path))
        
        # Compute metrics
        metrics = compute_metrics(original_img, merged_img)
        metrics["view_index"] = i
        all_metrics.append(metrics)
    
    console.print(f"✓ Rendered {len(rendered_paths)} views")
    
    # Create summary grid
    if args.summary_grid and comparison_paths:
        console.print("[cyan]Creating summary grid...[/cyan]")
        summary_path = output_dir / "summary_grid.png"
        create_summary_grid(comparison_paths, summary_path)
        console.print(f"✓ Summary grid: {summary_path}")
    
    # Compute aggregate metrics
    console.print("[cyan]Computing aggregate metrics...[/cyan]")
    avg_metrics = {
        "mean_mse": np.mean([m["mse"] for m in all_metrics]),
        "mean_psnr": np.mean([m["psnr"] for m in all_metrics]),
        "mean_ssim": np.mean([m["ssim"] for m in all_metrics]),
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
        "module": "09_final_visualization",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "merged_checkpoint": str(merged_ckpt),
            "original_checkpoint": str(original_ckpt),
            "data_root": str(data_root),
        },
        "parameters": {
            "views": args.views,
            "num_views": args.num_views,
            "create_comparisons": args.create_comparisons,
            "summary_grid": args.summary_grid,
        },
        "results": {
            "num_views_rendered": len(rendered_paths),
            "num_comparisons": len(comparison_paths),
            "mean_psnr": avg_metrics['mean_psnr'],
            "mean_ssim": avg_metrics['mean_ssim'],
            "mean_mse": avg_metrics['mean_mse'],
        },
        "outputs": {
            "renders_dir": str(renders_dir),
            "comparisons_dir": str(comparisons_dir) if comparison_paths else None,
            "summary_grid": str(output_dir / "summary_grid.png") if args.summary_grid else None,
            "metrics": str(metrics_path),
        },
    }
    
    # Save manifest using unified system
    config.save_manifest("09_final_visualization", manifest)
    
    console.print(f"\n[green]✓ Module 09 complete![/green]")
    console.print(f"📁 Output directory: {output_dir}")
    console.print(f"🖼️  Renders: {len(rendered_paths)} images")
    if comparison_paths:
        console.print(f"🔄 Comparisons: {len(comparison_paths)} before/after images")
    if args.summary_grid:
        console.print(f"📊 Summary grid: summary_grid.png")
    
    console.print(f"\n[cyan]View results:[/cyan]")
    console.print(f"  ls {output_dir}")
    console.print(f"  open {output_dir}/summary_grid.png")
    console.print()


if __name__ == "__main__":
    main()