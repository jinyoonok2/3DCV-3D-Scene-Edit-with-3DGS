#!/usr/bin/env python3
"""
05b_inpaint_holes.py - Inpaint Holes with LaMa

Goal: Use LaMa inpainting model to fill holes in rendered views.
      LaMa provides clean object removal without hallucination.

Approach:
  1. Load holed renders and masks from 05a
  2. For each view, run LaMa inpainting to fill the hole
  3. Save inpainted target images

Inputs:
  --holed_dir: Directory with holed renders (from 05a)
  --output_dir: Output directory (default: sibling of holed_dir)
  --mask_blur: Mask blur radius (default: 8)
  --mask_erode: Erode mask by N pixels (shrink hole)
  --mask_dilate: Dilate mask by N pixels (expand hole)

Outputs (saved in output_dir/05b_inpainted/):
  - targets/train/*.png: Inpainted target images
  - manifest.json: Metadata

Note: For SDXL-based inpainting (ablation study), use 05b_inpaint_holes_sdxl.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from rich.console import Console

from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint holes with LaMa")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--mask_blur", type=int, default=8, help="Mask blur radius (default: 8)")
    parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")
    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate mask by N pixels (expand hole)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def load_lama_inpainting(device):
    """Load LaMa Inpainting model"""
    from simple_lama_inpainting import SimpleLama
    
    console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")
    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")
    
    simple_lama = SimpleLama(device=device)
    console.print("[green]✓ LaMa Inpainting loaded[/green]")
    return simple_lama


def process_mask(mask, blur_radius=8, erode=0, dilate=0):
    """Process mask with erosion/dilation and blurring"""
    import cv2
    
    mask_np = np.array(mask)
    
    # Erode (shrink mask)
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)
    
    # Dilate (expand mask)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Blur edges
    if blur_radius > 0:
        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), 0)
    
    return Image.fromarray(mask_np)


def inpaint_view_lama(lama_model, image_path, mask_path, mask_blur=8, mask_erode=0, mask_dilate=0):
    """Inpaint a single view with LaMa (no prompts, pure texture continuation)"""
    # Load images
    img_pil = Image.open(image_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    
    # Process mask for inpainting (with dilation/blur)
    mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)
    
    # Run LaMa inpainting (simple API - no prompts, no seeds)
    result = lama_model(img_pil, mask_pil_processed)
    
    return result


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')
    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_inpainted')
    
    device = torch.device(args.device)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05b - Inpaint Holes with LaMa[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    holed_dir = Path(holed_dir)
    renders_dir = holed_dir / "renders" / "train"
    masks_dir = holed_dir / "masks" / "train"
    
    if not renders_dir.exists() or not masks_dir.exists():
        console.print(f"[red]ERROR: Could not find renders or masks in {holed_dir}[/red]")
        console.print(f"[red]Make sure to run 05a first![/red]")
        sys.exit(1)
    
    output_dir = Path(output_dir)
    targets_dir = output_dir / "targets" / "train"
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Input directory:[/cyan] {holed_dir}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Model:[/cyan] LaMa")
    console.print("")
    
    # Get image list
    image_files = sorted(renders_dir.glob("*.png"))
    n_images = len(image_files)
    console.print(f"[cyan]Found {n_images} images to inpaint[/cyan]\n")
    
    # Load LaMa model
    lama_model = load_lama_inpainting(device)
    
    # Inpaint each view
    console.print("\n[yellow]Inpainting views...[/yellow]")
    
    for img_path in tqdm(image_files, desc="Inpainting"):
        idx = img_path.stem  # e.g., "00042"
        mask_path = masks_dir / f"{idx}.png"
        
        if not mask_path.exists():
            console.print(f"[yellow]Warning: No mask for {idx}, skipping[/yellow]")
            continue
        
        # Inpaint with LaMa
        result = inpaint_view_lama(
            lama_model=lama_model,
            image_path=img_path,
            mask_path=mask_path,
            mask_blur=args.mask_blur,
            mask_erode=args.mask_erode,
            mask_dilate=args.mask_dilate,
        )
        
        # Save
        result.save(targets_dir / f"{idx}.png")
    
    console.print(f"[green]✓ Saved inpainted targets to {targets_dir}[/green]")
    
    # Save manifest
    manifest = {
        "module": "05b_inpaint_holes",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "holed_dir": str(holed_dir),
        },
        "parameters": {
            "model": "lama",
            "mask_blur": args.mask_blur,
            "mask_erode": args.mask_erode,
            "mask_dilate": args.mask_dilate,
        },
        "results": {
            "n_inpainted": n_images,
        },
    }
    
    config.save_manifest("05b_inpaint_holes", manifest)
    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05b_inpaint_holes_manifest.json'}[/green]")
    
    console.print("\n[bold green]✓ Module 05b Complete![/bold green]")
    console.print(f"[bold green]Next: Run 05c to optimize 3DGS to match targets[/bold green]\n")


if __name__ == "__main__":
    main()
