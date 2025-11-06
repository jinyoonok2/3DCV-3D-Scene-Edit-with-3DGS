#!/usr/bin/env python3
"""
05b_inpaint_holes.py - Inpaint Holes with SDXL

Goal: Use SDXL Inpainting to fill holes in rendered views.

Approach:
  1. Load holed renders and masks from 05a
  2. For each view, run SDXL Inpainting to fill the hole
  3. Save inpainted target images

Inputs:
  --holed_dir: Directory with holed renders (from 05a)
  --output_dir: Output directory (default: sibling of holed_dir)
  --prompt: Inpainting prompt
  --negative_prompt: Negative prompt  
  --strength: SDXL denoising strength (0-1)
  --guidance_scale: CFG scale (default: 7.5)
  --num_steps: Number of denoising steps (default: 50)

Outputs (saved in output_dir/05b_inpainted/):
  - targets/train/*.png: Inpainted target images
  - manifest.json: Metadata
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
    parser = argparse.ArgumentParser(description="Inpaint holes with SDXL")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")
    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt (overrides config)")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (overrides config)")
    parser.add_argument("--strength", type=float, default=None, help="Denoising strength (overrides config)")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale (overrides config)")
    parser.add_argument("--num_steps", type=int, default=None, help="Denoising steps (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--mask_blur", type=int, default=8, help="Mask blur radius (default: 8)")
    parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")
    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate mask by N pixels (expand hole)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def load_sdxl_inpainting(device):
    """Load SDXL Inpainting pipeline"""
    from diffusers import StableDiffusionXLInpaintPipeline
    
    console.print("[cyan]Loading SDXL Inpainting model...[/cyan]")
    console.print("[dim](This may take a few minutes on first run - downloading ~6GB)[/dim]")
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    
    pipe.enable_attention_slicing()
    console.print("[green]✓ SDXL Inpainting loaded[/green]")
    return pipe


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


def inpaint_view(pipe, image_path, mask_path, prompt, negative_prompt, strength, guidance_scale, num_steps, seed, mask_blur=8, mask_erode=0, mask_dilate=0):
    """Inpaint a single view"""
    # Load images
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Process mask
    mask = process_mask(mask, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)
    
    # Run inpainting
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        mask_image=mask,
        strength=strength,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return result


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')
    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_inpainted')
    
    # IMPROVED DEFAULTS for clean removal (empty table)
    default_prompt = 'clean wooden table surface, empty table, natural wood texture'
    default_negative = 'objects, plants, pots, flowers, items, things, blurry, distorted, artifacts'
    
    prompt = args.prompt if args.prompt else config.config.get('inpainting', {}).get('prompt', default_prompt)
    negative_prompt = args.negative_prompt if args.negative_prompt else config.config.get('inpainting', {}).get('negative_prompt', default_negative)
    strength = args.strength if args.strength is not None else config.config.get('inpainting', {}).get('strength', 0.99)
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else config.config.get('inpainting', {}).get('guidance_scale', 7.5)
    num_steps = args.num_steps if args.num_steps is not None else config.config.get('inpainting', {}).get('num_steps', 50)
    seed = args.seed if args.seed is not None else config.config['dataset']['seed']
    
    device = torch.device(args.device)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05b - Inpaint Holes with SDXL[/bold cyan]")
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
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    console.print(f"[cyan]Strength:[/cyan] {strength}\n")
    
    # Get image list
    image_files = sorted(renders_dir.glob("*.png"))
    n_images = len(image_files)
    console.print(f"[cyan]Found {n_images} images to inpaint[/cyan]\n")
    
    # Load SDXL
    pipe = load_sdxl_inpainting(device)
    
    # Inpaint each view
    console.print("\n[yellow]Inpainting views...[/yellow]")
    
    for img_path in tqdm(image_files, desc="Inpainting"):
        idx = img_path.stem  # e.g., "00042"
        mask_path = masks_dir / f"{idx}.png"
        
        if not mask_path.exists():
            console.print(f"[yellow]Warning: No mask for {idx}, skipping[/yellow]")
            continue
        
        # Inpaint
        # Use SAME seed for all views for consistency (not seed + idx)
        result = inpaint_view(
            pipe=pipe,
            image_path=img_path,
            mask_path=mask_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,  # SAME seed for consistency across views
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
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "seed": seed,
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
