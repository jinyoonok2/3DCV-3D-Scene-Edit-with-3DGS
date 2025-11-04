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

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint holes with SDXL")
    parser.add_argument("--holed_dir", type=str, required=True, help="Directory from 05a")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--prompt", type=str, default="natural outdoor scene, grass, plants", help="Prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, artifacts", help="Negative prompt")
    parser.add_argument("--strength", type=float, default=0.99, help="Denoising strength")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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


def inpaint_view(pipe, image_path, mask_path, prompt, negative_prompt, strength, guidance_scale, num_steps, seed):
    """Inpaint a single view"""
    # Load images
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
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
    device = torch.device(args.device)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]05b - Inpaint Holes with SDXL[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    holed_dir = Path(args.holed_dir)
    renders_dir = holed_dir / "renders" / "train"
    masks_dir = holed_dir / "masks" / "train"
    
    if not renders_dir.exists() or not masks_dir.exists():
        console.print(f"[red]ERROR: Could not find renders or masks in {holed_dir}[/red]")
        console.print(f"[red]Make sure to run 05a first![/red]")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = holed_dir.parent / "05b_inpainted"
    else:
        output_dir = Path(args.output_dir)
    
    targets_dir = output_dir / "targets" / "train"
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Input directory:[/cyan] {holed_dir}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Prompt:[/cyan] {args.prompt}")
    console.print(f"[cyan]Strength:[/cyan] {args.strength}\n")
    
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
        result = inpaint_view(
            pipe=pipe,
            image_path=img_path,
            mask_path=mask_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            seed=args.seed + int(idx),  # Different seed per view
        )
        
        # Save
        result.save(targets_dir / f"{idx}.png")
    
    console.print(f"[green]✓ Saved inpainted targets to {targets_dir}[/green]")
    
    # Save manifest
    manifest = {
        "module": "05b_inpaint_holes",
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "holed_dir": str(holed_dir),
        },
        "parameters": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "strength": args.strength,
            "guidance_scale": args.guidance_scale,
            "num_steps": args.num_steps,
            "seed": args.seed,
        },
        "results": {
            "n_inpainted": n_images,
        },
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    console.print("\n[bold green]✓ Module 05b Complete![/bold green]")
    console.print(f"[bold green]Next: Run 05c to optimize 3DGS to match targets[/bold green]\n")


if __name__ == "__main__":
    main()
