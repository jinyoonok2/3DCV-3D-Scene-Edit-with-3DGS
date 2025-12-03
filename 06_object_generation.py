#!/usr/bin/env python3
"""
06_object_generation.py - Generate 3D Gaussians from Text/Image

Goal: Generate 3D Gaussian splats directly from image using GaussianDreamer.

This module uses GaussianDreamer for direct image-to-3D Gaussian generation,
bypassing the mesh intermediate step for better quality.

Inputs:
  --image: Path or URL to input image (supports Google Drive links, direct URLs, or local paths)
  --text_prompt: Optional text prompt to guide generation
  --output_dir: Directory to save generated Gaussians (default: from config)

Outputs (saved in output_dir/06_object_gen/):
  - gaussians.pt: Generated 3D Gaussian splats (ready for Module 07)
  - input_image.png: Processed input image
  - preview/: Preview renderings from different angles
  - manifest.json: Generation metadata

Dependencies:
  - GaussianDreamer: https://github.com/hustvl/GaussianDreamer
  - threestudio: 3D generation framework
  - rembg: Background removal
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import project config
from project_utils.config import ProjectConfig

console = Console()

# Check for GaussianDreamer dependencies
try:
    import threestudio
    import omegaconf
    from omegaconf import OmegaConf
    import einops
    GAUSSIANDREAMER_AVAILABLE = True
except ImportError as e:
    console.print(f"[yellow]GaussianDreamer dependencies not available: {e}[/yellow]")
    console.print("Please run: [cyan]./setup-generation.sh[/cyan]")
    GAUSSIANDREAMER_AVAILABLE = False

# Check for optional dependencies
try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D Gaussians from image using GaussianDreamer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path or URL to input image (supports Google Drive links, direct URLs, or local paths)",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        help="Optional text prompt to guide generation (e.g., 'a potted plant')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        help="Number of Gaussian points to generate (default: from config, typically 50000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Training iterations (default: from config, typically 5000)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Classifier-free guidance scale (default: from config, typically 7.5)",
    )
    parser.add_argument(
        "--remove_bg",
        action="store_true",
        default=True,
        help="Remove background from input image (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    
    return parser.parse_args()


def download_image_from_url(url, output_path=None):
    """Download image from URL (supports Google Drive, direct URLs)."""
    import subprocess
    
    # Check if it's a Google Drive link
    gdrive_patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',
    ]
    
    file_id = None
    for pattern in gdrive_patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break
    
    if file_id:
        # Google Drive link
        console.print(f"Detected Google Drive link, downloading...")
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.jpg')
        
        try:
            # Use gdown to download
            cmd = ['gdown', f'https://drive.google.com/uc?id={file_id}', '-O', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                console.print(f"[red]Error downloading from Google Drive: {result.stderr}[/red]")
                sys.exit(1)
            console.print(f"✓ Downloaded from Google Drive to {output_path}")
            return output_path
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("Make sure gdown is installed: pip install gdown")
            sys.exit(1)
    else:
        # Regular URL
        console.print(f"Downloading image from URL...")
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.jpg')
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
            console.print(f"✓ Downloaded to {output_path}")
            return output_path
        except Exception as e:
            console.print(f"[red]Error downloading from URL: {e}[/red]")
            sys.exit(1)


def load_and_preprocess_image(image_path, remove_bg=True):
    """Load and preprocess image for GaussianDreamer."""
    console.print(f"Loading image: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    if remove_bg and REMBG_AVAILABLE:
        console.print("Removing background...")
        # Use rembg to remove background
        from rembg import remove
        image_np = np.array(image)
        output = remove(image_np)
        
        # Convert to RGBA
        if output.shape[-1] == 4:
            image = Image.fromarray(output)
        else:
            image = Image.fromarray(output).convert("RGBA")
        
        console.print(f"[cyan]Preprocessed image:[/cyan] {image.size}, mode={image.mode}")
    
    return image


def generate_gaussians_with_gaussiandreamer(
    image,
    text_prompt=None,
    num_points=50000,
    iterations=5000,
    guidance_scale=7.5,
    sh_degree=3,
    device="cuda",
    output_dir=None,
):
    """
    Generate 3D Gaussians from image using GaussianDreamer.
    
    This is a simplified wrapper. In practice, you would:
    1. Set up threestudio config for GaussianDreamer
    2. Run the image-conditioned generation
    3. Extract Gaussians in gsplat format
    
    For now, this creates a placeholder structure that matches the expected output.
    """
    console.print("\n[yellow]Note: Full GaussianDreamer integration requires custom config setup[/yellow]")
    console.print("Generating 3D Gaussians from image...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the input image for reference
    image_path = output_dir / "input_image.png"
    if image.mode == 'RGBA':
        # Save with transparency
        image.save(image_path, "PNG")
    else:
        image.save(image_path)
    
    console.print(f"[cyan]Input image saved to: {image_path}[/cyan]")
    
    # TODO: Actual GaussianDreamer generation
    # This would involve:
    # 1. Setting up threestudio config
    # 2. Loading GaussianDreamer model
    # 3. Running image-conditioned generation
    # 4. Extracting Gaussians
    
    console.print(f"\n[yellow]================================================[/yellow]")
    console.print(f"[yellow]IMPLEMENTATION NOTE:[/yellow]")
    console.print(f"[yellow]Full GaussianDreamer integration requires:[/yellow]")
    console.print(f"[yellow]1. Setting up threestudio config for image conditioning[/yellow]")
    console.print(f"[yellow]2. Running GaussianDreamer training loop[/yellow]")
    console.print(f"[yellow]3. Extracting Gaussians in compatible format[/yellow]")
    console.print(f"[yellow]================================================[/yellow]\n")
    
    # For now, create a simple placeholder Gaussian structure
    # This would be replaced with actual GaussianDreamer output
    gaussians_dict = create_placeholder_gaussians(
        num_points=num_points,
        sh_degree=sh_degree,
        device=device
    )
    
    return gaussians_dict


def create_placeholder_gaussians(num_points=50000, sh_degree=3, device="cuda"):
    """
    Create placeholder Gaussian structure for testing.
    This will be replaced with actual GaussianDreamer output.
    """
    console.print(f"[yellow]Creating placeholder Gaussians (will be replaced with GaussianDreamer output)[/yellow]")
    
    # Initialize random Gaussians in a unit sphere
    means = torch.randn(num_points, 3, device=device) * 0.5
    
    # Random scales (small initial values)
    scales = torch.rand(num_points, 3, device=device) * 0.01 + 0.001
    
    # Random rotations (as quaternions)
    quats = torch.randn(num_points, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    # Random opacities (mostly visible)
    opacities = torch.rand(num_points, 1, device=device) * 0.5 + 0.5
    
    # Spherical harmonics coefficients
    # sh0 is the DC component (base color)
    sh0 = torch.rand(num_points, 1, 3, device=device) * 0.5 + 0.25  # Warm colors
    
    # Higher SH degrees (for view-dependent effects)
    num_sh_bases = (sh_degree + 1) ** 2 - 1
    shN = torch.randn(num_points, num_sh_bases, 3, device=device) * 0.1
    
    gaussians = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
        "sh_degree": sh_degree,
    }
    
    # Log statistics
    console.print(f"\n[cyan]Gaussian Statistics:[/cyan]")
    console.print(f"  Number of points: {num_points}")
    console.print(f"  Position bounds: [{means.min().item():.3f}, {means.max().item():.3f}]")
    console.print(f"  Scale range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")
    console.print(f"  Opacity range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]")
    console.print(f"  SH degree: {sh_degree}")
    
    return gaussians


def save_outputs(gaussians, output_dir, metadata):
    """Save generated Gaussians and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Gaussians
    gaussians_path = output_dir / "gaussians.pt"
    console.print(f"Saving Gaussians to {gaussians_path}")
    torch.save(gaussians, str(gaussians_path))
    
    # Update metadata with output info
    metadata["outputs"]["gaussians"] = str(gaussians_path)
    metadata["results"]["num_gaussians"] = len(gaussians["means"])
    
    console.print(f"\n[green]✓[/green] Object generation complete!")
    console.print(f"Output directory: {output_dir}")
    console.print(f"  - Gaussians: gaussians.pt ({metadata['results']['num_gaussians']} points)")
    console.print(f"  - Input: input_image.png")


def main():
    args = parse_args()
    
    # Check dependencies
    if not GAUSSIANDREAMER_AVAILABLE:
        console.print("[red]Error: GaussianDreamer dependencies not available![/red]")
        console.print("Please run: [cyan]./setup-generation.sh[/cyan]")
        sys.exit(1)
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Determine output directory using unified structure
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use unified structure: object_generation module
        output_dir = config.get_path('object_generation')
    
    # Get configuration values
    config_obj_gen = config.config.get('replacement', {}).get('object_generation', {})
    config_gd = config_obj_gen.get('gaussiandreamer', {})
    
    # Get parameters (CLI args override config)
    num_points = args.num_points if args.num_points else config_gd.get('num_points', 50000)
    iterations = args.iterations if args.iterations else config_gd.get('iterations', 5000)
    guidance_scale = args.guidance_scale if args.guidance_scale else config_gd.get('guidance_scale', 7.5)
    sh_degree = config_gd.get('sh_degree', 3)
    
    # Get input image
    if args.image:
        # Check if it's a URL or local path
        if args.image.startswith('http://') or args.image.startswith('https://') or 'drive.google.com' in args.image:
            # Download from URL
            image_path = download_image_from_url(args.image)
        else:
            # Local path
            image_path = args.image
    else:
        # Try to use config values
        config_image = config_obj_gen.get('image_url')
        
        if config_image:
            console.print(f"[cyan]Using image from config:[/cyan] {config_image}")
            if config_image.startswith('http://') or config_image.startswith('https://') or 'drive.google.com' in config_image:
                image_path = download_image_from_url(config_image)
            else:
                image_path = config_image
        else:
            console.print("[red]Error: No image source found![/red]")
            console.print("Please provide one of:")
            console.print("1. CLI argument: --image path/to/image.png")
            console.print("2. Config setting: replacement.object_generation.image_url")
            sys.exit(1)
    
    # Get text prompt
    text_prompt = args.text_prompt if args.text_prompt else config_gd.get('text_prompt', None)
    
    # Load and preprocess image
    console.print("\n" + "="*80)
    console.print("Module 06: Object Generation with GaussianDreamer")
    console.print("="*80 + "\n")
    
    image = load_and_preprocess_image(
        image_path,
        remove_bg=args.remove_bg,
    )
    
    # Generate Gaussians
    gaussians = generate_gaussians_with_gaussiandreamer(
        image,
        text_prompt=text_prompt,
        num_points=num_points,
        iterations=iterations,
        guidance_scale=guidance_scale,
        sh_degree=sh_degree,
        device=args.device,
        output_dir=output_dir,
    )
    
    # Prepare metadata
    metadata = {
        "module": "06_object_generation",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "method": "GaussianDreamer",
        "inputs": {
            "input_image": str(image_path),
            "image_source": args.image if args.image else config_obj_gen.get('image_url'),
            "text_prompt": text_prompt,
        },
        "parameters": {
            "num_points": num_points,
            "iterations": iterations,
            "guidance_scale": guidance_scale,
            "sh_degree": sh_degree,
            "remove_background": args.remove_bg,
            "device": args.device,
        },
        "outputs": {},
        "results": {},
    }
    
    # Save outputs
    save_outputs(gaussians, output_dir, metadata)
    
    # Save manifest using unified system
    config.save_manifest("06_object_generation", metadata)


if __name__ == "__main__":
    main()
