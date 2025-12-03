#!/usr/bin/env python3
"""
06_object_generation.py - Generate 3D Object from Text/Image

Goal: Generate a 3D mesh from image using TripoSR.

This module uses TripoSR (VAST AI Research) for fast single-image 3D reconstruction.
It converts images to 3D meshes directly.

Inputs:
  --image: Path or URL to input image (supports Google Drive links, direct URLs, or local paths)
  --output_dir: Directory to save generated mesh (default: from config)

Outputs (saved in output_dir/06_object_gen/):
  - mesh.obj: Generated 3D mesh
  - mesh.ply: Alternative mesh format
  - input_image.png: Processed input image
  - preview.png: Mesh preview rendering
  - manifest.json: Generation metadata

Dependencies:
  - TripoSR (VAST AI Research): https://github.com/VAST-AI-Research/TripoSR
  - rembg: Background removal
  - trimesh: Mesh processing
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

# Import project config
from project_utils.config import ProjectConfig

console = Console()

# Check for TripoSR
try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    TRIPOSR_AVAILABLE = True
except ImportError:
    console.print("[yellow]TripoSR not available![/yellow]")
    console.print("Please run: [cyan]./setup.sh[/cyan]")
    TRIPOSR_AVAILABLE = False

# Check for optional dependencies
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D object from image using TripoSR"
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
        "--output_dir",
        type=str,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Marching cubes resolution for mesh extraction (default: 256)",
    )
    parser.add_argument(
        "--remove_bg",
        action="store_true",
        default=True,
        help="Remove background from input image (default: True)",
    )
    parser.add_argument(
        "--foreground_ratio",
        type=float,
        default=0.85,
        help="Foreground ratio for image preprocessing (default: 0.85)",
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


def load_and_preprocess_image(image_path, remove_bg=True, foreground_ratio=0.85):
    """Load and preprocess image for TripoSR."""
    console.print(f"Loading image: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    if remove_bg and REMBG_AVAILABLE:
        console.print("Removing background...")
        image = remove_background(image, rembg.new_session())
        image = resize_foreground(image, foreground_ratio)
        
        # Fill transparent background with white
        image_np = np.array(image).astype(np.float32) / 255.0
        if image_np.shape[-1] == 4:  # Has alpha channel
            rgb = image_np[:, :, :3]
            alpha = image_np[:, :, 3:4]
            image_np = rgb * alpha + (1 - alpha)  # White background
        image = Image.fromarray((image_np * 255.0).astype(np.uint8))
    
    return image


def generate_mesh(
    image,
    device="cuda",
    mc_resolution=256,
):
    """Generate 3D mesh from image using TripoSR."""
    console.print("Loading TripoSR model...")
    
    # Load pretrained TripoSR model
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    
    console.print("Generating 3D representation...")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    
    console.print(f"Extracting mesh (resolution: {mc_resolution})...")
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
    mesh = meshes[0]
    
    return mesh


def save_outputs(mesh, image, output_dir, metadata):
    """Save generated mesh and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mesh in multiple formats
    mesh_obj = output_dir / "mesh.obj"
    mesh_ply = output_dir / "mesh.ply"
    
    console.print(f"Saving mesh to {mesh_obj}")
    mesh.export(str(mesh_obj))
    
    console.print(f"Saving mesh to {mesh_ply}")
    mesh.export(str(mesh_ply))
    
    # Save input image
    input_image_path = output_dir / "input_image.png"
    console.print(f"Saving input image to {input_image_path}")
    image.save(input_image_path)
    
    # Generate preview if trimesh is available
    if TRIMESH_AVAILABLE:
        try:
            console.print("Generating preview...")
            scene = trimesh.Scene(mesh)
            preview_data = scene.save_image(resolution=[512, 512])
            preview_path = output_dir / "preview.png"
            with open(preview_path, 'wb') as f:
                f.write(preview_data)
            console.print(f"Preview saved to {preview_path}")
        except Exception as e:
            console.print(f"[yellow]Could not generate preview: {e}[/yellow]")
    
    console.print(f"\n[green]✓[/green] Object generation complete!")
    console.print(f"Output directory: {output_dir}")
    console.print(f"  - Mesh: mesh.obj, mesh.ply")
    console.print(f"  - Input: input_image.png")
    if (output_dir / "preview.png").exists():
        console.print(f"  - Preview: preview.png")


def main():
    args = parse_args()
    
    # Check dependencies
    if not TRIPOSR_AVAILABLE:
        console.print("[red]Error: TripoSR not available![/red]")
        console.print("Please run: [cyan]./setup.sh[/cyan]")
        sys.exit(1)
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Determine output directory using unified structure
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use unified structure: object_generation module
        output_dir = config.get_path('object_generation')
    
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
        config_obj_gen = config.config.get('replacement', {}).get('object_generation', {})
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
    
    # Load and preprocess image
    console.print("\n" + "="*80)
    console.print("Module 06: Object Generation")
    console.print("="*80 + "\n")
    
    image = load_and_preprocess_image(
        image_path,
        remove_bg=args.remove_bg,
        foreground_ratio=args.foreground_ratio,
    )
    
    # Generate mesh
    mesh = generate_mesh(
        image,
        device=args.device,
        mc_resolution=args.resolution,
    )
    
    # Prepare metadata
    metadata = {
        "module": "06_object_generation",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "input_image": str(image_path),
            "image_source": args.image if args.image else config.config.get('replacement', {}).get('object_generation', {}).get('image_url'),
        },
        "parameters": {
            "resolution": args.resolution,
            "remove_background": args.remove_bg,
            "foreground_ratio": args.foreground_ratio,
            "device": args.device,
        },
        "outputs": {
            "mesh_obj": str(output_dir / "mesh.obj"),
            "mesh_ply": str(output_dir / "mesh.ply"),
            "input_image": str(output_dir / "input_image.png"),
            "preview": str(output_dir / "preview.png"),
        },
        "results": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "extents": mesh.extents.tolist(),
        }
    }
    
    # Save outputs
    save_outputs(mesh, image, output_dir, metadata)
    
    # Save manifest using unified system
    config.save_manifest("06_object_generation", metadata)


if __name__ == "__main__":
    main()
