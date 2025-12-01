#!/usr/bin/env python3
"""
06_object_generation.py - Generate 3D Object from Text/Image

Goal: Generate a 3D mesh from text prompt or image using TripoSR.

This module uses TripoSR (VAST AI Research) for fast single-image 3D reconstruction.
It supports text-to-image (via diffusion models) → image-to-3D workflows.

Inputs:
  --prompt: Text description of object to generate (e.g., "red flower in pot")
  --image: Path to input image (alternative to text prompt)
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
import sys
from datetime import datetime
from pathlib import Path

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
        description="Generate 3D object from text prompt or image using TripoSR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for object generation (e.g., 'red flower in pot')",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (alternative to text prompt)",
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
    
    # Save metadata
    manifest_path = output_dir / "manifest.json"
    console.print(f"Saving metadata to {manifest_path}")
    with open(manifest_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use same structure as other modules: outputs/{project.name}/06_object_gen/
        project_name = config.get("project", {}).get("name", "garden")
        output_dir = Path(f"outputs/{project_name}/06_object_gen")
    
    # Get input image
    if args.image:
        image_path = args.image
    elif args.prompt:
        console.print("[red]Error: Text-to-image generation not yet implemented![/red]")
        console.print("Please provide an image with --image")
        console.print("\nAlternatively, you can:")
        console.print("1. Generate an image using Stable Diffusion")
        console.print("2. Find/download an image of the object")
        console.print("3. Pass it to this script with --image path/to/image.png")
        sys.exit(1)
    else:
        console.print("[red]Error: Either --prompt or --image must be provided![/red]")
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
        "parameters": {
            "prompt": args.prompt,
            "input_image": str(image_path) if args.image else None,
            "resolution": args.resolution,
            "remove_background": args.remove_bg,
            "foreground_ratio": args.foreground_ratio,
            "device": args.device,
        },
        "outputs": {
            "mesh": "mesh.obj",
            "mesh_ply": "mesh.ply",
            "input_image": "input_image.png",
            "preview": "preview.png",
        },
        "mesh_stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "extents": mesh.extents.tolist(),
        }
    }
    
    # Save outputs
    save_outputs(mesh, image, output_dir, metadata)


if __name__ == "__main__":
    main()
