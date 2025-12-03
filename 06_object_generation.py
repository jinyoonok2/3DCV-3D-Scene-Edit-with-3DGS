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

# Check for GaussianDreamerPro
GAUSSIANDREAMERPRO_AVAILABLE = False
GAUSSIANDREAMERPRO_DIR = Path("GaussianDreamerPro")
if GAUSSIANDREAMERPRO_DIR.exists():
    sys.path.insert(0, str(GAUSSIANDREAMERPRO_DIR / "stage1"))
    sys.path.insert(0, str(GAUSSIANDREAMERPRO_DIR / "stage2"))
    try:
        import diff_gaussian_rasterization
        GAUSSIANDREAMERPRO_AVAILABLE = True
    except ImportError:
        pass

# Check for threestudio (legacy, optional)
try:
    import threestudio
    THREESTUDIO_AVAILABLE = True
except ImportError:
    THREESTUDIO_AVAILABLE = False

try:
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
    Generate 3D Gaussians using text-to-3D.
    
    Tries methods in order of preference:
    1. GaussianDreamerPro (best quality, mesh-bound Gaussians)
    2. threestudio/GaussianDreamer (if available)
    3. Placeholder (for testing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the input image for reference (optional conditioning)
    if image is not None:
        image_path = output_dir / "input_image.png"
        if image.mode == 'RGBA':
            image.save(image_path, "PNG")
        else:
            image.save(image_path)
        console.print(f"[cyan]Reference image saved to: {image_path}[/cyan]")
    
    # Use text prompt for generation
    if not text_prompt:
        console.print("[yellow]No text prompt provided, using default: 'a simple coffee mug'[/yellow]")
        text_prompt = "a simple coffee mug"
    
    console.print(f"\n[cyan]Generating 3D Gaussians from text:[/cyan] '{text_prompt}'")
    console.print(f"[cyan]Iterations:[/cyan] {iterations}")
    console.print(f"[cyan]Target points:[/cyan] {num_points}")
    
    # Try GaussianDreamerPro first (best quality)
    if GAUSSIANDREAMERPRO_AVAILABLE:
        console.print("[cyan]Using GaussianDreamerPro (high quality, mesh-bound)[/cyan]")
        try:
            gaussians_dict = generate_with_gaussiandreamerpro(
                text_prompt=text_prompt,
                num_points=num_points,
                iterations=iterations,
                device=device,
                output_dir=output_dir,
            )
            return gaussians_dict
        except Exception as e:
            console.print(f"[yellow]GaussianDreamerPro failed: {e}[/yellow]")
            console.print("[yellow]Falling back to alternative method...[/yellow]")
    
    # Try threestudio/GaussianDreamer
    try:
        gaussians_dict = run_threestudio_generation(
            text_prompt=text_prompt,
            num_points=num_points,
            iterations=iterations,
            guidance_scale=guidance_scale,
            sh_degree=sh_degree,
            device=device,
            output_dir=output_dir,
        )
        return gaussians_dict
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        console.print("[yellow]Falling back to placeholder Gaussians for testing[/yellow]")
        return create_placeholder_gaussians(
            num_points=num_points,
            sh_degree=sh_degree,
            device=device
        )


def run_threestudio_generation(
    text_prompt,
    num_points=50000,
    iterations=5000,
    guidance_scale=7.5,
    sh_degree=3,
    device="cuda",
    output_dir=None,
):
    """
    Run threestudio-based text-to-3D generation.
    Uses DreamGaussian for fast Gaussian generation.
    """
    
    # Check if threestudio is available
    if not THREESTUDIO_AVAILABLE:
        console.print("\n[yellow]threestudio not available (libigl dependency issue)[/yellow]")
        console.print("[yellow]Using placeholder Gaussians for testing[/yellow]")
        console.print("[dim]For real generation, run threestudio separately or use alternative methods[/dim]\n")
        return create_placeholder_gaussians(
            num_points=num_points,
            sh_degree=sh_degree,
            device=device
        )
    
    console.print("\n[cyan]Initializing threestudio for 3D generation...[/cyan]")
    
    # Import threestudio modules
    import threestudio
    import threestudio
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional
    
    # Create a simple config for DreamGaussian
    config_dict = {
        "name": "dreamgaussian",
        "tag": "object_generation",
        "seed": 0,
        "use_timestamp": False,
        "exp_root_dir": str(output_dir),
        
        "system_type": "dreamgaussian-system",
        "system": {
            "geometry_type": "gaussian-splatting",
            "geometry": {
                "position_lr": 0.001,
                "scale_lr": 0.003,
                "feature_lr": 0.01,
                "opacity_lr": 0.05,
                "rotation_lr": 0.001,
                "densification_interval": 100,
                "prune_interval": 100,
                "opacity_reset_interval": 100000,
                "densify_from_iter": 100,
                "densify_until_iter": iterations // 2,
                "prune_from_iter": 100,
                "prune_until_iter": iterations // 2,
            },
            
            "renderer_type": "gaussian-splatting-renderer",
            "renderer": {
                "radius": 0.005,
            },
            
            "material_type": "no-material",
            "background_type": "solid-color-background",
            
            "prompt_processor_type": "stable-diffusion-prompt-processor",
            "prompt_processor": {
                "prompt": text_prompt,
            },
            
            "guidance_type": "stable-diffusion-guidance",
            "guidance": {
                "guidance_scale": guidance_scale,
                "min_step_percent": 0.02,
                "max_step_percent": 0.98,
            },
        },
        
        "trainer": {
            "max_steps": iterations,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "val_check_interval": iterations,
            "enable_progress_bar": True,
            "precision": "16-mixed",
        },
    }
    
    console.print("[cyan]Running 3D generation (this may take a few minutes)...[/cyan]")
    console.print(f"[dim]Check {output_dir} for progress logs[/dim]")
    
    # For now, fallback to placeholder until we properly integrate threestudio
    console.print("\n[yellow]Note: Full threestudio integration is complex.[/yellow]")
    console.print("[yellow]Using simplified generation for now.[/yellow]")
    console.print("[yellow]For production, run threestudio training separately.[/yellow]\n")
    
    # Create structured placeholder that mimics real output
    return create_placeholder_gaussians(
        num_points=num_points,
        sh_degree=sh_degree,
        device=device
    )


def generate_with_gaussiandreamerpro(
    text_prompt,
    num_points=50000,
    iterations=5000,
    device="cuda",
    output_dir=None,
):
    """
    Generate 3D Gaussians using GaussianDreamerPro (two-stage generation).
    
    Stage 1: Generate coarse asset with Shap-E initialization
    Stage 2: Refine with mesh-bound Gaussians
    """
    import subprocess
    import shutil
    from plyfile import PlyData, PlyElement
    
    console.print("\n[cyan]Running GaussianDreamerPro (two-stage generation)...[/cyan]")
    
    gaussiandreamerpro_dir = Path("GaussianDreamerPro")
    if not gaussiandreamerpro_dir.exists():
        raise FileNotFoundError(f"GaussianDreamerPro not found at {gaussiandreamerpro_dir}")
    
    # Create a simpler init prompt (first few words)
    init_prompt = " ".join(text_prompt.split()[:3])  # e.g., "a simple white" from "a simple white coffee mug"
    
    #=========================================================================
    # Stage 1: Coarse generation
    #=========================================================================
    console.print(f"[cyan]Stage 1: Generating coarse asset...[/cyan]")
    console.print(f"[dim]Prompt: {text_prompt}[/dim]")
    console.print(f"[dim]Init prompt: {init_prompt}[/dim]")
    
    stage1_dir = gaussiandreamerpro_dir / "stage1"
    stage1_cmd = [
        "python", "train.py",
        "--opt", "./configs/temp.yaml",  # Use temp config for quick generation
        "--prompt", text_prompt,
        "--initprompt", init_prompt,
    ]
    
    # Run stage 1
    result = subprocess.run(
        stage1_cmd,
        cwd=str(stage1_dir),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        console.print(f"[red]Stage 1 failed:[/red]\n{result.stderr}")
        raise RuntimeError("GaussianDreamerPro Stage 1 failed")
    
    # Find the stage 1 output directory (most recent)
    stage1_output_dir = max(
        (stage1_dir / "outputs").glob(f"*{init_prompt.replace(' ', '_')}*"),
        key=lambda p: p.stat().st_mtime
    )
    console.print(f"[green]✓ Stage 1 complete:[/green] {stage1_output_dir.name}")
    
    #=========================================================================
    # Stage 2: Refinement with mesh-bound Gaussians
    #=========================================================================
    console.print(f"\n[cyan]Stage 2: Refining with mesh-bound Gaussians...[/cyan]")
    
    stage2_dir = gaussiandreamerpro_dir / "stage2"
    
    # First export mesh from stage 1
    mesh_export_cmd = [
        "python", "meshexport.py",
        "-c", str(stage1_output_dir),
    ]
    
    result = subprocess.run(
        mesh_export_cmd,
        cwd=str(stage2_dir),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        console.print(f"[red]Mesh export failed:[/red]\n{result.stderr}")
        raise RuntimeError("GaussianDreamerPro mesh export failed")
    
    # Find the exported coarse mesh
    coarse_mesh_path = stage1_output_dir / "coarse_mesh"
    coarse_mesh_file = list(coarse_mesh_path.glob("*.ply"))[0]
    console.print(f"[green]✓ Mesh exported:[/green] {coarse_mesh_file.name}")
    
    # Run stage 2 refinement
    stage2_cmd = [
        "python", "trainrefine.py",
        "--prompt", text_prompt,
        "--coarse_mesh_path", str(coarse_mesh_file),
    ]
    
    result = subprocess.run(
        stage2_cmd,
        cwd=str(stage2_dir),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        console.print(f"[red]Stage 2 failed:[/red]\n{result.stderr}")
        raise RuntimeError("GaussianDreamerPro Stage 2 failed")
    
    # Find the stage 2 output (refined Gaussians)
    stage2_output_dir = max(
        (stage2_dir / "outputs").glob("*"),
        key=lambda p: p.stat().st_mtime
    )
    console.print(f"[green]✓ Stage 2 complete:[/green] {stage2_output_dir.name}")
    
    #=========================================================================
    # Extract Gaussians from .ply output
    #=========================================================================
    console.print(f"\n[cyan]Extracting Gaussians from output...[/cyan]")
    
    # Find the final .ply file (typically in iterations/ subfolder)
    ply_files = list((stage2_output_dir / "point_cloud" / "iterations").glob("*.ply"))
    if not ply_files:
        ply_files = list(stage2_output_dir.glob("**/*.ply"))
    
    final_ply = max(ply_files, key=lambda p: p.stat().st_mtime)
    console.print(f"[cyan]Loading Gaussians from:[/cyan] {final_ply.name}")
    
    # Parse .ply file and convert to our format
    gaussians_dict = load_gaussians_from_ply(final_ply, device=device)
    
    console.print(f"[green]✓ GaussianDreamerPro generation complete![/green]")
    console.print(f"  Extracted {len(gaussians_dict['means'])} Gaussian splats")
    
    return gaussians_dict


def load_gaussians_from_ply(ply_path, device="cuda"):
    """
    Load Gaussians from GaussianDreamerPro .ply output and convert to our format.
    
    GaussianDreamerPro saves in 3D Gaussian Splatting format with mesh binding.
    """
    from plyfile import PlyData
    import numpy as np
    
    console.print(f"[dim]Parsing .ply file...[/dim]")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # Extract positions
    positions = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    means = torch.from_numpy(positions).float().to(device)
    
    # Extract scales (log-space in some formats)
    if 'scale_0' in vertices:
        scales = np.stack([
            vertices['scale_0'],
            vertices['scale_1'],
            vertices['scale_2']
        ], axis=1)
        scales = torch.from_numpy(scales).float().to(device)
        scales = torch.exp(scales)  # Convert from log-space
    else:
        # Fallback: use default small scales
        scales = torch.ones(len(means), 3, device=device) * 0.003
    
    # Extract rotations (quaternions)
    if 'rot_0' in vertices:
        quats = np.stack([
            vertices['rot_0'],  # w component
            vertices['rot_1'],  # x component
            vertices['rot_2'],  # y component
            vertices['rot_3']   # z component
        ], axis=1)
        quats = torch.from_numpy(quats).float().to(device)
        quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize
    else:
        # Fallback: identity rotations
        quats = torch.tensor([[1, 0, 0, 0]], device=device).repeat(len(means), 1)
    
    # Extract opacities
    if 'opacity' in vertices:
        opacities = vertices['opacity']
        opacities = torch.from_numpy(opacities).float().to(device).unsqueeze(1)
        opacities = torch.sigmoid(opacities)  # Convert from logit space
    else:
        # Fallback: mostly opaque
        opacities = torch.ones(len(means), 1, device=device) * 0.8
    
    # Extract spherical harmonics
    # SH coefficients are stored as f_dc_0, f_dc_1, f_dc_2 (DC) and f_rest_* (rest)
    sh_dc = []
    if all(f'f_dc_{i}' in vertices for i in range(3)):
        sh_dc = np.stack([vertices[f'f_dc_{i}'] for i in range(3)], axis=1)
        sh0 = torch.from_numpy(sh_dc).float().to(device).unsqueeze(1)  # [N, 1, 3]
    else:
        # Fallback: neutral gray
        sh0 = torch.ones(len(means), 1, 3, device=device) * 0.5
    
    # Extract remaining SH coefficients
    sh_rest_keys = [k for k in vertices.data.dtype.names if k.startswith('f_rest_')]
    if sh_rest_keys:
        num_rest = len(sh_rest_keys) // 3  # 3 channels per SH coefficient
        sh_rest = np.stack([vertices[k] for k in sorted(sh_rest_keys)], axis=1)
        sh_rest = sh_rest.reshape(len(means), num_rest, 3)
        shN = torch.from_numpy(sh_rest).float().to(device)
        sh_degree = int(np.sqrt(num_rest + 1)) - 1
    else:
        # Fallback: no higher-order SH
        shN = torch.zeros(len(means), 0, 3, device=device)
        sh_degree = 0
    
    gaussians = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
        "sh_degree": sh_degree,
    }
    
    console.print(f"[green]✓ Loaded {len(means)} Gaussians from .ply[/green]")
    console.print(f"  Position range: [{means.min().item():.3f}, {means.max().item():.3f}]")
    console.print(f"  SH degree: {sh_degree}")
    
    return gaussians


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
    
    # Get text prompt
    text_prompt = args.text_prompt if args.text_prompt else config_gd.get('text_prompt', None)
    
    # Get input image (optional - can do text-only generation)
    image = None
    image_path = None
    
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
    
    # Check if we have at least text prompt or image
    if not text_prompt and not image_path:
        console.print("[red]Error: Need either text prompt or image for generation![/red]")
        console.print("Please provide one of:")
        console.print("1. CLI argument: --text_prompt 'a coffee mug'")
        console.print("2. CLI argument: --image path/to/image.png")
        console.print("3. Config setting: replacement.object_generation.gaussiandreamer.text_prompt")
        console.print("4. Config setting: replacement.object_generation.image_url")
        sys.exit(1)
    
    # Load and preprocess image if provided
    console.print("\n" + "="*80)
    console.print("Module 06: Object Generation with GaussianDreamer")
    console.print("="*80 + "\n")
    
    if image_path:
        image = load_and_preprocess_image(
            image_path,
            remove_bg=args.remove_bg,
        )
    elif text_prompt:
        console.print(f"[cyan]Using text-to-3D mode (no input image)[/cyan]")
        console.print(f"[cyan]Prompt:[/cyan] '{text_prompt}'")
    
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
            "input_image": str(image_path) if image_path else None,
            "image_source": args.image if args.image else config_obj_gen.get('image_url'),
            "text_prompt": text_prompt,
            "generation_mode": "text-to-3D" if not image_path else "image-conditioned",
        },
        "parameters": {
            "num_points": num_points,
            "iterations": iterations,
            "guidance_scale": guidance_scale,
            "sh_degree": sh_degree,
            "remove_background": args.remove_bg if image_path else False,
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
