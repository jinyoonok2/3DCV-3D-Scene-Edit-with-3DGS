#!/usr/bin/env python3
"""
07_mesh_to_gaussians.py - Convert Mesh to Gaussian Splats

Goal: Convert the generated mesh (from Module 06) into 3D Gaussian splats.

This module samples points on the mesh surface and initializes Gaussian parameters
(position, scale, rotation, color, opacity, spherical harmonics) for each point.

Inputs:
  --mesh: Path to mesh file (.ply or .obj from Module 06)
  --num_points: Number of Gaussian splats to generate (default: 10000)
  --output: Path to save Gaussian splats (.pt file)

Outputs:
  - gaussians.pt: Checkpoint file with Gaussian parameters
  - manifest.json: Conversion metadata

Dependencies:
  - trimesh: Mesh loading and surface sampling
  - torch: Gaussian parameter storage
  - numpy: Point processing
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
from rich.console import Console
from scipy.spatial.transform import Rotation as R

# Import project config
from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mesh to Gaussian splats"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to input mesh (.ply or .obj)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=10000,
        help="Number of Gaussian splats to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for Gaussian splats (.pt file)",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Spherical harmonics degree (default: 3)",
    )
    parser.add_argument(
        "--initial_opacity",
        type=float,
        default=0.1,
        help="Initial opacity value (default: 0.1)",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.01,
        help="Scale factor for Gaussian sizes (default: 0.01)",
    )
    
    return parser.parse_args()


def sample_mesh_surface(mesh, num_points):
    """Sample points uniformly on mesh surface."""
    console.print(f"Sampling {num_points} points on mesh surface...")
    
    # Sample points and get face normals
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]
    
    # Get vertex colors if available
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # Get colors from the mesh vertices
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        
        # For each sampled point, get color from nearest vertex
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh.vertices)
        _, nearest_vertices = tree.query(points)
        colors = vertex_colors[nearest_vertices]
    else:
        # Default gray color
        colors = np.ones((num_points, 3), dtype=np.float32) * 0.5
    
    console.print(f"✓ Sampled {len(points)} points")
    return points, normals, colors


def compute_rotations_from_normals(normals):
    """Compute rotation quaternions that align with surface normals."""
    console.print("Computing rotations from surface normals...")
    
    num_points = len(normals)
    rotations = np.zeros((num_points, 4), dtype=np.float32)  # quaternions (w, x, y, z)
    
    # Default up vector
    up = np.array([0, 0, 1], dtype=np.float32)
    
    for i, normal in enumerate(normals):
        # Normalize normal
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Compute rotation that aligns up vector with normal
        v = np.cross(up, normal)
        s = np.linalg.norm(v)
        c = np.dot(up, normal)
        
        if s < 1e-6:  # Normal is parallel to up
            if c > 0:  # Same direction
                rotations[i] = [1, 0, 0, 0]  # Identity quaternion
            else:  # Opposite direction
                rotations[i] = [0, 1, 0, 0]  # 180 degree rotation
        else:
            # Rodrigues' rotation formula
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            
            # Convert to quaternion
            rot = R.from_matrix(R_matrix)
            q = rot.as_quat()  # Returns [x, y, z, w]
            rotations[i] = [q[3], q[0], q[1], q[2]]  # Convert to [w, x, y, z]
    
    console.print(f"✓ Computed rotations for {num_points} points")
    return rotations


def initialize_gaussians(points, normals, colors, args):
    """Initialize Gaussian splat parameters."""
    console.print("Initializing Gaussian parameters...")
    
    num_points = len(points)
    
    # Positions (xyz)
    positions = torch.from_numpy(points).float()
    
    # Rotations (quaternions)
    rotations = compute_rotations_from_normals(normals)
    rotations = torch.from_numpy(rotations).float()
    
    # Scales (log scale)
    # Initialize with small uniform scale
    scales = torch.ones(num_points, 3).float() * np.log(args.scale_factor)
    
    # Colors (RGB from mesh)
    rgb_colors = torch.from_numpy(colors).float()
    
    # Spherical harmonics (SH coefficients)
    # SH degree 0 (DC component) initialized from RGB colors
    # Higher degrees initialized to zero
    num_sh_coeffs = (args.sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(num_points, num_sh_coeffs, 3).float()
    
    # Set DC component (first SH coefficient) from RGB
    # Convert RGB to SH DC component: SH_DC = (RGB - 0.5) / 0.28209479177387814
    C0 = 0.28209479177387814
    sh_coeffs[:, 0, :] = (rgb_colors - 0.5) / C0
    
    # Opacity (inverse sigmoid space)
    # sigmoid(x) = opacity, so x = log(opacity / (1 - opacity))
    opacity_value = args.initial_opacity
    opacity_logit = np.log(opacity_value / (1 - opacity_value + 1e-8))
    opacities = torch.ones(num_points, 1).float() * opacity_logit
    
    console.print(f"✓ Initialized Gaussian parameters:")
    console.print(f"  Positions: {positions.shape}")
    console.print(f"  Rotations: {rotations.shape}")
    console.print(f"  Scales: {scales.shape}")
    console.print(f"  SH coeffs: {sh_coeffs.shape}")
    console.print(f"  Opacities: {opacities.shape}")
    
    return {
        "means": positions,
        "quats": rotations,
        "scales": scales,
        "sh0": sh_coeffs[:, 0, :],  # DC component
        "shN": sh_coeffs[:, 1:, :] if num_sh_coeffs > 1 else torch.zeros(num_points, 0, 3),  # Higher order
        "opacities": opacities,
    }


def save_gaussians(gaussians, mesh_path, output_path, metadata):
    """Save Gaussian parameters to checkpoint file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Saving Gaussians to {output_path}")
    
    # Save checkpoint
    checkpoint = {
        "gaussians": gaussians,
        "metadata": metadata,
    }
    torch.save(checkpoint, output_path)
    
    # Save metadata JSON
    manifest_path = output_path.parent / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"\n[green]✓[/green] Mesh to Gaussians conversion complete!")
    console.print(f"Output: {output_path}")
    console.print(f"  - Gaussians: {len(gaussians['means'])} splats")
    console.print(f"  - Manifest: {manifest_path}")


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 07: Mesh to Gaussians Conversion")
    console.print("="*80 + "\n")
    
    # Load mesh
    console.print(f"Loading mesh: {args.mesh}")
    try:
        mesh = trimesh.load(args.mesh)
        console.print(f"✓ Mesh loaded:")
        console.print(f"  Vertices: {len(mesh.vertices)}")
        console.print(f"  Faces: {len(mesh.faces)}")
        console.print(f"  Bounds: {mesh.bounds}")
    except Exception as e:
        console.print(f"[red]Error loading mesh: {e}[/red]")
        sys.exit(1)
    
    # Sample mesh surface
    points, normals, colors = sample_mesh_surface(mesh, args.num_points)
    
    # Initialize Gaussian parameters
    gaussians = initialize_gaussians(points, normals, colors, args)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save in 06_object_gen directory
        config = ProjectConfig(args.config)
        project_name = config.get("project", {}).get("name", "garden")
        output_path = Path(f"outputs/{project_name}/06_object_gen/gaussians.pt")
    
    # Prepare metadata
    metadata = {
        "module": "07_mesh_to_gaussians",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "parameters": {
            "mesh_file": str(args.mesh),
            "num_points": args.num_points,
            "sh_degree": args.sh_degree,
            "initial_opacity": args.initial_opacity,
            "scale_factor": args.scale_factor,
        },
        "mesh_stats": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
        },
        "gaussian_stats": {
            "num_gaussians": len(gaussians['means']),
            "position_bounds": [
                gaussians['means'].min(dim=0)[0].tolist(),
                gaussians['means'].max(dim=0)[0].tolist(),
            ],
        },
    }
    
    # Save Gaussians
    save_gaussians(gaussians, args.mesh, output_path, metadata)


if __name__ == "__main__":
    main()
