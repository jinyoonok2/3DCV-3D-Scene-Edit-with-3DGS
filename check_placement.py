#!/usr/bin/env python3
"""
check_placement.py - Diagnostic tool to check object placement

This utility helps you understand where objects will be placed and debug placement issues.
It shows:
- ROI center and bounds
- Object bounds and center
- Final placement after scale and offset
- Suggested xyz_offset to move object to specific locations

Usage:
    python check_placement.py --config configs/kitchen_config.yaml
    python check_placement.py --config configs/garden_config.yaml
    python check_placement.py --config configs/kitchen_config.yaml --suggest-center
"""

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Check object placement configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/garden_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--suggest-center",
        action="store_true",
        help="Suggest xyz_offset to place object at scene center [0, 0, 0]",
    )
    return parser.parse_args()


def load_checkpoint(path, name):
    """Load a checkpoint or tensor file."""
    try:
        path = Path(path)
        if not path.exists():
            console.print(f"[red]Error: {name} not found at {path}[/red]")
            return None
        
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        console.print(f"✓ {name} loaded")
        return ckpt
    except Exception as e:
        console.print(f"[red]Error loading {name}: {e}[/red]")
        return None


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Object Placement Diagnostic Tool")
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    placement_config = config.config.get('replacement', {}).get('placement', {})
    project_name = config.get("project", "name")
    
    # Get paths from config
    object_path = str(placement_config.get('object_gaussians', '')).replace('${project.name}', project_name)
    roi_path = str(placement_config.get('roi_file', '')).replace('${project.name}', project_name)
    
    # Get initial scene path
    original_path = config.get_path('initial_training') / 'ckpt_initial.pt'
    
    # Get placement parameters
    scale = placement_config.get('scale', 1.0)
    xyz_offset = placement_config.get('xyz_offset', [0.0, 0.0, 0.0])
    
    console.print(f"[cyan]Config:[/cyan] {args.config}")
    console.print(f"[cyan]Project:[/cyan] {project_name}\n")
    
    # Load files
    console.print("[bold]Loading files...[/bold]")
    roi_mask = load_checkpoint(roi_path, "ROI mask")
    original_ckpt = load_checkpoint(original_path, "Original scene")
    
    if roi_mask is None or original_ckpt is None:
        console.print("[red]Cannot proceed without ROI and scene data[/red]")
        sys.exit(1)
    
    # Load object if it exists
    object_ckpt = load_checkpoint(object_path, "Object gaussians")
    
    # Get scene positions
    original_gaussians = original_ckpt.get("splats", original_ckpt.get("gaussians", original_ckpt))
    scene_positions = original_gaussians["means"]
    
    # Convert ROI mask to boolean
    if roi_mask.dtype not in [torch.bool, torch.uint8]:
        roi_mask = roi_mask.bool()
    
    # Get ROI positions
    roi_positions = scene_positions[roi_mask]
    roi_center = roi_positions.mean(dim=0)
    roi_min = roi_positions.min(dim=0)[0]
    roi_max = roi_positions.max(dim=0)[0]
    roi_size = roi_max - roi_min
    
    # Scene bounds
    scene_min = scene_positions.min(dim=0)[0]
    scene_max = scene_positions.max(dim=0)[0]
    
    console.print("\n[bold]Scene Information:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property")
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Z", justify="right")
    
    table.add_row("Scene Min", f"{scene_min[0]:.3f}", f"{scene_min[1]:.3f}", f"{scene_min[2]:.3f}")
    table.add_row("Scene Max", f"{scene_max[0]:.3f}", f"{scene_max[1]:.3f}", f"{scene_max[2]:.3f}")
    table.add_row("Scene Center", "0.000", "0.000", "0.000", style="dim")
    console.print(table)
    
    console.print("\n[bold]ROI Information:[/bold]")
    roi_table = Table(show_header=True, header_style="bold yellow")
    roi_table.add_column("Property")
    roi_table.add_column("X", justify="right")
    roi_table.add_column("Y", justify="right")
    roi_table.add_column("Z", justify="right")
    
    roi_table.add_row("ROI Gaussians", str(roi_positions.shape[0]), "", "")
    roi_table.add_row("ROI Min", f"{roi_min[0]:.3f}", f"{roi_min[1]:.3f}", f"{roi_min[2]:.3f}")
    roi_table.add_row("ROI Max", f"{roi_max[0]:.3f}", f"{roi_max[1]:.3f}", f"{roi_max[2]:.3f}")
    roi_table.add_row("ROI Center", f"{roi_center[0]:.3f}", f"{roi_center[1]:.3f}", f"{roi_center[2]:.3f}", style="bold")
    roi_table.add_row("ROI Size", f"{roi_size[0]:.3f}", f"{roi_size[1]:.3f}", f"{roi_size[2]:.3f}")
    console.print(roi_table)
    
    # Object information if available
    if object_ckpt is not None:
        if object_path.endswith('.ply'):
            object_gaussians = object_ckpt
        else:
            object_gaussians = object_ckpt.get("gaussians", object_ckpt)
        
        obj_positions = object_gaussians["means"]
        obj_center = obj_positions.mean(dim=0)
        obj_min = obj_positions.min(dim=0)[0]
        obj_max = obj_positions.max(dim=0)[0]
        obj_size = obj_max - obj_min
        
        console.print("\n[bold]Object Information:[/bold]")
        obj_table = Table(show_header=True, header_style="bold green")
        obj_table.add_column("Property")
        obj_table.add_column("X", justify="right")
        obj_table.add_column("Y", justify="right")
        obj_table.add_column("Z", justify="right")
        
        obj_table.add_row("Object Gaussians", str(obj_positions.shape[0]), "", "")
        obj_table.add_row("Object Min", f"{obj_min[0]:.3f}", f"{obj_min[1]:.3f}", f"{obj_min[2]:.3f}")
        obj_table.add_row("Object Max", f"{obj_max[0]:.3f}", f"{obj_max[1]:.3f}", f"{obj_max[2]:.3f}")
        obj_table.add_row("Object Center", f"{obj_center[0]:.3f}", f"{obj_center[1]:.3f}", f"{obj_center[2]:.3f}")
        obj_table.add_row("Object Size", f"{obj_size[0]:.3f}", f"{obj_size[1]:.3f}", f"{obj_size[2]:.3f}")
        console.print(obj_table)
        
        # Calculate final placement
        centered_positions = obj_positions - obj_center
        scaled_positions = centered_positions * scale
        scaled_size = obj_size * scale
        
        target_center = roi_center.clone()
        target_center[0] += xyz_offset[0]
        target_center[1] += xyz_offset[1]
        target_center[2] += xyz_offset[2]
        
        final_positions = scaled_positions + target_center
        final_min = final_positions.min(dim=0)[0]
        final_max = final_positions.max(dim=0)[0]
        
        console.print("\n[bold]Placement Configuration:[/bold]")
        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Parameter")
        config_table.add_column("Value")
        
        config_table.add_row("Scale", f"{scale:.3f}")
        config_table.add_row("XYZ Offset", f"[{xyz_offset[0]:.3f}, {xyz_offset[1]:.3f}, {xyz_offset[2]:.3f}]")
        console.print(config_table)
        
        console.print("\n[bold]Final Placement After Transformation:[/bold]")
        final_table = Table(show_header=True, header_style="bold blue")
        final_table.add_column("Property")
        final_table.add_column("X", justify="right")
        final_table.add_column("Y", justify="right")
        final_table.add_column("Z", justify="right")
        
        final_table.add_row("Final Center", f"{target_center[0]:.3f}", f"{target_center[1]:.3f}", f"{target_center[2]:.3f}", style="bold")
        final_table.add_row("Final Min", f"{final_min[0]:.3f}", f"{final_min[1]:.3f}", f"{final_min[2]:.3f}")
        final_table.add_row("Final Max", f"{final_max[0]:.3f}", f"{final_max[1]:.3f}", f"{final_max[2]:.3f}")
        final_table.add_row("Scaled Size", f"{scaled_size[0]:.3f}", f"{scaled_size[1]:.3f}", f"{scaled_size[2]:.3f}")
        console.print(final_table)
        
        # Suggestions
        if args.suggest_center:
            console.print("\n[bold yellow]Suggestions:[/bold yellow]")
            # To place at scene center [0, 0, 0]
            offset_to_center = -roi_center
            console.print(f"To place object at scene center [0, 0, 0]:")
            console.print(f"  xyz_offset: [{offset_to_center[0]:.3f}, {offset_to_center[1]:.3f}, {offset_to_center[2]:.3f}]")
            
            # To place at specific Z heights
            offset_to_ground = -roi_center.clone()
            offset_to_ground[2] = scene_min[2] - roi_center[2] + scaled_size[2] / 2
            console.print(f"\nTo place object on ground (Z = {scene_min[2]:.3f}):")
            console.print(f"  xyz_offset: [{offset_to_ground[0]:.3f}, {offset_to_ground[1]:.3f}, {offset_to_ground[2]:.3f}]")
    
    console.print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
