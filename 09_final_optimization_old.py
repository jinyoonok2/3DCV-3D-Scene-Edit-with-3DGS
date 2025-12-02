#!/usr/bin/env python3
"""
09_final_optimization.py - Final Scene Optimization with New Object

This module wraps 05c_optimize_to_targets.py to optimize the merged scene from Module 08.
It automatically sets up the correct paths and calls 05c with appropriate arguments.

Usage:
  python 09_final_optimization.py --init_ckpt <merged_scene.pt> --data_root <dataset>
"""

import argparse
import subprocess
import sys
from pathlib import Path
from rich.console import Console

from project_utils.config import ProjectConfig

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final optimization of merged scene (wraps Module 05c)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--init_ckpt",
        type=str,
        required=True,
        help="Merged checkpoint from Module 08",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (default: auto from config)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Optimization iterations (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    console.print("\n" + "="*80)
    console.print("Module 09: Final Scene Optimization")
    console.print("="*80 + "\n")
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_name = config.get("project", "name") or "project"
        output_dir = f"outputs/{project_name}/09_final_optimized"
    
    console.print(f"Input checkpoint: {args.init_ckpt}")
    console.print(f"Dataset: {args.data_root}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Iterations: {args.iters}\n")
    
    # Since Module 08 merged scene with holed scene (no inpainted targets),
    # we need to optimize against original training views
    # We'll create dummy targets directory or use original renders
    
    console.print("[cyan]Calling Module 05c for optimization...[/cyan]\n")
    
    # Build command for 05c
    # For Module 09 (object placement), we optimize against original dataset images
    # not inpainted targets. We need to create a symbolic link or copy dataset images
    # to a targets directory for 05c to use.
    
    from pathlib import Path
    import shutil
    
    # Create temporary targets directory with dataset images
    temp_targets = Path(output_dir) / "temp_targets"
    temp_targets.mkdir(parents=True, exist_ok=True)
    
    # Copy or link dataset images to targets directory
    dataset_images = Path(args.data_root) / "images_4"  # Assuming factor=4
    if not dataset_images.exists():
        dataset_images = Path(args.data_root) / "images"
    
    if dataset_images.exists():
        console.print(f"[cyan]Using dataset images from: {dataset_images}[/cyan]")
        # Create symlink instead of copying to save space
        for img in dataset_images.glob("*.JPG") or dataset_images.glob("*.jpg") or dataset_images.glob("*.png"):
            target_link = temp_targets / img.name
            if not target_link.exists():
                target_link.symlink_to(img.absolute())
    else:
        console.print(f"[red]Error: Could not find dataset images in {args.data_root}[/red]")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        "05c_optimize_to_targets.py",
        "--config", args.config,
        "--ckpt", args.init_ckpt,
        "--targets_dir", str(temp_targets),
        "--data_root", args.data_root,
        "--output_dir", output_dir,
        "--iters", str(args.iters),
        "--device", args.device,
    ]
    
    # For Module 09, we don't have inpainted targets - we optimize against original views
    # Create a targets_dir pointing to original rendered views or handle in 05c
    # For now, pass the init_ckpt path parent as a hint
    init_path = Path(args.init_ckpt)
    
    # Add note about targets
    console.print("[yellow]Note: Optimizing merged scene against original training views[/yellow]")
    console.print("[yellow]No inpainted targets needed - using original dataset images[/yellow]\n")
    
    # Run 05c
    try:
        result = subprocess.run(cmd, check=True)
        console.print(f"\n[green]✓[/green] Module 09 complete!")
        console.print(f"Optimized checkpoint saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]✗[/red] Optimization failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]![/yellow] Optimization interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
