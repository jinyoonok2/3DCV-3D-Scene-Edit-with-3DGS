#!/usr/bin/env python3
"""
make_comparison_grid.py - Create before/after comparison grids for scene editing results.

This script creates side-by-side comparison images showing:
- Original renders (before removal)
- Final optimized renders (after removal)

Useful for creating figures for reports and presentations.
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from tqdm import tqdm

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Create before/after comparison grid for scene editing.")
    parser.add_argument(
        "--before_dir",
        type=Path,
        required=True,
        help="Directory with original renders (e.g., outputs/.../02_renders/train)",
    )
    parser.add_argument(
        "--after_dir",
        type=Path,
        required=True,
        help="Directory with final renders (e.g., outputs/.../05c_optimized/renders/train)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("comparison_grids"),
        help="Directory to save output grids (default: comparison_grids/)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of image pairs per row (default: 4)",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=16,
        help="Maximum number of image pairs to include (default: 16)",
    )
    parser.add_argument(
        "--cell_width",
        type=int,
        default=256,
        help="Width of each before/after cell in pixels (default: 256)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=10,
        help="Gap between before/after images in pixels (default: 10)",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Add 'Before' and 'After' labels to images",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort images alphabetically before creating the grid",
    )
    return parser.parse_args()


def add_label(img, text, position="top"):
    """Add a text label to an image."""
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    x = (img.width - text_width) // 2
    if position == "top":
        y = 10
    else:  # bottom
        y = img.height - text_height - 10
    
    # Draw text with outline for visibility
    outline_color = "black"
    text_color = "white"
    
    # Draw outline
    for adj_x in [-1, 0, 1]:
        for adj_y in [-1, 0, 1]:
            draw.text((x + adj_x, y + adj_y), text, font=font, fill=outline_color)
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)
    
    return img


def main():
    args = parse_args()
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Creating Before/After Comparison Grid[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    console.print(f"  [cyan]Before Dir:[/cyan] {args.before_dir}")
    console.print(f"  [cyan]After Dir:[/cyan] {args.after_dir}")
    console.print(f"  [cyan]Output Dir:[/cyan] {args.output_dir}")
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images in before directory
    before_paths = list(args.before_dir.glob("*.png")) + \
                   list(args.before_dir.glob("*.jpg")) + \
                   list(args.before_dir.glob("*.jpeg"))
                   
    if args.sort:
        before_paths.sort()
        
    if not before_paths:
        console.print(f"[red]Error: No images found in {args.before_dir}[/red]")
        return

    # Apply limit
    if args.limit > 0:
        before_paths = before_paths[:args.limit]
        
    num_images = len(before_paths)
    console.print(f"\n[cyan]Found {num_images} images to process...[/cyan]")

    # Calculate grid dimensions
    num_cols = args.cols
    num_rows = math.ceil(num_images / num_cols)
    
    # Each cell contains: before_image + gap + after_image
    pair_width = args.cell_width * 2 + args.gap
    pair_height = args.cell_width  # Assuming square cells
    
    # Add padding between rows and columns
    padding = 10
    
    # Calculate final grid size
    grid_width = num_cols * pair_width + (num_cols - 1) * padding
    grid_height = num_rows * pair_height + (num_rows - 1) * padding
    
    console.print(f"  [cyan]Grid Layout:[/cyan] {num_cols} pairs per row x {num_rows} rows")
    console.print(f"  [cyan]Pair Size:[/cyan] {pair_width} x {pair_height} px (per before/after pair)")
    console.print(f"  [cyan]Final Size:[/cyan] {grid_width} x {grid_height} px\n")
    
    # Create a new white background image
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Process each image pair
    matched_pairs = 0
    for i, before_path in enumerate(tqdm(before_paths, desc="Processing image pairs")):
        try:
            # Find corresponding after image (same filename)
            after_path = args.after_dir / before_path.name
            
            if not after_path.exists():
                console.print(f"[yellow]Warning: No matching after image for {before_path.name}[/yellow]")
                continue
            
            # Open and resize both images
            before_img = Image.open(before_path)
            after_img = Image.open(after_path)
            
            # Resize to cell size (maintaining aspect ratio)
            cell_size = (args.cell_width, args.cell_width)
            before_img.thumbnail(cell_size, Image.Resampling.LANCZOS)
            after_img.thumbnail(cell_size, Image.Resampling.LANCZOS)
            
            # Create cells with white background
            before_cell = Image.new('RGB', cell_size, 'white')
            after_cell = Image.new('RGB', cell_size, 'white')
            
            # Center images in cells
            before_x = (cell_size[0] - before_img.width) // 2
            before_y = (cell_size[1] - before_img.height) // 2
            before_cell.paste(before_img, (before_x, before_y))
            
            after_x = (cell_size[0] - after_img.width) // 2
            after_y = (cell_size[1] - after_img.height) // 2
            after_cell.paste(after_img, (after_x, after_y))
            
            # Add labels if requested
            if args.labels:
                before_cell = add_label(before_cell, "Before", "top")
                after_cell = add_label(after_cell, "After", "top")
            
            # Calculate position in grid
            row = matched_pairs // num_cols
            col = matched_pairs % num_cols
            
            # Calculate base position for this pair
            base_x = col * (pair_width + padding)
            base_y = row * (pair_height + padding)
            
            # Paste before and after images
            grid_image.paste(before_cell, (base_x, base_y))
            grid_image.paste(after_cell, (base_x + args.cell_width + args.gap, base_y))
            
            matched_pairs += 1
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to process {before_path.name}: {e}[/yellow]")

    # Save the final grid
    output_file = args.output_dir / "comparison_grid.png"
    grid_image.save(output_file)
    
    console.print(f"\n[bold green]✓ Successfully created comparison grid with {matched_pairs} image pairs[/bold green]")
    console.print(f"[bold green]✓ Saved to: {output_file}[/bold green]\n")
    
    # Also create individual before/after grids
    console.print("[cyan]Creating individual grids...[/cyan]")
    
    # Before-only grid
    create_single_grid(
        before_paths[:matched_pairs],
        args.output_dir / "before_grid.png",
        args.cols,
        args.cell_width,
        "Before: Original Scene"
    )
    
    # After-only grid
    after_paths = [args.after_dir / p.name for p in before_paths[:matched_pairs]]
    create_single_grid(
        after_paths,
        args.output_dir / "after_grid.png",
        args.cols,
        args.cell_width,
        "After: Object Removed"
    )
    
    console.print(f"\n[bold green]All grids saved to: {args.output_dir}/[/bold green]")
    console.print(f"  - comparison_grid.png (side-by-side)")
    console.print(f"  - before_grid.png")
    console.print(f"  - after_grid.png\n")


def create_single_grid(image_paths, output_file, cols, cell_size, title=None):
    """Create a single grid from a list of image paths."""
    num_images = len(image_paths)
    num_rows = math.ceil(num_images / cols)
    
    padding = 10
    grid_width = cols * cell_size + (cols - 1) * padding
    grid_height = num_rows * cell_size + (num_rows - 1) * padding
    
    # Add space for title if provided
    title_height = 40 if title else 0
    grid_image = Image.new('RGB', (grid_width, grid_height + title_height), 'white')
    
    # Add title
    if title:
        draw = ImageDraw.Draw(grid_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = (grid_width - text_width) // 2
        draw.text((x, 10), title, font=font, fill="black")
    
    # Process each image
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            img.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
            
            cell = Image.new('RGB', (cell_size, cell_size), 'white')
            paste_x = (cell_size - img.width) // 2
            paste_y = (cell_size - img.height) // 2
            cell.paste(img, (paste_x, paste_y))
            
            row = i // cols
            col = i % cols
            
            grid_x = col * (cell_size + padding)
            grid_y = row * (cell_size + padding) + title_height
            
            grid_image.paste(cell, (grid_x, grid_y))
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to process {path.name}: {e}[/yellow]")
    
    grid_image.save(output_file)
    console.print(f"  [green]✓ Saved {output_file.name}[/green]")


if __name__ == "__main__":
    main()
