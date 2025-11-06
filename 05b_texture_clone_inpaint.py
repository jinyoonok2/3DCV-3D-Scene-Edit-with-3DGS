#!/usr/bin/env python3
"""
05b_texture_clone_inpaint.py - Texture Cloning Inpainting (No SDXL)

Alternative to SDXL: Clone texture from surrounding regions to fill holes.
Better for cases where you just want to continue existing patterns.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from project_utils.config import ProjectConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Texture-clone inpainting")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--holed_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def texture_clone_inpaint(image, mask):
    """
    Use OpenCV inpainting (Telea or Navier-Stokes) to fill holes.
    This clones surrounding texture without "understanding" the scene.
    """
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask)
    
    # Ensure mask is binary
    mask_cv = (mask_cv > 127).astype(np.uint8) * 255
    
    # Telea inpainting - good for texture continuation
    result = cv2.inpaint(img_cv, mask_cv, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    
    # Convert back to PIL
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def main():
    args = parse_args()
    config = ProjectConfig(args.config)
    
    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')
    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_texture_cloned')
    
    holed_dir = Path(holed_dir)
    renders_dir = holed_dir / "holed" / "train"
    masks_dir = holed_dir / "masks" / "train"
    
    output_dir = Path(output_dir)
    targets_dir = output_dir / "targets" / "train"
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Texture cloning inpainting...")
    print(f"Input: {holed_dir}")
    print(f"Output: {output_dir}")
    
    image_files = sorted(renders_dir.glob("*.png"))
    
    for img_path in tqdm(image_files, desc="Cloning textures"):
        idx = img_path.stem
        mask_path = masks_dir / f"{idx}.png"
        
        if not mask_path.exists():
            print(f"Warning: No mask for {idx}, copying original")
            img = Image.open(img_path)
            img.save(targets_dir / f"{idx}.png")
            continue
        
        # Load
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Texture clone inpaint
        result = texture_clone_inpaint(img, mask)
        
        # Save
        result.save(targets_dir / f"{idx}.png")
    
    print(f"✓ Done! Saved to {targets_dir}")
    print(f"Next: python 05c_optimize_to_targets.py")


if __name__ == "__main__":
    main()
