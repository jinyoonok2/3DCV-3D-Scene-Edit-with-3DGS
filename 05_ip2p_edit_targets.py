#!/usr/bin/env python3
"""
05_ip2p_edit_targets.py - Apply InstructPix2Pix to Create Edited Targets

Goal: Apply image editing to pre-edit rendered views using InstructPix2Pix
      to create target images for 3DGS optimization.

Inputs:
  --images_root: Directory with pre-edit rendered images
  --instruction: Text instruction for editing (e.g., "Replace the brown plant with an empty black cup")
  --output_dir: Output directory for edited targets
  --masks_root: (Optional) Directory with 2D masks to constrain edits
  --ip2p_steps: Number of diffusion steps (default: 50)
  --image_guidance_scale: Guidance scale for image conditioning (default: 1.5)
  --guidance_scale: CFG scale (default: 7.5)
  --seed: Random seed for reproducibility (default: 42)

Outputs (saved in output_dir/edited_targets/):
  - edited_view_XXX.png: Edited images
  - side_by_side/view_XXX_comparison.png: Pre-edit vs edited comparison
  - manifest.json: Parameters and metadata
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Try to import diffusers for InstructPix2Pix
try:
    from diffusers import StableDiffusionInstructPix2PixPipeline
except ImportError:
    print("ERROR: diffusers not installed. Run: pip install diffusers")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply InstructPix2Pix to create edited targets")
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Directory with pre-edit rendered images",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Text instruction for editing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for edited targets",
    )
    parser.add_argument(
        "--masks_root",
        type=str,
        default=None,
        help="Optional: Directory with 2D masks to constrain edits",
    )
    parser.add_argument(
        "--ip2p_steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Guidance scale for image conditioning",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_ip2p_model(device="cuda"):
    """Load InstructPix2Pix model."""
    print("Loading InstructPix2Pix model...")
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("✓ Model loaded")
    return pipe


def load_images(images_root):
    """Load all images from directory."""
    images_root = Path(images_root)
    image_files = sorted(images_root.glob("*.png")) + sorted(images_root.glob("*.jpg"))
    
    images = {}
    for img_file in image_files:
        img_name = img_file.stem
        images[img_name] = img_file
    
    return images


def load_masks(masks_root, image_names):
    """Load masks corresponding to image names."""
    if masks_root is None:
        return {}
    
    masks_root = Path(masks_root)
    masks = {}
    
    for img_name in image_names:
        # Try different naming conventions
        for mask_name in [f"mask_{img_name}", img_name]:
            for ext in [".npy", ".png", ".jpg"]:
                mask_path = masks_root / f"{mask_name}{ext}"
                if mask_path.exists():
                    # Load mask
                    if ext == ".npy":
                        mask = np.load(mask_path)
                    else:
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        mask = mask.astype(np.float32) / 255.0
                    
                    masks[img_name] = mask
                    break
            if img_name in masks:
                break
    
    return masks


def apply_ip2p(pipe, image, instruction, args, generator):
    """Apply InstructPix2Pix to a single image."""
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    elif isinstance(image, Path):
        image = Image.open(image).convert("RGB")
    
    # Apply InstructPix2Pix
    edited = pipe(
        instruction,
        image=image,
        num_inference_steps=args.ip2p_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    
    return edited


def apply_mask_blending(original, edited, mask, blend_margin=10):
    """Blend edited region with original using mask."""
    # Convert to numpy
    original_np = np.array(original).astype(np.float32) / 255.0
    edited_np = np.array(edited).astype(np.float32) / 255.0
    
    # Resize mask to match EDITED image (IP2P may have resized)
    target_h, target_w = edited_np.shape[:2]
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Resize original to match edited if they differ
    if original_np.shape[:2] != edited_np.shape[:2]:
        original_np = cv2.resize(original_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Feather mask edges for smooth blending (operates on 2D mask)
    if blend_margin > 0:
        kernel_size = blend_margin * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), blend_margin / 3)
    
    # Expand mask to 3 channels AFTER blur
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    
    # Ensure mask has 3 channels
    if mask.shape[2] == 1:
        mask = np.repeat(mask, 3, axis=2)
    
    # Blend
    blended = original_np * (1 - mask) + edited_np * mask
    blended = (blended * 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def create_comparison(original, edited, output_path):
    """Create side-by-side comparison visualization."""
    import matplotlib.pyplot as plt
    
    # Resize original to match edited if dimensions differ
    original_np = np.array(original)
    edited_np = np.array(edited)
    if original_np.shape[:2] != edited_np.shape[:2]:
        original_np = cv2.resize(original_np, (edited_np.shape[1], edited_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_np)
    axes[0].set_title("Pre-Edit")
    axes[0].axis('off')
    
    # Edited
    axes[1].imshow(edited_np)
    axes[1].set_title("Edited (IP2P)")
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original_np.astype(np.float32) - edited_np.astype(np.float32))
    diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    axes[2].imshow(diff)
    axes[2].set_title("Difference")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    edited_dir = output_dir / "edited_targets"
    comparison_dir = edited_dir / "side_by_side"
    edited_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("InstructPix2Pix Editing")
    print("=" * 80)
    print(f"Images root: {args.images_root}")
    print(f"Instruction: {args.instruction}")
    print(f"Output dir: {edited_dir}")
    print(f"Masks root: {args.masks_root}")
    print()
    
    # Load images
    print("Loading images...")
    images = load_images(args.images_root)
    print(f"✓ Loaded {len(images)} images")
    
    # Load masks if provided
    masks = load_masks(args.masks_root, images.keys()) if args.masks_root else {}
    if masks:
        print(f"✓ Loaded {len(masks)} masks")
    print()
    
    # Load InstructPix2Pix model
    pipe = load_ip2p_model(device)
    print()
    
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Process each image
    print(f"Applying InstructPix2Pix with instruction: '{args.instruction}'")
    for img_name, img_path in tqdm(images.items(), desc="Editing images"):
        # Load original image
        original = Image.open(img_path).convert("RGB")
        
        # Apply InstructPix2Pix
        edited = apply_ip2p(pipe, original, args.instruction, args, generator)
        
        # Apply mask blending if mask is available
        if img_name in masks:
            edited = apply_mask_blending(original, edited, masks[img_name])
        
        # Save edited image
        edited_path = edited_dir / f"edited_{img_name}.png"
        edited.save(edited_path)
        
        # Create comparison visualization
        comparison_path = comparison_dir / f"{img_name}_comparison.png"
        create_comparison(original, edited, comparison_path)
    
    print()
    print(f"✓ Saved {len(images)} edited images to {edited_dir}")
    print(f"✓ Saved {len(images)} comparison visualizations to {comparison_dir}")
    print()
    
    # Save manifest
    manifest = {
        "instruction": args.instruction,
        "images_root": str(args.images_root),
        "masks_root": str(args.masks_root) if args.masks_root else None,
        "num_images": len(images),
        "num_masks": len(masks),
        "ip2p_params": {
            "steps": args.ip2p_steps,
            "image_guidance_scale": args.image_guidance_scale,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        },
        "model": "timbrooks/instruct-pix2pix",
        "timestamp": datetime.now().isoformat(),
    }
    
    manifest_path = edited_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Saved manifest to {manifest_path}")
    print()
    
    print("=" * 80)
    print("EDITING COMPLETE")
    print("=" * 80)
    print(f"✓ Edited targets: {edited_dir}")
    print(f"✓ Comparisons: {comparison_dir}")
    print()
    print("Next step: Optimize 3DGS with ROI (Module 06)")
    print("=" * 80)


if __name__ == "__main__":
    main()
