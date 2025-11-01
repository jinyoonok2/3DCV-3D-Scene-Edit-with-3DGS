#!/usr/bin/env python3
"""
test_ip2p_single.py - Quick InstructPix2Pix Test on Single Image

Test IP2P with different parameters on one image to find optimal settings.

Usage:
    python test_ip2p_single.py \
        --image outputs/garden/round_001/pre_edit/train/pre_edit_view_000.png \
        --mask outputs/garden/round_001/masks_brown_plant/sam_masks/mask_view_000.npy \
        --instruction "Replace the brown plant with a black metal cup" \
        --image_guidance_scale 2.5 \
        --guidance_scale 10.0 \
        --steps 50 \
        --output test_result.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Test IP2P on single image")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mask", type=str, required=True, help="Mask path (.npy or .png)")
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Edit instruction"
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.5,
        help="Image guidance scale (how much to change)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.0,
        help="Text guidance scale (CFG)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_ip2p_result.png",
        help="Output path for result",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Skip mask blending (apply IP2P to full image)",
    )
    return parser.parse_args()


def load_mask(mask_path):
    """Load mask from disk."""
    mask_path = Path(mask_path)
    
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.float32) / 255.0
    
    return mask


def apply_mask_blending(original, edited, mask, blend_margin=10):
    """Blend edited region with original using mask."""
    original_np = np.array(original).astype(np.float32) / 255.0
    edited_np = np.array(edited).astype(np.float32) / 255.0
    
    # Resize mask to match edited image
    target_h, target_w = edited_np.shape[:2]
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Resize original to match edited
    if original_np.shape[:2] != edited_np.shape[:2]:
        original_np = cv2.resize(original_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Feather mask edges
    if blend_margin > 0:
        kernel_size = blend_margin * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), blend_margin / 3)
    
    # Expand mask to 3 channels
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    if mask.shape[2] == 1:
        mask = np.repeat(mask, 3, axis=2)
    
    # Blend
    blended = original_np * (1 - mask) + edited_np * mask
    blended = (blended * 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def create_comparison(original, edited, mask, output_path):
    """Create side-by-side comparison."""
    import matplotlib.pyplot as plt
    
    # Convert mask for visualization
    mask_vis = mask
    if mask_vis.ndim == 2:
        mask_vis = np.stack([mask_vis] * 3, axis=-1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")
    
    axes[1].imshow(mask_vis, cmap='gray' if mask.ndim == 2 else None)
    axes[1].set_title("Mask", fontsize=14)
    axes[1].axis("off")
    
    axes[2].imshow(edited)
    axes[2].set_title("IP2P Edited (Full)", fontsize=14)
    axes[2].axis("off")
    
    # Show blended if mask provided
    axes[3].imshow(edited)
    axes[3].set_title("Final (Masked Blend)", fontsize=14)
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved comparison to {output_path}")


def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("InstructPix2Pix Single Image Test")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    print(f"Instruction: {args.instruction}")
    print(f"Image guidance: {args.image_guidance_scale}")
    print(f"Text guidance: {args.guidance_scale}")
    print(f"Steps: {args.steps}")
    print(f"Device: {device}")
    print()
    
    # Load image
    print("Loading image...")
    original_image = Image.open(args.image).convert("RGB")
    print(f"  Image size: {original_image.size}")
    
    # Load mask
    if not args.no_mask:
        print("Loading mask...")
        mask = load_mask(args.mask)
        print(f"  Mask size: {mask.shape}")
        print(f"  Mask coverage: {(mask > 0.5).sum() / mask.size * 100:.1f}%")
        print()
    
    # Load IP2P model
    print("Loading InstructPix2Pix model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    print("✓ Model loaded")
    print()
    
    # Apply IP2P
    print("Applying InstructPix2Pix...")
    print(f"  This will take ~{args.steps * 0.05:.0f}-{args.steps * 0.1:.0f} seconds...")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    edited_image = pipe(
        prompt=args.instruction,
        image=original_image,
        num_inference_steps=args.steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    
    print("✓ IP2P complete")
    print()
    
    # Apply mask blending if mask provided
    if not args.no_mask:
        print("Applying mask blending...")
        final_image = apply_mask_blending(original_image, edited_image, mask)
        print("✓ Blending complete")
    else:
        final_image = edited_image
        print("⏭️  Skipping mask blending (--no_mask)")
    
    print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save final result
    final_image.save(output_path)
    print(f"✓ Saved result to {output_path}")
    
    # Save comparison
    if not args.no_mask:
        comparison_path = output_path.with_name(output_path.stem + "_comparison.png")
        create_comparison(original_image, final_image, mask, comparison_path)
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Result: {output_path}")
    if not args.no_mask:
        print(f"Comparison: {comparison_path}")
    print()
    print("Next steps:")
    print("  1. View the result image")
    print("  2. If not good enough, try:")
    print("     - Higher --image_guidance_scale (e.g., 3.5, 4.5)")
    print("     - Higher --guidance_scale (e.g., 12.0, 15.0)")
    print("     - More --steps (e.g., 75, 100)")
    print("     - Different --instruction wording")
    print("  3. Once satisfied, run full pipeline (step 05)")
    print("=" * 80)


if __name__ == "__main__":
    main()
