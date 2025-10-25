#!/usr/bin/env python3
"""
03_ground_text_to_masks.py - Text → Boxes → Masks (GroundingDINO + SAM2)

Goal: From text, produce 2D masks isolating the target object on each rendered view.

Inputs:
  --images_root: Path to rendered images (e.g., outputs/garden/round_001/pre_edit/train/)
  --text: Text prompt for object detection (e.g., "plant pot . potted plant . planter . flowerpot")
  --output_dir: Directory to save masks (default: outputs/<dataset_name>/round_001/masks/)
  --dino_thresh: GroundingDINO detection threshold (default: 0.30)
  --sam_model: SAM2 model size ("tiny", "small", "base_plus", "large", default: "large")

Outputs (saved in output_dir):
  - boxes/box_view_000.png: Visualization with boxes
  - sam_masks/mask_view_000.png: Binary PNG (0/255)
  - sam_masks/mask_view_000.npy: Float mask in [0,1]
  - overlays/overlay_view_000.png: Mask overlay on image
  - coverage.csv: Per-view mask area %, detection scores
  - manifest.json: Text prompt, thresholds, model versions, seed
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Check for required packages
try:
    from groundingdino.util.inference import load_model, load_image, predict
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# Check if models are available
if not GROUNDING_DINO_AVAILABLE or not SAM2_AVAILABLE:
    print("\n" + "=" * 80)
    print("ERROR: Required models not installed!")
    print("=" * 80)
    if not GROUNDING_DINO_AVAILABLE:
        print("✗ GroundingDINO is not installed")
    if not SAM2_AVAILABLE:
        print("✗ SAM2 is not installed")
    print("\nTo install and download model weights, run ONE of the following:")
    print("\n  Option 1 (Recommended - Full Setup):")
    print("    chmod +x setup_vast.sh download_models.sh")
    print("    ./setup_vast.sh")
    print("\n  Option 2 (Models Only):")
    print("    chmod +x download_models.sh")
    print("    ./download_models.sh")
    print("    pip install groundingdino-py")
    print("    pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    print("\n" + "=" * 80 + "\n")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate masks from text using GroundingDINO + SAM2")
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Path to directory containing rendered images",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt for object detection (use . to separate synonyms)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for masks",
    )
    parser.add_argument(
        "--dino_thresh",
        type=float,
        default=0.30,
        help="GroundingDINO detection threshold",
    )
    parser.add_argument(
        "--box_thresh",
        type=float,
        default=0.25,
        help="Box confidence threshold",
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="large",
        choices=["tiny", "small", "base_plus", "large"],
        help="SAM2 model size",
    )
    parser.add_argument(
        "--dino_config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config file",
    )
    parser.add_argument(
        "--dino_checkpoint",
        type=str,
        default="groundingdino_swint_ogc.pth",
        help="GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=None,
        help="SAM2 checkpoint path (auto-determined if not provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_grounding_dino(config_path, checkpoint_path, device="cuda"):
    """Load GroundingDINO model."""
    print(f"Loading GroundingDINO from {checkpoint_path}...")
    model = load_model(config_path, checkpoint_path, device=device)
    print("✓ GroundingDINO loaded")
    return model


def load_sam2_model(model_size, checkpoint_path=None, device="cuda"):
    """Load SAM2 model."""
    # SAM2 model configs
    sam2_configs = {
        "tiny": "sam2_hiera_t.yaml",
        "small": "sam2_hiera_s.yaml",
        "base_plus": "sam2_hiera_b+.yaml",
        "large": "sam2_hiera_l.yaml",
    }
    
    # Default checkpoint paths
    if checkpoint_path is None:
        sam2_checkpoints = {
            "tiny": "sam2_hiera_tiny.pt",
            "small": "sam2_hiera_small.pt",
            "base_plus": "sam2_hiera_base_plus.pt",
            "large": "sam2_hiera_large.pt",
        }
        checkpoint_path = sam2_checkpoints[model_size]
    
    config_file = sam2_configs[model_size]
    
    print(f"Loading SAM2 ({model_size}) from {checkpoint_path}...")
    sam2_model = build_sam2(config_file, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✓ SAM2 loaded")
    return predictor


def detect_objects(image_path, text_prompt, dino_model, box_threshold, text_threshold):
    """Detect objects using GroundingDINO."""
    # Load image for DINO
    image_source, image = load_image(str(image_path))
    
    # Predict boxes
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    
    # Convert boxes to xyxy format
    h, w, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
    
    return image_source, boxes_xyxy.cpu().numpy(), logits.cpu().numpy(), phrases


def segment_with_sam2(image_np, boxes, sam_predictor):
    """Generate masks using SAM2."""
    # Set image
    sam_predictor.set_image(image_np)
    
    if len(boxes) == 0:
        return None, None
    
    # Convert boxes to SAM2 format
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    
    # Predict masks
    masks, scores, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Combine masks (take union if multiple detections)
    if len(masks) > 0:
        combined_mask = masks.max(axis=0)  # Union of all masks
        return combined_mask, scores
    
    return None, None


def visualize_boxes(image, boxes, logits, output_path):
    """Visualize detection boxes on image."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for box, logit in zip(boxes, logits):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = plt.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"{logit:.2f}",
            color='red',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_overlay(image, mask, output_path):
    """Visualize mask overlay on image."""
    overlay = image.copy()
    # Create colored mask (red with transparency)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = 255  # Red channel
    
    # Apply mask with transparency
    alpha = 0.5
    overlay[mask > 0.5] = (
        alpha * mask_colored[mask > 0.5] + 
        (1 - alpha) * image[mask > 0.5]
    ).astype(np.uint8)
    
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Determine output directory
    if args.output_dir is None:
        # Try to infer from images_root structure
        images_path = Path(args.images_root)
        if "round_" in str(images_path):
            round_dir = images_path.parent.parent
            args.output_dir = round_dir / "masks"
        else:
            args.output_dir = images_path.parent / "masks"
    
    output_dir = Path(args.output_dir)
    boxes_dir = output_dir / "boxes"
    masks_dir = output_dir / "sam_masks"
    overlays_dir = output_dir / "overlays"
    
    boxes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Text-to-Masks Generation")
    print("=" * 80)
    print(f"Images root: {args.images_root}")
    print(f"Text prompt: {args.text}")
    print(f"Output directory: {output_dir}")
    print(f"DINO threshold: {args.dino_thresh}")
    print(f"SAM2 model: {args.sam_model}")
    print()
    
    # Load models
    try:
        dino_model = load_grounding_dino(
            args.dino_config,
            args.dino_checkpoint,
            device=device
        )
    except Exception as e:
        print(f"ERROR loading GroundingDINO: {e}")
        print("Make sure you have the model weights downloaded.")
        sys.exit(1)
    
    try:
        sam_predictor = load_sam2_model(
            args.sam_model,
            args.sam_checkpoint,
            device=device
        )
    except Exception as e:
        print(f"ERROR loading SAM2: {e}")
        print("Make sure you have the model weights downloaded.")
        sys.exit(1)
    
    print()
    
    # Get list of images
    images_root = Path(args.images_root)
    image_files = sorted(images_root.glob("*.png")) + sorted(images_root.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {images_root}")
        sys.exit(1)
    
    print(f"Processing {len(image_files)} images...")
    print()
    
    # Process each image
    coverage_data = []
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files):
        img_name = img_path.stem
        
        try:
            # Detect objects with GroundingDINO
            image_np, boxes, logits, phrases = detect_objects(
                img_path,
                args.text,
                dino_model,
                args.box_thresh,
                args.dino_thresh,
            )
            
            # Visualize boxes
            box_vis_path = boxes_dir / f"box_{img_name}.png"
            visualize_boxes(image_np, boxes, logits, box_vis_path)
            
            if len(boxes) == 0:
                # No detection
                coverage_data.append({
                    "image": img_name,
                    "num_detections": 0,
                    "max_score": 0.0,
                    "mask_coverage": 0.0,
                    "status": "no_detection"
                })
                failed += 1
                # Save empty mask
                h, w = image_np.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(masks_dir / f"mask_{img_name}.png"), empty_mask)
                np.save(masks_dir / f"mask_{img_name}.npy", empty_mask.astype(np.float32))
                continue
            
            # Segment with SAM2
            mask, scores = segment_with_sam2(image_np, boxes, sam_predictor)
            
            if mask is None:
                coverage_data.append({
                    "image": img_name,
                    "num_detections": len(boxes),
                    "max_score": float(logits.max()),
                    "mask_coverage": 0.0,
                    "status": "segmentation_failed"
                })
                failed += 1
                continue
            
            # Calculate coverage
            mask_binary = (mask > 0.5).astype(np.uint8)
            coverage = (mask_binary.sum() / mask_binary.size) * 100
            
            # Save mask
            mask_uint8 = (mask_binary * 255).astype(np.uint8)
            cv2.imwrite(str(masks_dir / f"mask_{img_name}.png"), mask_uint8)
            np.save(masks_dir / f"mask_{img_name}.npy", mask.astype(np.float32))
            
            # Create overlay
            overlay_path = overlays_dir / f"overlay_{img_name}.png"
            visualize_overlay(image_np, mask_binary, overlay_path)
            
            # Record stats
            coverage_data.append({
                "image": img_name,
                "num_detections": len(boxes),
                "max_score": float(logits.max()),
                "mask_coverage": coverage,
                "status": "success"
            })
            successful += 1
            
        except Exception as e:
            print(f"\nERROR processing {img_name}: {e}")
            coverage_data.append({
                "image": img_name,
                "num_detections": 0,
                "max_score": 0.0,
                "mask_coverage": 0.0,
                "status": f"error: {str(e)}"
            })
            failed += 1
    
    print()
    print(f"✓ Processed {len(image_files)} images: {successful} successful, {failed} failed")
    print()
    
    # Save coverage CSV
    csv_path = output_dir / "coverage.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "num_detections", "max_score", "mask_coverage", "status"])
        writer.writeheader()
        writer.writerows(coverage_data)
    
    print(f"Coverage stats saved to: {csv_path}")
    
    # Calculate summary stats
    successful_masks = [d for d in coverage_data if d["status"] == "success"]
    if successful_masks:
        avg_coverage = np.mean([d["mask_coverage"] for d in successful_masks])
        avg_score = np.mean([d["max_score"] for d in successful_masks])
        print(f"Average mask coverage: {avg_coverage:.2f}%")
        print(f"Average detection score: {avg_score:.3f}")
    
    # Create manifest
    manifest = {
        "module": "03_ground_text_to_masks",
        "timestamp": datetime.now().isoformat(),
        "images_root": str(args.images_root),
        "output_dir": str(output_dir),
        "parameters": {
            "text_prompt": args.text,
            "dino_thresh": args.dino_thresh,
            "box_thresh": args.box_thresh,
            "sam_model": args.sam_model,
            "seed": args.seed,
        },
        "statistics": {
            "num_images": len(image_files),
            "successful": successful,
            "failed": failed,
            "avg_coverage": float(np.mean([d["mask_coverage"] for d in successful_masks])) if successful_masks else 0.0,
            "avg_score": float(np.mean([d["max_score"] for d in successful_masks])) if successful_masks else 0.0,
        },
        "outputs": {
            "boxes_dir": str(boxes_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "coverage_csv": str(csv_path),
        },
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_path}")
    print()
    print("=" * 80)
    print("MASK GENERATION COMPLETE")
    print("=" * 80)
    print(f"✓ Boxes: {boxes_dir}")
    print(f"✓ Masks: {masks_dir}")
    print(f"✓ Overlays: {overlays_dir}")
    print()
    print("Next step: Review overlays to verify mask quality, then use for ROI lifting (04).")
    print("=" * 80)


if __name__ == "__main__":
    main()
