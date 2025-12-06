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

# Import project config
from project_utils.config import ProjectConfig

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
        "--config",
        type=str,
        default="configs/garden_config.yaml",
        help="Path to config file (default: configs/garden_config.yaml)",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Path to directory containing rendered images (overrides config)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt for object detection (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for masks (overrides config)",
    )
    parser.add_argument(
        "--dino_thresh",
        type=float,
        default=None,
        help="GroundingDINO detection threshold (overrides config)",
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
        default=None,
        help="GroundingDINO config file (will auto-detect if not provided)",
    )
    parser.add_argument(
        "--dino_checkpoint",
        type=str,
        default="models/groundingdino_swint_ogc.pth",
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
    parser.add_argument(
        "--select_index",
        type=int,
        default=-1,
        help="Select the detection index (0-based) to use for segmentation for all images (-1 = disabled)",
    )
    parser.add_argument(
        "--select_index_file",
        type=str,
        default=None,
        help="Path to JSON file mapping image_stem -> index to select per-image (overrides --select_index)",
    )
    parser.add_argument(
        "--manual_box_file",
        type=str,
        default=None,
        help="Path to JSON file mapping image_stem -> [x1,y1,x2,y2] (pixel coords) to force a manual box per image",
    )
    parser.add_argument(
        "--reference_box",
        type=str,
        default=None,
        help="Reference bounding box [x1,y1,x2,y2] in pixels or normalized [0-1]. Filter detections by overlap with this box.",
    )
    parser.add_argument(
        "--reference_box_normalized",
        action="store_true",
        help="If set, reference_box coordinates are normalized [0-1], otherwise pixel coordinates",
    )
    parser.add_argument(
        "--reference_overlap_thresh",
        type=float,
        default=0.5,
        help="Minimum fraction of detection box that must overlap with reference_box (0.0-1.0)",
    )
    parser.add_argument(
        "--dino_selection",
        type=str,
        default=None,
        choices=["largest", "confidence", None],
        help="Box selection from GroundingDINO: 'confidence' (highest DINO score), 'largest' (largest box area), or None (use all boxes)",
    )
    parser.add_argument(
        "--sam_selection",
        type=str,
        default=None,
        choices=["confidence", None],
        help="Mask selection from SAM2: 'confidence' (highest SAM2 score only), or None (save all masks)",
    )
    parser.add_argument(
        "--sam_thresh",
        type=float,
        default=0.5,
        help="SAM2 mask confidence threshold (0.0-1.0). Only masks above this confidence are kept. Lower=more permissive (try 0.3-0.5)",
    )
    return parser.parse_args()


def load_grounding_dino(config_path, checkpoint_path, device="cuda"):
    """Load GroundingDINO model."""
    # Auto-detect config path if not provided
    if config_path is None:
        # Try common locations
        possible_paths = []
        
        # Check in site-packages first (groundingdino-py)
        try:
            import groundingdino
            pkg_path = Path(groundingdino.__file__).parent
            possible_paths.extend([
                str(pkg_path / "config" / "GroundingDINO_SwinT_OGC.py"),
                str(pkg_path / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
            ])
        except:
            pass
        
        # Check local clone
        possible_paths.extend([
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ])
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "\n" + "="*80 + "\n"
                "ERROR: GroundingDINO config file not found!\n"
                "="*80 + "\n"
                "Please clone GroundingDINO repository (config files only):\n\n"
                "  git clone https://github.com/IDEA-Research/GroundingDINO.git\n\n"
                "Note: You don't need to install it with pip, just having the repo\n"
                "      cloned for config files is enough.\n"
                "="*80
            )
    
    print(f"Loading GroundingDINO from {checkpoint_path}...")
    print(f"Using config: {config_path}")
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
            "tiny": "models/sam2_hiera_tiny.pt",
            "small": "models/sam2_hiera_small.pt",
            "base_plus": "models/sam2_hiera_base_plus.pt",
            "large": "models/sam2_hiera_large.pt",
        }
        checkpoint_path = sam2_checkpoints[model_size]
    
    config_file = sam2_configs[model_size]
    
    print(f"Loading SAM2 ({model_size}) from {checkpoint_path}...")
    sam2_model = build_sam2(config_file, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✓ SAM2 loaded")
    return predictor


def compute_box_overlap(box1, box2):
    """
    Compute the fraction of box1 that overlaps with box2.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        overlap_fraction: fraction of box1's area that overlaps with box2 (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if box1_area == 0:
        return 0.0
    
    return intersection_area / box1_area


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
    boxes_np = boxes_xyxy.cpu().numpy()
    logits_np = logits.cpu().numpy()
    
    return image_source, boxes_np, logits_np, phrases


def segment_with_sam2(image_np, boxes, sam_predictor, sam_selection=None, sam_thresh=0.5):
    """Generate masks using SAM2.
    
    Args:
        image_np: Image array
        boxes: Detection boxes from GroundingDINO (already filtered by dino_selection)
        sam_predictor: SAM2 predictor
        sam_selection: 'confidence' (highest SAM2 score only), or None (keep all masks)
        sam_thresh: SAM2 confidence threshold - only keep masks with confidence >= this value
    
    Returns:
        masks_to_save: List of masks to save (H, W) - can be empty if none pass threshold
        all_masks: List of all individual masks that passed threshold (for visualization)
        selected_indices: List of indices that should be saved (highlighted green in overlay)
        all_sam_scores: List of SAM2 confidence scores for each mask
    """
    # Set image
    sam_predictor.set_image(image_np)
    
    if len(boxes) == 0:
        return [], [], [], []
    
    # Convert boxes to SAM2 format
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    
    # Predict masks for each box individually
    all_masks = []
    all_sam_scores = []
    
    for box in input_boxes:
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box.unsqueeze(0),  # SAM2 expects (1, 4) for single box
            multimask_output=False,
        )
        # masks shape: (1, 1, H, W) or (1, H, W)
        # Extract the single mask: (H, W)
        mask_2d = masks.squeeze()
        sam_score = scores.max().item()
        
        # Only keep masks above SAM2 confidence threshold
        if sam_score >= sam_thresh:
            all_masks.append(mask_2d)
            all_sam_scores.append(sam_score)
    
    # If no masks passed threshold, return empty
    if len(all_masks) == 0:
        return [], [], [], []
    
    # Select based on sam_selection
    if sam_selection == "confidence":
        # Find the mask with highest SAM2 confidence - save only that one
        selected_idx = np.argmax(all_sam_scores)
        return [all_masks[selected_idx]], all_masks, [selected_idx], all_sam_scores
    else:
        # Save all masks that passed threshold
        return all_masks, all_masks, list(range(len(all_masks))), all_sam_scores


def visualize_boxes(image, boxes, logits, phrases, output_path, reference_box=None, selected_indices=None):
    """Visualize detection boxes on image. Annotates index, phrase and score.
    
    Args:
        image: Image array
        boxes: Detection boxes
        logits: Detection scores
        phrases: Detection phrases
        output_path: Where to save
        reference_box: Optional [x1,y1,x2,y2] reference box to draw in blue
        selected_indices: Optional list of indices that will be used (highlight in green)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw reference box first (so it's in background)
    if reference_box is not None:
        x1, y1, x2, y2 = reference_box
        w, h = x2 - x1, y2 - y1
        ref_rect = plt.Rectangle(
            (x1, y1), w, h,
            linewidth=3,
            edgecolor='blue',
            facecolor='none',
            linestyle='--',
            label='Reference Box'
        )
        ax.add_patch(ref_rect)
        ax.text(
            x1, y1 - 25,
            "Reference Box",
            color='blue',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9)
        )
    
    # Draw detection boxes
    for i, (box, logit) in enumerate(zip(boxes, logits)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Highlight selected boxes in green, others in red
        is_selected = selected_indices is not None and i in selected_indices
        color = 'green' if is_selected else 'red'
        linewidth = 3 if is_selected else 2
        
        rect = plt.Rectangle(
            (x1, y1), w, h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        label = f"#{i} {phrases[i] if i < len(phrases) else ''} {logit:.2f}"
        if is_selected:
            label += " [SELECTED]"
        ax.text(
            x1, y1 - 8,
            label,
            color=color,
            fontsize=10,
            fontweight='bold' if is_selected else 'normal',
            bbox=dict(facecolor='white', alpha=0.9 if is_selected else 0.8)
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_overlay(image, mask, output_path, all_masks=None, selected_idx=None, reference_box=None):
    """Visualize mask overlay on image.
    
    Args:
        image: Image array
        mask: Final mask (H, W) - what's actually used
        output_path: Where to save
        all_masks: Optional list of all individual masks (for multi-mask visualization)
        selected_idx: Optional index of the selected mask (highlighted in green)
        reference_box: Optional [x1,y1,x2,y2] reference box to draw
    """
    import matplotlib.pyplot as plt
    
    # Always show all masks with selection highlighted (if we have mask info)
    if all_masks is not None and len(all_masks) > 0:
        # If only one mask and no explicit selection, auto-select it
        if len(all_masks) == 1 and selected_idx is None:
            selected_idx = 0
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw reference box first (background)
        if reference_box is not None:
            x1, y1, x2, y2 = reference_box
            w, h = x2 - x1, y2 - y1
            ref_rect = plt.Rectangle(
                (x1, y1), w, h,
                linewidth=3,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                alpha=0.7
            )
            ax.add_patch(ref_rect)
        
        # Draw all masks
        has_any_mask = False
        for i, m in enumerate(all_masks):
            mask_binary = (m > 0.5)
            if mask_binary.sum() == 0:
                continue  # Skip empty masks but track if we have any
            
            has_any_mask = True
            
            # Selected masks in green, others in red (selected_idx can be a list)
            if isinstance(selected_idx, list):
                is_selected = i in selected_idx
            else:
                is_selected = (selected_idx is not None and i == selected_idx)
            color = np.array([0, 255, 0] if is_selected else [255, 0, 0])  # RGB
            alpha = 0.6 if is_selected else 0.3
            
            # Create colored overlay for this mask
            mask_colored = np.zeros_like(image)
            mask_colored[mask_binary] = color
            
            # Blend with image
            blended = image.copy()
            blended[mask_binary] = (alpha * mask_colored[mask_binary] + (1 - alpha) * image[mask_binary]).astype(np.uint8)
            ax.imshow(blended, alpha=1.0)
        
        # Add warning text if no masks were drawn
        if not has_any_mask:
            ax.text(0.5, 0.05, 'WARNING: Empty mask(s)', 
                   transform=ax.transAxes, fontsize=16, color='red',
                   ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        # Simple single mask overlay (original behavior)
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = 255  # Red channel
        
        alpha = 0.5
        mask_bool = (mask > 0.5)[:, :, np.newaxis]  # (H, W, 1)
        overlay = np.where(
            mask_bool,
            (alpha * mask_colored + (1 - alpha) * image).astype(np.uint8),
            overlay
        )
        
        # Draw reference box if provided
        if reference_box is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(overlay)
            
            x1, y1, x2, y2 = reference_box
            w, h = x2 - x1, y2 - y1
            ref_rect = plt.Rectangle(
                (x1, y1), w, h,
                linewidth=3,
                edgecolor='blue',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(ref_rect)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    images_root = args.images_root if args.images_root else str(config.get_path('renders') / 'train')
    text = args.text if args.text else config.config['segmentation']['text_prompt']
    output_dir = args.output_dir if args.output_dir else str(config.get_path('masks'))
    dino_thresh = args.dino_thresh if args.dino_thresh is not None else config.config['segmentation']['dino_threshold']
    
    # Selection parameters (CLI overrides config)
    dino_selection = args.dino_selection if args.dino_selection is not None else config.get('segmentation', 'dino_selection')
    sam_selection = args.sam_selection if args.sam_selection is not None else config.get('segmentation', 'sam_selection') 
    sam_thresh = args.sam_thresh if args.sam_thresh is not None else config.get('segmentation', 'sam_thresh', default=0.5)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    boxes_dir = output_dir / "boxes"
    masks_dir = output_dir / "sam_masks"
    overlays_dir = output_dir / "overlays"
    
    boxes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Text-to-Masks Generation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Images root: {images_root}")
    print(f"Text prompt: {text}")
    print(f"Output directory: {output_dir}")
    print(f"DINO threshold: {dino_thresh}")
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
    images_root_path = Path(images_root)
    image_files = sorted(images_root_path.glob("*.png")) + sorted(images_root_path.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {images_root_path}")
        sys.exit(1)
    
    print(f"Processing {len(image_files)} images...")
    print()
    
    # Parse reference box (CLI overrides config)
    ref_box_template = None
    ref_box_normalized = args.reference_box_normalized
    ref_overlap_thresh = args.reference_overlap_thresh
    
    if args.reference_box is not None:
        # Use CLI reference box
        try:
            ref_box_template = [float(x) for x in args.reference_box.strip('[]').split(',')]
            if len(ref_box_template) != 4:
                print(f"WARNING: reference_box must have 4 values, got {len(ref_box_template)}")
                ref_box_template = None
            else:
                print(f"Using CLI reference box: {ref_box_template} ({'normalized' if ref_box_normalized else 'pixels'})")
        except Exception as e:
            print(f"WARNING: Failed to parse reference_box: {e}")
            ref_box_template = None
    elif config.get('segmentation', 'reference_box') is not None:
        # Use config reference box
        try:
            ref_box_template = config.get('segmentation', 'reference_box')
            if len(ref_box_template) != 4:
                print(f"WARNING: config reference_box must have 4 values, got {len(ref_box_template)}")
                ref_box_template = None
            else:
                ref_box_normalized = config.get('segmentation', 'reference_box_normalized', default=False)
                ref_overlap_thresh = config.get('segmentation', 'reference_overlap_thresh', default=0.8)
                print(f"Using config reference box: {ref_box_template} ({'normalized' if ref_box_normalized else 'pixels'})")
        except Exception as e:
            print(f"WARNING: Failed to parse config reference_box: {e}")
            ref_box_template = None
    
    # Load per-image selection/manual box files if provided
    index_map = {}
    if args.select_index_file:
        try:
            with open(args.select_index_file, 'r') as f:
                index_map = json.load(f)
            print(f"Loaded select_index_file: {args.select_index_file} ({len(index_map)} entries)")
        except Exception as e:
            print(f"WARNING: Failed to load select_index_file {args.select_index_file}: {e}")

    manual_box_map = {}
    if args.manual_box_file:
        try:
            with open(args.manual_box_file, 'r') as f:
                manual_box_map = json.load(f)
            print(f"Loaded manual_box_file: {args.manual_box_file} ({len(manual_box_map)} entries)")
        except Exception as e:
            print(f"WARNING: Failed to load manual_box_file {args.manual_box_file}: {e}")

    # Process each image
    coverage_data = []
    successful = 0
    failed = 0
    
    for img_path in tqdm(image_files):
        img_name = img_path.stem
        
        try:
            # Load image
            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)
            h, w = image_np.shape[:2]
            
            # Detect objects with GroundingDINO
            image_np, boxes, logits, phrases = detect_objects(
                img_path,
                text,
                dino_model,
                args.box_thresh,
                dino_thresh,
            )
            
            # Keep original boxes for visualization
            original_boxes = boxes.copy()
            original_logits = logits.copy()
            original_phrases = phrases.copy()
            
            # Parse reference box for this image (convert to pixel coords if needed)
            ref_box = None
            if ref_box_template is not None:
                if ref_box_normalized:
                    ref_box = [ref_box_template[0] * w, ref_box_template[1] * h, 
                              ref_box_template[2] * w, ref_box_template[3] * h]
                else:
                    ref_box = ref_box_template.copy()
            
            # Apply reference box filtering if provided
            filtered_indices = []
            if ref_box is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    overlap = compute_box_overlap(box, ref_box)
                    if overlap >= ref_overlap_thresh:
                        filtered_indices.append(i)
                
                if len(filtered_indices) > 0:
                    boxes = boxes[filtered_indices]
                    logits = logits[filtered_indices]
                    phrases = [phrases[i] for i in filtered_indices]
                    print(f"Filtered {len(filtered_indices)}/{len(original_boxes)} boxes by reference_box overlap for {img_name}")
                else:
                    print(f"WARNING: No boxes pass reference_box overlap threshold for {img_name} - skipping image")
                    # Skip this image entirely - no processing, no mask generation
                    continue
            
            # Apply DINO selection or manual boxes
            selected_boxes = boxes
            selected_logits = logits
            selected_phrases = phrases
            dino_selected_indices = list(range(len(boxes)))  # Track which boxes were selected by DINO

            # Manual box mapping (overrides everything)
            if img_name in manual_box_map:
                try:
                    mb = manual_box_map[img_name]
                    if len(mb) == 4:
                        selected_boxes = np.array([mb], dtype=float)
                        selected_logits = np.array([1.0])
                        selected_phrases = ["manual_box"]
                        dino_selected_indices = [0]  # Only the manual box
                        print(f"Using manual box for {img_name}")
                    else:
                        print(f"WARNING: manual_box for {img_name} does not have 4 values: {mb}")
                except Exception as e:
                    print(f"WARNING: failed to apply manual_box for {img_name}: {e}")
            # Per-image index mapping
            elif img_name in index_map:
                try:
                    idx = int(index_map[img_name])
                    if 0 <= idx < len(boxes):
                        selected_boxes = np.array([boxes[idx]])
                        selected_logits = np.array([logits[idx]])
                        selected_phrases = [phrases[idx]]
                        dino_selected_indices = [idx]
                        print(f"Selecting index {idx} for {img_name} from select_index_file")
                    else:
                        print(f"WARNING: select_index {idx} out of range for {img_name}")
                except Exception as e:
                    print(f"WARNING: invalid index for {img_name} in select_index_file: {e}")
            elif args.select_index is not None and args.select_index >= 0:
                idx = int(args.select_index)
                if 0 <= idx < len(boxes):
                    selected_boxes = np.array([boxes[idx]])
                    selected_logits = np.array([logits[idx]])
                    selected_phrases = [phrases[idx]]
                    dino_selected_indices = [idx]
                    print(f"Selecting index {idx} for {img_name} from --select_index")
                else:
                    print(f"WARNING: --select_index {idx} out of range for {img_name}")
            # Apply dino_selection if no manual override
            elif dino_selection is not None and len(boxes) > 0:
                if dino_selection == "confidence":
                    # Pick box with highest DINO confidence
                    idx = np.argmax(logits)
                    selected_boxes = np.array([boxes[idx]])
                    selected_logits = np.array([logits[idx]])
                    selected_phrases = [phrases[idx]]
                    dino_selected_indices = [idx]
                    print(f"DINO selection: picked box {idx} with confidence {logits[idx]:.3f} for {img_name}")
                elif dino_selection == "largest":
                    # Pick largest box by area
                    box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                    idx = np.argmax(box_areas)
                    selected_boxes = np.array([boxes[idx]])
                    selected_logits = np.array([logits[idx]])
                    selected_phrases = [phrases[idx]]
                    dino_selected_indices = [idx]
                    print(f"DINO selection: picked largest box {idx} with area {box_areas[idx]:.1f} for {img_name}")
            
            if len(boxes) == 0:
                # No detection - skip this image (don't save empty mask)
                coverage_data.append({
                    "image": img_name,
                    "num_detections": 0,
                    "num_saved_masks": 0,
                    "max_score": 0.0,
                    "mask_coverage": 0.0,
                    "status": "no_detection"
                })
                failed += 1
                continue
            
            # Segment with SAM2 (use selected boxes, apply SAM threshold)
            masks_to_save, all_masks, sam_selected_indices, all_sam_scores = segment_with_sam2(
                image_np, selected_boxes, sam_predictor, 
                sam_selection=sam_selection,
                sam_thresh=sam_thresh
            )
            
            if len(masks_to_save) == 0:
                # No masks passed SAM2 confidence threshold - skip this image (don't save)
                coverage_data.append({
                    "image": img_name,
                    "num_detections": len(boxes),
                    "num_saved_masks": 0,
                    "max_score": float(logits.max()),
                    "mask_coverage": 0.0,
                    "status": f"no_mask_above_sam_thresh_{args.sam_thresh}"
                })
                failed += 1
                continue
            
            # Determine which boxes to highlight in visualization (green)
            # Map dino_selected_indices from filtered boxes back to original boxes
            box_highlight_indices = []
            if filtered_indices:
                for dino_idx in dino_selected_indices:
                    if dino_idx < len(filtered_indices):
                        box_highlight_indices.append(filtered_indices[dino_idx])
            else:
                box_highlight_indices = dino_selected_indices
            
            # Visualize ALL original boxes with reference box and selected boxes highlighted in green
            box_vis_path = boxes_dir / f"box_{img_name}.png"
            visualize_boxes(image_np, original_boxes, original_logits, original_phrases, box_vis_path, 
                          reference_box=ref_box, selected_indices=box_highlight_indices)
            
            # Save all masks that were selected by SAM
            h, w = image_np.shape[:2]
            total_coverage = 0.0
            
            for mask_idx, mask in enumerate(masks_to_save):
                # Validate mask dimensions
                if mask.shape != (h, w):
                    print(f"\nWARNING: Mask shape {mask.shape} doesn't match image shape ({h}, {w})")
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Calculate coverage
                mask_binary = (mask > 0.5).astype(np.uint8)
                coverage = (mask_binary.sum() / mask_binary.size) * 100
                total_coverage += coverage
                
                # Save mask (append _0, _1, _2 if multiple masks)
                suffix = f"_{mask_idx}" if len(masks_to_save) > 1 else ""
                mask_uint8 = (mask_binary * 255).astype(np.uint8)
                cv2.imwrite(str(masks_dir / f"mask_{img_name}{suffix}.png"), mask_uint8)
                np.save(masks_dir / f"mask_{img_name}{suffix}.npy", mask.astype(np.float32))
            
            # Create overlay with all masks + reference box
            overlay_path = overlays_dir / f"overlay_{img_name}.png"
            visualize_overlay(image_np, masks_to_save[0] if len(masks_to_save) == 1 else None, overlay_path, 
                            all_masks=all_masks, selected_idx=sam_selected_indices, reference_box=ref_box)
            
            # Record stats
            coverage_data.append({
                "image": img_name,
                "num_detections": len(original_boxes),
                "num_saved_masks": len(masks_to_save),
                "max_score": float(original_logits.max()),
                "mask_coverage": total_coverage,
                "status": "success"
            })
            successful += 1
            
        except Exception as e:
            print(f"\nERROR processing {img_name}: {e}")
            coverage_data.append({
                "image": img_name,
                "num_detections": 0,
                "num_saved_masks": 0,
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
        writer = csv.DictWriter(f, fieldnames=["image", "num_detections", "num_saved_masks", "max_score", "mask_coverage", "status"])
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
        "config_file": args.config,
        "images_root": str(images_root),
        "output_dir": str(output_dir),
        "parameters": {
            "text_prompt": text,
            "dino_thresh": dino_thresh,
            "box_thresh": args.box_thresh,
            "sam_model": args.sam_model,
            "seed": args.seed,
            "select_index": args.select_index,
            "select_index_file": args.select_index_file,
            "manual_box_file": args.manual_box_file,
            "reference_box": args.reference_box,
            "reference_box_normalized": ref_box_normalized,
            "reference_overlap_thresh": ref_overlap_thresh,
            "dino_selection": dino_selection,
            "sam_selection": sam_selection,
            "sam_thresh": args.sam_thresh,
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
    
    config.save_manifest("03_ground_text_to_masks", manifest)
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
