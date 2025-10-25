#!/usr/bin/env python3
"""
02_render_training_views.py - Pre-Edit Renders for a Round

Goal: Render the current 3D scene (from checkpoint) for the chosen views;
      these are the images sent to diffusion for editing.

Inputs:
  --ckpt: Path to the checkpoint file (e.g., outputs/garden/01_gs_base/ckpt_initial.pt)
  --data_root: Path to the dataset directory
  --output_dir: Directory to save renders (default: outputs/<dataset_name>/round_001/pre_edit/)
  --views: Which views to render ("all", "train", "val", or comma-separated indices)
  --seed: Random seed (default: 42)
  --factor: Downsample factor for dataset (default: 4)

Outputs (saved in output_dir):
  - pre_edit_view_000.png, pre_edit_view_001.png, ...
  - manifest.json: Which checkpoint, which poses, resolution, seed
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

# Add gsplat examples to path
sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))
from datasets.colmap import Dataset, Parser
from utils import set_random_seed

from gsplat.rendering import rasterization


def parse_args():
    parser = argparse.ArgumentParser(description="Render pre-edit training views")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for renders",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="train",
        help="Which views to render: 'all', 'train', 'val', or comma-separated indices",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Downsample factor for dataset",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Every N images is a test image",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=3,
        help="Degree of spherical harmonics",
    )
    parser.add_argument(
        "--round_num",
        type=int,
        default=1,
        help="Round number (for output directory naming)",
    )
    return parser.parse_args()


def load_checkpoint(ckpt_path, device="cuda"):
    """Load checkpoint and return splats dictionary."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract splats
    splats = checkpoint["splats"]
    
    # Convert to torch parameters
    splats_dict = {}
    for key, value in splats.items():
        if isinstance(value, torch.Tensor):
            splats_dict[key] = value.to(device)
        else:
            splats_dict[key] = torch.tensor(value).to(device)
    
    print(f"✓ Checkpoint loaded: {len(splats_dict['means'])} Gaussians")
    return splats_dict


def rasterize_splats(
    splats,
    camtoworlds,
    Ks,
    width,
    height,
    sh_degree=3,
    **kwargs
):
    """Rasterize Gaussian splats to images."""
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    
    # Concatenate SH coefficients
    num_sh_bases = (sh_degree + 1) ** 2
    sh0 = splats["sh0"]  # [N, 1, 3]
    shN = splats["shN"]  # [N, K-1, 3]
    colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
    colors = colors[:, :num_sh_bases, :]  # Truncate to current sh_degree
    
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        **kwargs,
    )
    
    return render_colors, render_alphas, info


def select_views(dataset, views_arg):
    """Select which views to render based on the views argument."""
    if views_arg == "all":
        return list(range(len(dataset)))
    elif views_arg == "train":
        # Return all training indices
        return list(range(len(dataset)))
    elif views_arg == "val":
        # This would need the validation dataset
        return []
    else:
        # Parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in views_arg.split(",")]
            return indices
        except:
            raise ValueError(f"Invalid views argument: {views_arg}")


def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    # Determine output directory
    if args.output_dir is None:
        dataset_name = Path(args.data_root).name
        args.output_dir = f"outputs/{dataset_name}/round_{args.round_num:03d}/pre_edit"
    
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("Render Pre-Edit Training Views")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Views: {args.views}")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    splats = load_checkpoint(args.ckpt, device=device)
    
    # Load dataset
    print("Loading dataset...")
    parser_obj = Parser(
        data_dir=args.data_root,
        factor=args.factor,
        normalize=True,
        test_every=args.test_every,
    )
    
    # Handle different view selections with subfolders
    splits_to_render = []
    if args.views == "all":
        splits_to_render = [("train", Dataset(parser_obj, split="train")),
                           ("val", Dataset(parser_obj, split="val"))]
        print(f"✓ Dataset loaded: {len(splits_to_render[0][1])} train + {len(splits_to_render[1][1])} val images")
    elif args.views == "val":
        splits_to_render = [("val", Dataset(parser_obj, split="val"))]
        print(f"✓ Dataset loaded: {len(splits_to_render[0][1])} validation images")
    else:  # "train" or specific indices
        splits_to_render = [("train", Dataset(parser_obj, split="train"))]
        print(f"✓ Dataset loaded: {len(splits_to_render[0][1])} training images")
    print()
    
    # Render each split
    all_rendered_paths = []
    all_view_info = []
    
    for split_name, dataset in splits_to_render:
        # Create subfolder for this split
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select views to render for this split
        if args.views in ["all", "train", "val"]:
            view_indices = list(range(len(dataset)))
        else:
            view_indices = select_views(dataset, args.views)
        
        print(f"Rendering {len(view_indices)} {split_name} views...")
        
        # Render each view
        for idx in tqdm.tqdm(view_indices, desc=f"{split_name}"):
            data = dataset[idx]
            
            # Get image dimensions from the image tensor
            image_tensor = data["image"]  # [H, W, 3]
            height, width = image_tensor.shape[:2]
            
            camtoworld = data["camtoworld"].unsqueeze(0).to(device)  # [1, 4, 4]
            K = data["K"].unsqueeze(0).to(device)  # [1, 3, 3]
            
            # Render
            with torch.no_grad():
                colors, _, _ = rasterize_splats(
                    splats=splats,
                    camtoworlds=camtoworld,
                    Ks=K,
                    width=width,
                    height=height,
                    sh_degree=args.sh_degree,
                )
            
            # Clamp and convert to numpy
            colors = torch.clamp(colors, 0.0, 1.0)
            image = colors.squeeze(0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            
            # Save render with split-specific path
            output_path = split_output_dir / f"view_{idx:03d}.png"
            imageio.imwrite(output_path, image)
            all_rendered_paths.append(str(output_path))
            
            # Store view info with split information
            all_view_info.append({
                "split": split_name,
                "index": idx,
                "width": width,
                "height": height,
                "camtoworld": data["camtoworld"].cpu().numpy().tolist(),
                "K": data["K"].cpu().numpy().tolist(),
            })
        
        print(f"✓ Rendered {len(view_indices)} {split_name} views to {split_output_dir}")
        print()
    
    print(f"✓ Total rendered: {len(all_rendered_paths)} views")
    print()
    print(f"✓ Total rendered: {len(all_rendered_paths)} views")
    print()
    
    # Create manifest
    manifest = {
        "module": "02_render_training_views",
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.ckpt,
        "data_root": args.data_root,
        "output_dir": str(output_dir),
        "parameters": {
            "views": args.views,
            "seed": args.seed,
            "factor": args.factor,
            "sh_degree": args.sh_degree,
            "round_num": args.round_num,
        },
        "num_views": len(all_rendered_paths),
        "splits_rendered": [split_name for split_name, _ in splits_to_render],
        "view_info": all_view_info,
        "outputs": {
            "renders": all_rendered_paths,
        },
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_path}")
    print()
    print("=" * 80)
    print("RENDERING COMPLETE")
    print("=" * 80)
    print(f"✓ {len(all_rendered_paths)} views rendered")
    print(f"✓ Outputs saved to: {output_dir}")
    for split_name, _ in splits_to_render:
        split_count = sum(1 for v in all_view_info if v["split"] == split_name)
        print(f"  - {split_name}/: {split_count} views")
    print()
    print("Next step: Use these renders for mask generation (03) or editing (05).")
    print("=" * 80)


if __name__ == "__main__":
    main()
