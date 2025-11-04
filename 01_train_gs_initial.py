#!/usr/bin/env python3
"""
01_train_gs_initial.py - Train/Load Initial 3DGS Scene

Goal: Produce a baseline 3D Gaussian scene and sanity-check rendering quality.

Inputs:
  --data_root: Path to the dataset (e.g., datasets/360_v2/garden/)
  --output_dir: Directory to save training outputs (default: outputs/<dataset_name>/01_gs_base/)
  --iters: Number of training iterations (default: 30000)
  --seed: Random seed (default: 42)
  --factor: Downsample factor for dataset (default: 4)
  --ckpt: Optional checkpoint to load instead of training from scratch

Outputs (saved in output_dir):
  - ckpt_initial.pt: Trained gsplat model checkpoint
  - renders/: Renders of training views
  - metrics.json: PSNR/SSIM metrics over training views
  - manifest.json: Config, training parameters, seed, gsplat version
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Add gsplat examples to path
sys.path.insert(0, str(Path(__file__).parent / "gsplat-src" / "examples"))

from datasets.colmap import Dataset, Parser
from fused_ssim import fused_ssim
from utils import knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

# Import project config (no name conflict now)
from project_utils.config import ProjectConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train initial 3D Gaussian Splatting scene")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the dataset directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for training results (overrides config)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of training iterations (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=None,
        help="Downsample factor for dataset (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=None,
        help="Every N images is a test image (overrides config)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to load (skip training)",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="Only render from checkpoint, skip training",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=None,
        help="Degree of spherical harmonics",
    )
    parser.add_argument(
        "--ssim_lambda",
        type=float,
        default=0.2,
        help="Weight for SSIM loss",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Export checkpoint as PLY file for visualization",
    )
    return parser.parse_args()


def create_splats_with_optimizers(
    parser_obj: Parser,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
):
    """Initialize Gaussian splats from COLMAP points."""
    # Use SfM points for initialization
    points = torch.from_numpy(parser_obj.points).float()
    rgbs = torch.from_numpy(parser_obj.points_rgb / 255.0).float()
    
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    
    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    
    # Color is SH coefficients
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    
    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr),
    ]
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    # Scale learning rate based on batch size
    BS = batch_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    
    return splats, optimizers


def rasterize_splats(
    splats,
    camtoworlds,
    Ks,
    width,
    height,
    sh_degree=3,
    near_plane=0.01,
    far_plane=1e10,
    **kwargs
):
    """Rasterize Gaussian splats to images."""
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    
    # Concatenate SH coefficients up to the specified degree
    # Only use coefficients up to (sh_degree + 1)^2
    num_sh_bases = (sh_degree + 1) ** 2
    sh0 = splats["sh0"]  # [N, 1, 3]
    shN = splats["shN"]  # [N, 15, 3] for sh_degree=3
    colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
    
    # Truncate to the current sh_degree
    colors = colors[:, :num_sh_bases, :]  # [N, num_sh_bases, 3]
    
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
        sh_degree=sh_degree,  # Important: tells rasterization to treat colors as SH coefficients
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        **kwargs,
    )
    
    return render_colors, render_alphas, info


def train(iters, sh_degree, splats, optimizers, trainloader, parser_obj, device, output_dir):
    """Train the 3D Gaussian Splatting model."""
    # Setup strategy for densification
    scene_scale = parser_obj.scene_scale * 1.1
    strategy = DefaultStrategy()
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    
    # Metrics
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    
    # Learning rate scheduler for means
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / iters)
    )
    
    # Training loop
    trainloader_iter = iter(trainloader)
    pbar = tqdm.tqdm(range(iters))
    
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)
        
        camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        height, width = pixels.shape[1:3]
        
        # SH degree schedule
        sh_degree_to_use = min(step // 1000, sh_degree)
        
        # Forward pass
        renders, alphas, info = rasterize_splats(
            splats=splats,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
        )
        colors = renders
        
        # Pre-backward step for strategy
        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )
        
        # Loss
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - args.ssim_lambda) + ssimloss * args.ssim_lambda
        
        loss.backward()
        
        # Optimize
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        # Post-backward step for strategy (densification)
        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=False,
        )
        
        # Progress bar
        pbar.set_description(
            f"loss={loss.item():.3f} | l1={l1loss.item():.3f} | "
            f"ssim={ssimloss.item():.3f} | GS={len(splats['means'])}"
        )
        
        # Logging
        if step % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/l1loss", l1loss.item(), step)
            writer.add_scalar("train/ssimloss", ssimloss.item(), step)
            writer.add_scalar("train/num_GS", len(splats["means"]), step)
    
    writer.close()
    return splats


@torch.no_grad()
def evaluate_and_render(sh_degree, splats, valset, device, output_dir):
    """Evaluate the model and render validation views."""
    render_dir = output_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    
    # Metrics
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=1
    )
    
    metrics = defaultdict(list)
    
    print("\nRendering validation views...")
    for i, data in enumerate(tqdm.tqdm(valloader)):
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        
        # Render
        colors, _, _ = rasterize_splats(
            splats=splats,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
        )
        
        colors = torch.clamp(colors, 0.0, 1.0)
        
        # Save render
        canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        imageio.imwrite(render_dir / f"view_{i:03d}.png", canvas)
        
        # Compute metrics
        pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        metrics["psnr"].append(psnr_fn(colors_p, pixels_p))
        metrics["ssim"].append(ssim_fn(colors_p, pixels_p))
        metrics["lpips"].append(lpips_fn(colors_p, pixels_p))
    
    # Aggregate metrics
    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats["num_GS"] = len(splats["means"])
    
    print(f"\nValidation Metrics:")
    print(f"  PSNR:  {stats['psnr']:.3f}")
    print(f"  SSIM:  {stats['ssim']:.4f}")
    print(f"  LPIPS: {stats['lpips']:.3f}")
    print(f"  Number of Gaussians: {stats['num_GS']}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    args = parse_args()
    
    # Load config
    config = ProjectConfig(args.config)
    
    # Override config with command-line arguments
    data_root = args.data_root if args.data_root else str(config.get_path('dataset_root'))
    output_dir = args.output_dir if args.output_dir else str(config.get_path('initial_training'))
    iters = args.iters if args.iters is not None else config.config['training']['iterations']
    seed = args.seed if args.seed is not None else config.config['dataset']['seed']
    factor = args.factor if args.factor is not None else config.config['dataset']['factor']
    test_every = args.test_every if args.test_every is not None else config.config['dataset']['test_every']
    sh_degree = args.sh_degree if hasattr(args, 'sh_degree') and args.sh_degree is not None else config.config['training']['sh_degree']
    
    set_random_seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Initial 3DGS Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {iters}")
    print(f"Seed: {seed}")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    print("Loading dataset...")
    parser_obj = Parser(
        data_dir=data_root,
        factor=factor,
        normalize=True,
        test_every=test_every,
    )
    trainset = Dataset(parser_obj, split="train")
    valset = Dataset(parser_obj, split="val")
    print(f"✓ Dataset loaded: {len(trainset)} train, {len(valset)} val images")
    print()
    
    # Check for existing checkpoint
    ckpt_path = output_dir / "ckpt_initial.pt"
    
    if args.ckpt:
        # Load specified checkpoint
        print(f"Loading checkpoint from: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        
        # Recreate splats structure
        splats, optimizers = create_splats_with_optimizers(
            parser_obj,
            scene_scale=parser_obj.scene_scale * 1.1,
            sh_degree=sh_degree,
            batch_size=args.batch_size,
            device=device,
        )
        splats.load_state_dict(checkpoint["splats"])
        print(f"✓ Checkpoint loaded: {len(splats['means'])} Gaussians")
        
    elif ckpt_path.exists() and not args.render_only:
        print(f"Found existing checkpoint: {ckpt_path}")
        response = input("Use existing checkpoint? (y/n): ")
        if response.lower() == 'y':
            checkpoint = torch.load(ckpt_path, map_location=device)
            splats, optimizers = create_splats_with_optimizers(
                parser_obj,
                scene_scale=parser_obj.scene_scale * 1.1,
                sh_degree=sh_degree,
                batch_size=args.batch_size,
                device=device,
            )
            splats.load_state_dict(checkpoint["splats"])
            print(f"✓ Checkpoint loaded: {len(splats['means'])} Gaussians")
        else:
            # Train from scratch
            splats, optimizers = create_splats_with_optimizers(
                parser_obj,
                scene_scale=parser_obj.scene_scale * 1.1,
                sh_degree=args.sh_degree,
                batch_size=args.batch_size,
                device=device,
            )
            print(f"✓ Initialized: {len(splats['means'])} Gaussians from SfM points")
            print()
            
            # Train
            print("Starting training...")
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
            
            train(iters, sh_degree, splats, optimizers, trainloader, parser_obj, device, output_dir)
            
            # Save checkpoint
            print(f"\nSaving checkpoint to: {ckpt_path}")
            torch.save({"step": iters, "splats": splats.state_dict()}, ckpt_path)
    else:
        # Train from scratch
        splats, optimizers = create_splats_with_optimizers(
            parser_obj,
            scene_scale=parser_obj.scene_scale * 1.1,
            sh_degree=sh_degree,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"✓ Initialized: {len(splats['means'])} Gaussians from SfM points")
        print()
        
        # Train
        print("Starting training...")
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        
        train(iters, sh_degree, splats, optimizers, trainloader, parser_obj, device, output_dir)
        
    # Save checkpoint
    print(f"\nSaving checkpoint to: {ckpt_path}")
    torch.save({"step": iters, "splats": splats.state_dict()}, ckpt_path)
    
    # Export PLY if requested
    if args.save_ply:
        ply_path = output_dir / "scene_initial.ply"
        print(f"Exporting PLY to: {ply_path}")
        export_splats(
            means=splats["means"],
            scales=torch.exp(splats["scales"]),
            quats=splats["quats"],
            opacities=torch.sigmoid(splats["opacities"]),
            sh0=splats["sh0"],
            shN=splats["shN"],
            format="ply",
            save_to=str(ply_path),
        )
        print(f"✓ PLY file saved")
    
    # Evaluate
    print("\nEvaluating model...")
    stats = evaluate_and_render(sh_degree, splats, valset, device, output_dir)
    
    # Create manifest
    manifest = {
        "module": "01_train_gs_initial",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "data_root": data_root,
        "output_dir": str(output_dir),
        "parameters": {
            "iters": iters,
            "seed": seed,
            "factor": factor,
            "batch_size": args.batch_size,
            "sh_degree": sh_degree,
            "ssim_lambda": args.ssim_lambda,
        },
        "metrics": stats,
        "checkpoint": str(ckpt_path),
    }
    
    config.save_manifest("01_train_gs_initial", manifest)
    
    print(f"\nManifest saved to: {manifest_path}")
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ Checkpoint saved: {ckpt_path}")
    print(f"✓ Renders saved: {output_dir / 'renders'}")
    print(f"✓ Metrics: PSNR={stats['psnr']:.3f}, SSIM={stats['ssim']:.4f}")
    print()
    print("Next step: Use this checkpoint for rendering and editing.")
    print("=" * 80)


if __name__ == "__main__":
    main()
