"""
Phase 1: Training - Dataset Validation + Initial 3DGS Training

Directly implements training without legacy scripts.

This phase:
1. Validates dataset paths and structure  
2. Loads COLMAP data
3. Trains initial 3D Gaussian Splatting model
4. Saves checkpoint and metrics

Outputs:
- phase1_training/
  ├── 00_dataset_validation/
  │   ├── thumbs/              # Dataset thumbnails
  │   └── summary.txt          # Dataset statistics
  ├── 01_initial_training/
  │   ├── ckpt_initial.pt      # Trained GS model
  │   ├── renders/
  │   │   └── train/           # Training renders
  │   ├── checkpoints/         # Intermediate checkpoints
  │   └── metrics/             # Training metrics
  └── manifest.json            # Phase metadata
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add gsplat to path
gsplat_path = str(Path(__file__).parent.parent / "gsplat-src" / "examples")
if gsplat_path not in sys.path:
    sys.path.insert(0, gsplat_path)

from .base import BasePhase


class Phase1Training(BasePhase):
    """Phase 1: Dataset validation and initial GS training."""
    
    def __init__(self, config):
        super().__init__(config, phase_name="training", phase_number=1)
        
        # Get training config - use correct syntax: config.get(key1, key2, ..., default=val)
        self.dataset_root = Path(config.get("paths", "dataset_root"))
        self.factor = config.get("dataset", "factor", default=4)
        self.test_every = config.get("dataset", "test_every", default=8)
        self.seed = config.get("dataset", "seed", default=42)
        
        self.iterations = config.get("training", "iterations", default=30000)
        self.sh_degree = config.get("training", "sh_degree", default=3)
        self.eval_steps = config.get("training", "eval_steps", default=[7000, 30000])
        self.save_steps = config.get("training", "save_steps", default=[7000, 30000])
        
        # Create subdirectories matching old numbered structure
        (self.phase_dir / "00_dataset_validation" / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "01_initial_training" / "renders" / "train").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "01_initial_training" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "01_initial_training" / "metrics").mkdir(parents=True, exist_ok=True)
        
        self.metadata.update({
            "dataset_root": str(self.dataset_root),
            "factor": self.factor,
            "iterations": self.iterations,
            "sh_degree": self.sh_degree,
        })
    
    def validate_inputs(self) -> bool:
        """Validate that dataset exists."""
        if not self.dataset_root.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]Dataset not found:[/red] {self.dataset_root}")
            return False
        
        # Check for required COLMAP files
        sparse_dir = self.dataset_root / "sparse" / "0"
        if not sparse_dir.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]COLMAP sparse reconstruction not found:[/red] {sparse_dir}")
            return False
        
        return True
    
    def is_complete(self) -> bool:
        """Check if Phase 1 is already complete."""
        training_dir = self.phase_dir / "01_initial_training"
        ckpt_path = training_dir / "ckpt_initial.pt"
        metrics_path = training_dir / "metrics.json"
        render_dir = training_dir / "renders"
        
        # Check all required outputs exist
        return (ckpt_path.exists() and 
                metrics_path.exists() and 
                render_dir.exists() and 
                len(list(render_dir.glob("*.png"))) > 0)
    
    def execute(self) -> Dict[str, Any]:
        """Execute Phase 1: Dataset validation + training."""
        from rich.console import Console
        import torch
        import numpy as np
        from datasets.colmap import Dataset, Parser
        from utils import set_random_seed, rgb_to_sh, knn
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        import imageio.v2 as imageio
        from tqdm import tqdm
        
        console = Console()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        set_random_seed(self.seed)
        
        console.print(f"[bold cyan]Loading dataset:[/bold cyan] {self.dataset_root}")
        
        # Load COLMAP data using Parser (like the old code does)
        parser_obj = Parser(
            data_dir=str(self.dataset_root),
            factor=self.factor,
            normalize=True,
            test_every=self.test_every,
        )
        
        # Create dataset from parser
        trainset = Dataset(parser_obj, split="train")
        
        console.print(f"✓ Loaded {len(trainset)} training images\n")
        
        # Save dataset validation outputs
        console.print("[bold cyan]Saving dataset validation...[/bold cyan]")
        self._save_dataset_validation(parser_obj, trainset, console)
        console.print("✓ Dataset validation saved\n")
        
        # Initialize Gaussians from COLMAP points (using ParameterDict like legacy code)
        console.print("[bold cyan]Initializing 3D Gaussians...[/bold cyan]")
        points = torch.from_numpy(parser_obj.points).float()
        rgbs = torch.from_numpy(parser_obj.points_rgb / 255.0).float()
        
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * 0.01).unsqueeze(-1).repeat(1, 3)
        
        N = points.shape[0]
        quats = torch.rand((N, 4))
        opacities = torch.logit(torch.full((N,), 0.1))
        
        # Initialize SH coefficients
        colors = torch.zeros((N, (self.sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        
        # Create ParameterDict like legacy code
        params = [
            ("means", torch.nn.Parameter(points), 1.6e-4),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
            ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
            ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
        ]
        
        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
        optimizers = {
            name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}])
            for name, _, lr in params
        }
        
        console.print(f"✓ Initialized {N} Gaussians\n")
        
        # Training loop
        console.print(f"[bold cyan]Training for {self.iterations} iterations...[/bold cyan]")
        
        # Create dataloader
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        
        # Setup strategy with scene_scale (like legacy code)
        scene_scale = parser_obj.scene_scale * 1.1
        strategy = DefaultStrategy()
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        
        trainloader_iter = iter(trainloader)
        pbar = tqdm(range(self.iterations), desc="Training")
        
        for step in pbar:
            # Get next batch
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            
            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]
            
            # Render (using splats dict like legacy code)
            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"])
            
            # Concatenate SH coefficients
            sh0 = splats["sh0"]  # [N, 1, 3]
            shN = splats["shN"]  # [N, K-1, 3]
            colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]
            
            renders, alphas, info = rasterization(
                means=means,
                quats=quats / quats.norm(dim=-1, keepdim=True),
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.sh_degree,
                packed=False,
                absgrad=True,
                sparse_grad=False,
            )
            
            # Pre-backward step for strategy
            strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )
            
            # Loss
            loss = torch.nn.functional.l1_loss(renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss.backward()
            
            # Optimize
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update progress
            if step % 100 == 0:
                psnr = psnr_metric(renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{psnr.item():.2f}"})
            
            # Densification (post-backward step)
            if step < 15000 and step % 100 == 0:
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=False,
                )
        
        console.print("\n✓ Training completed\n")
        
        # Save checkpoint
        console.print("[bold cyan]Saving checkpoint...[/bold cyan]")
        training_dir = self.phase_dir / "01_initial_training"
        ckpt_path = training_dir / "ckpt_initial.pt"
        torch.save({"step": self.iterations, "splats": splats.state_dict()}, ckpt_path)
        console.print(f"✓ Checkpoint saved: {ckpt_path}\n")
        
        # Render validation views and compute metrics
        console.print("[bold cyan]Rendering validation views...[/bold cyan]")
        metrics_dict = self._render_and_evaluate(splats, parser_obj, device, console)
        console.print(f"✓ Validation complete: PSNR={metrics_dict['psnr']:.2f}, SSIM={metrics_dict['ssim']:.4f}\n")
        
        results = {
            "checkpoint": str(ckpt_path),
            "dataset_root": str(self.dataset_root),
            "num_iterations": self.iterations,
            "num_gaussians": len(splats["means"]),
            "metrics": metrics_dict,
        }
        }
        
        return results
    
    def _save_dataset_validation(self, parser_obj, trainset, console):
        """Save dataset thumbnails and summary like legacy code."""
        import imageio.v2 as imageio
        
        val_dir = self.phase_dir / "00_dataset_validation"
        thumbs_dir = val_dir / "thumbs"
        
        # Save thumbnails (sample 6 images)
        num_thumbs = min(6, len(trainset))
        indices = np.linspace(0, len(trainset)-1, num_thumbs, dtype=int)
        
        for i, idx in enumerate(indices):
            data = trainset[idx]
            image = data["image"]
            imageio.imwrite(thumbs_dir / f"thumb_{i:02d}.png", image)
        
        console.print(f"  Saved {num_thumbs} thumbnails")
        
        # Write summary
        summary_path = val_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Dataset Validation Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset root: {self.dataset_root}\n")
            f.write(f"Downsample factor: {self.factor}\n")
            f.write(f"Test every: {self.test_every}\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"  Total images: {len(parser_obj.image_names)}\n")
            f.write(f"  Training images: {len(trainset)}\n")
            f.write(f"  COLMAP points: {len(parser_obj.points)}\n")
            f.write(f"  Cameras: {len(parser_obj.Ks_dict)}\n\n")
            
            # Camera intrinsics
            f.write("Camera Intrinsics:\n")
            for cam_id, K in parser_obj.Ks_dict.items():
                f.write(f"  Camera {cam_id}:\n")
                f.write(f"    fx={K[0,0]:.2f}, fy={K[1,1]:.2f}\n")
                f.write(f"    cx={K[0,2]:.2f}, cy={K[1,2]:.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        console.print(f"  Summary written to {summary_path.name}")
    
    def _render_and_evaluate(self, splats, parser_obj, device, console):
        """Render validation views and compute metrics like legacy code."""
        import torch
        import imageio.v2 as imageio
        from collections import defaultdict
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from gsplat.rendering import rasterization
        from tqdm import tqdm
        
        training_dir = self.phase_dir / "01_initial_training"
        render_dir = training_dir / "renders"
        render_dir.mkdir(exist_ok=True)
        
        # Setup validation dataset
        from datasets.colmap import Dataset
        valset = Dataset(parser_obj, split="val")
        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
        
        # Metrics
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        
        metrics = defaultdict(list)
        
        for i, data in enumerate(tqdm(valloader, desc="Rendering")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            
            # Render
            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"])
            colors = torch.cat([splats["sh0"], splats["shN"]], dim=-1)
            
            renders, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="classic",
                sh_degree=self.sh_degree,
            )
            
            renders = torch.clamp(renders, 0.0, 1.0)
            
            # Save side-by-side comparison (GT | Rendered)
            canvas = torch.cat([pixels, renders], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(render_dir / f"view_{i:03d}.png", canvas)
            
            # Compute metrics
            pixels_p = pixels.permute(0, 3, 1, 2)
            renders_p = renders.permute(0, 3, 1, 2)
            metrics["psnr"].append(psnr_fn(renders_p, pixels_p))
            metrics["ssim"].append(ssim_fn(renders_p, pixels_p))
            metrics["lpips"].append(lpips_fn(renders_p, pixels_p))
        
        # Aggregate metrics
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats["num_GS"] = len(splats["means"])
        
        # Save metrics to JSON
        import json
        metrics_path = training_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
