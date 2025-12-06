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
  ├── ckpt_initial.pt          # Trained GS model
  ├── metrics.json             # Training metrics
  └── renders/                 # Sample renders
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
        
        # Get training config
        self.dataset_root = Path(config.get("paths", "dataset_root"))
        self.factor = config.get("dataset", "factor", 4)
        self.test_every = config.get("dataset", "test_every", 8)
        self.seed = config.get("dataset", "seed", 42)
        
        self.iterations = config.get("training", "iterations", 30000)
        self.sh_degree = config.get("training", "sh_degree", 3)
        self.eval_steps = config.get("training", "eval_steps", [7000, 30000])
        self.save_steps = config.get("training", "save_steps", [7000, 30000])
        
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
    
    def execute(self) -> Dict[str, Any]:
        """Execute Phase 1: Dataset validation + training."""
        from rich.console import Console
        import torch
        import numpy as np
        from datasets.colmap import Dataset
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
        trainset = Dataset(
            self.dataset_root,
            split="train",
            patch_size=None,
            load_depths=False,
            factor=self.factor,
        )
        
        console.print(f"✓ Loaded {len(trainset)} training images\n")
        
        # Initialize Gaussians
        console.print("[bold cyan]Initializing 3D Gaussians...[/bold cyan]")
        points = torch.from_numpy(trainset.points).float()
        rgbs = torch.from_numpy(trainset.points_rgb / 255.0).float()
        
        N = points.shape[0]
        means = torch.nn.Parameter(points.to(device))
        scales = torch.nn.Parameter(torch.log(torch.ones(N, 3, device=device) * 0.01))
        quats = torch.nn.Parameter(torch.rand(N, 4, device=device))
        opacities = torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device)))
        
        # Initialize SH from RGB
        sh_coeffs = torch.zeros(N, (self.sh_degree + 1) ** 2, 3, device=device)
        sh_coeffs[:, 0, :] = rgb_to_sh(rgbs.to(device))
        sh_coeffs = torch.nn.Parameter(sh_coeffs)
        
        params = [means, scales, quats, opacities, sh_coeffs]
        optimizers = [torch.optim.Adam([p], lr=1.6e-4 if i == 0 else 5e-3) for i, p in enumerate(params)]
        
        console.print(f"✓ Initialized {N} Gaussians\n")
        
        # Training loop
        console.print(f"[bold cyan]Training for {self.iterations} iterations...[/bold cyan]")
        strategy = DefaultStrategy()
        psnr_metric = PeakSignalNoiseRatio().to(device)
        
        pbar = tqdm(range(self.iterations), desc="Training")
        for step in pbar:
            # Sample random image
            idx = np.random.randint(len(trainset))
            data = trainset[idx]
            camtoworlds = torch.from_numpy(data["camtoworld"]).float().to(device)
            Ks = torch.from_numpy(data["K"]).float().to(device)
            pixels = torch.from_numpy(data["image"]).float().to(device) / 255.0
            
            # Render
            renders, alphas, info = rasterization(
                means=means,
                quats=quats / quats.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=sh_coeffs,
                viewmats=torch.linalg.inv(camtoworlds)[None],
                Ks=Ks[None],
                width=pixels.shape[1],
                height=pixels.shape[0],
                sh_degree=self.sh_degree,
            )
            
            # Loss
            loss = torch.nn.functional.l1_loss(renders[0].permute(1, 2, 0), pixels)
            
            # Backward
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()
            
            # Update progress
            if step % 100 == 0:
                psnr = psnr_metric(renders[0].permute(1, 2, 0), pixels)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{psnr.item():.2f}"})
            
            # Densification
            if step < 15000 and step % 100 == 0:
                strategy.step_post_backward(
                    params={"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh_coeffs[:, 0, :], "shN": sh_coeffs[:, 1:, :]},
                    optimizers=dict(zip(["means", "scales", "quats", "opacities", "sh0", "shN"], optimizers)),
                    state=strategy.state,
                    step=step,
                    info=info,
                )
        
        console.print("\n✓ Training completed\n")
        
        # Save checkpoint
        console.print("[bold cyan]Saving checkpoint...[/bold cyan]")
        ckpt_path = self.phase_dir / "ckpt_initial.pt"
        torch.save({
            "means": means.detach().cpu(),
            "scales": scales.detach().cpu(),
            "quats": quats.detach().cpu(),
            "opacities": opacities.detach().cpu(),
            "sh_coeffs": sh_coeffs.detach().cpu(),
            "sh_degree": self.sh_degree,
        }, ckpt_path)
        console.print(f"✓ Checkpoint saved: {ckpt_path}\n")
        
        results = {
            "checkpoint": str(ckpt_path),
            "dataset_root": str(self.dataset_root),
            "num_iterations": self.iterations,
            "num_gaussians": N,
        }
        
        return results
