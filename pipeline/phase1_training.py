"""
Phase 1: Training - Dataset Validation + Initial 3DGS Training

Replicates the exact functionality of legacy 00_check_dataset.py and 01_train_gs_initial.py
but with class-based structure and new config format.

This phase:
1. Validates dataset (creates thumbnails, summary) - like 00_check_dataset.py
2. Trains initial 3D Gaussian Splatting model - like 01_train_gs_initial.py  
3. Saves checkpoint and metrics
4. Renders validation views with GT|Render side-by-side comparison

Outputs (matching legacy structure):
- phase1_training/
  ├── 00_dataset_validation/
  │   ├── thumbs/              # Dataset thumbnails
  │   ├── summary.txt          # Dataset statistics
  │   ├── manifest.json        # Validation metadata
  │   └── README.md
  ├── 01_initial_training/
  │   ├── ckpt_initial.pt      # Trained GS checkpoint
  │   ├── renders/             # GT|Render comparisons
  │   ├── metrics.json         # PSNR/SSIM/LPIPS
  │   ├── manifest.json        # Training metadata
  │   └── README.md
  └── logs/                    # Tensorboard logs
"""

import sys
from pathlib import Path
from typing import Any, Dict
import math

# Add gsplat to path
gsplat_path = str(Path(__file__).parent.parent / "gsplat-src" / "examples")
if gsplat_path not in sys.path:
    sys.path.insert(0, gsplat_path)

from .base import BasePhase


class Phase1Training(BasePhase):
    """Phase 1: Dataset validation and initial GS training (legacy-compatible)."""
    
    def __init__(self, config):
        super().__init__(config, phase_name="training", phase_number=1)
        
        # Dataset config
        self.dataset_root = Path(config.get("paths", "dataset_root"))
        self.factor = config.get("dataset", "factor", default=4)
        self.test_every = config.get("dataset", "test_every", default=8)
        self.seed = config.get("dataset", "seed", default=42)
        
        # Training config
        self.iterations = config.get("training", "iterations", default=30000)
        self.sh_degree = config.get("training", "sh_degree", default=3)
        self.eval_steps = config.get("training", "eval_steps", default=[7000, 30000])
        self.save_steps = config.get("training", "save_steps", default=[7000, 30000])
        
        # Hardware config
        self.device = config.get("hardware", "device", default="cuda")
        self.num_workers = config.get("hardware", "num_workers", default=4)
        
        # Create subdirectories matching legacy structure
        self.dataset_check_dir = self.phase_dir / "00_dataset_validation"
        self.training_dir = self.phase_dir / "01_initial_training"
        
        (self.dataset_check_dir / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.training_dir / "renders").mkdir(parents=True, exist_ok=True)
        (self.training_dir / "tb").mkdir(parents=True, exist_ok=True)  # Tensorboard
        
        self.metadata.update({
            "dataset_root": str(self.dataset_root),
            "factor": self.factor,
            "iterations": self.iterations,
            "sh_degree": self.sh_degree,
        })
    
    def validate_inputs(self) -> bool:
        """Validate that dataset exists (like legacy 00_check_dataset.py)."""
        if not self.dataset_root.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]❌ Dataset not found:[/red] {self.dataset_root}")
            return False
        
        # Check for required COLMAP files
        sparse_dir = self.dataset_root / "sparse" / "0"
        if not sparse_dir.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]❌ COLMAP sparse reconstruction not found:[/red] {sparse_dir}")
            return False
        
        return True
    
    def is_complete(self) -> bool:
        """Check if Phase 1 is already complete."""
        ckpt_path = self.training_dir / "ckpt_initial.pt"
        metrics_path = self.training_dir / "metrics.json"
        render_dir = self.training_dir / "renders"
        
        # Check all required outputs exist
        return (ckpt_path.exists() and 
                metrics_path.exists() and 
                render_dir.exists() and 
                len(list(render_dir.glob("*.png"))) > 0)
    
    def execute(self) -> Dict[str, Any]:
        """Execute Phase 1: Dataset validation + training (legacy-compatible)."""
        from rich.console import Console
        from datetime import datetime
        import torch
        import torch.nn.functional as F
        import numpy as np
        import json
        
        from datasets.colmap import Dataset, Parser
        from utils import set_random_seed, rgb_to_sh, knn
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
        from torchmetrics.functional import structural_similarity_index_measure
        from torch.utils.tensorboard import SummaryWriter
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        import imageio.v2 as imageio
        from tqdm import tqdm
        
        console = Console()
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        set_random_seed(self.seed)
        
        # ====================================================================
        # STEP 1: Dataset Validation (like 00_check_dataset.py)
        # ====================================================================
        console.print("=" * 80)
        console.print("Dataset Validation")
        console.print("=" * 80)
        console.print(f"Data root: {self.dataset_root}")
        console.print(f"Output directory: {self.dataset_check_dir}")
        console.print(f"Downsample factor: {self.factor}")
        console.print()
        
        console.print("Loading dataset...")
        parser_obj = Parser(
            data_dir=str(self.dataset_root),
            factor=self.factor,
            normalize=True,
            test_every=self.test_every,
        )
        trainset = Dataset(parser_obj, split="train")
        valset = Dataset(parser_obj, split="val")
        
        console.print(f"✓ Successfully loaded dataset")
        console.print(f"  - {len(parser_obj.image_names)} total images")
        console.print(f"  - {len(trainset)} train images")
        console.print(f"  - {len(valset)} val images")
        console.print(f"  - {len(parser_obj.points)} 3D points")
        console.print()
        
        # Compute dataset statistics (like legacy)
        console.print("Computing dataset statistics...")
        stats = self._compute_dataset_stats(parser_obj)
        console.print("✓ Statistics computed")
        console.print()
        
        # Save thumbnails (like legacy)
        console.print("Saving 6 thumbnail images...")
        self._save_thumbnails(parser_obj, n_thumbs=6)
        console.print("✓ Thumbnails saved")
        console.print()
        
        # Write summary (like legacy)
        console.print("Writing summary...")
        self._write_summary(stats, parser_obj)
        console.print("✓ Summary written")
        console.print()
        
        # Save validation manifest (like legacy)
        validation_manifest = {
            "module": "00_check_dataset",
            "timestamp": datetime.now().isoformat(),
            "data_root": str(self.dataset_root),
            "output_dir": str(self.dataset_check_dir),
            "parameters": {
                "factor": self.factor,
                "test_every": self.test_every,
            },
            "statistics": stats,
        }
        with open(self.dataset_check_dir / "manifest.json", 'w') as f:
            json.dump(validation_manifest, f, indent=2)
        
        console.print("=" * 80)
        console.print("VALIDATION COMPLETE")
        console.print("=" * 80)
        console.print()
        
        # ====================================================================
        # STEP 2: Initial 3DGS Training (like 01_train_gs_initial.py)
        # ====================================================================
        console.print("=" * 80)
        console.print("Initial 3DGS Training")
        console.print("=" * 80)
        console.print(f"Data root: {self.dataset_root}")
        console.print(f"Output directory: {self.training_dir}")
        console.print(f"Iterations: {self.iterations}")
        console.print(f"Seed: {self.seed}")
        console.print()
        
        # Initialize Gaussians from COLMAP points (like legacy create_splats_with_optimizers)
        console.print("Initializing 3D Gaussians from COLMAP points...")
        splats, optimizers = self._create_splats_with_optimizers(
            parser_obj, device, batch_size=1, scene_scale=parser_obj.scene_scale * 1.1
        )
        console.print(f"✓ Initialized: {len(splats['means'])} Gaussians from SfM points")
        console.print()
        
        # Training (like legacy train function)
        console.print("Starting training...")
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        
        self._train(splats, optimizers, trainloader, parser_obj, device)
        console.print()
        
        # Save checkpoint (like legacy)
        ckpt_path = self.training_dir / "ckpt_initial.pt"
        console.print(f"Saving checkpoint to: {ckpt_path}")
        torch.save({"step": self.iterations, "splats": splats.state_dict()}, ckpt_path)
        console.print(f"✓ Checkpoint saved")
        console.print()
        
        # Evaluate and render (like legacy evaluate_and_render)
        console.print("Evaluating model...")
        metrics_dict = self._evaluate_and_render(splats, valset, device)
        
        console.print(f"\nValidation Metrics:")
        console.print(f"  PSNR:  {metrics_dict['psnr']:.3f}")
        console.print(f"  SSIM:  {metrics_dict['ssim']:.4f}")
        console.print(f"  LPIPS: {metrics_dict['lpips']:.3f}")
        console.print(f"  Number of Gaussians: {metrics_dict['num_GS']}")
        console.print()
        
        # Save training manifest (like legacy)
        training_manifest = {
            "module": "01_train_gs_initial",
            "timestamp": datetime.now().isoformat(),
            "data_root": str(self.dataset_root),
            "output_dir": str(self.training_dir),
            "parameters": {
                "iters": self.iterations,
                "seed": self.seed,
                "factor": self.factor,
                "batch_size": 1,
                "sh_degree": self.sh_degree,
                "ssim_lambda": 0.2,
            },
            "metrics": metrics_dict,
            "checkpoint": str(ckpt_path),
        }
        with open(self.training_dir / "manifest.json", 'w') as f:
            json.dump(training_manifest, f, indent=2)
        
        console.print("=" * 80)
        console.print("TRAINING COMPLETE")
        console.print("=" * 80)
        console.print(f"✓ Checkpoint saved: {ckpt_path}")
        console.print(f"✓ Renders saved: {self.training_dir / 'renders'}")
        console.print(f"✓ Metrics: PSNR={metrics_dict['psnr']:.3f}, SSIM={metrics_dict['ssim']:.4f}")
        console.print()
        console.print("Next step: Use this checkpoint for rendering and editing.")
        console.print("=" * 80)
        
        results = {
            "checkpoint": str(ckpt_path),
            "dataset_root": str(self.dataset_root),
            "num_iterations": self.iterations,
            "num_gaussians": metrics_dict['num_GS'],
            "metrics": metrics_dict,
        }
        
        return results
    
    def _compute_dataset_stats(self, parser_obj):
        """Compute dataset statistics (like legacy 00_check_dataset.py)."""
        import numpy as np
        
        stats = {}
        
        # Basic counts
        stats["num_images"] = len(parser_obj.image_names)
        stats["num_points"] = len(parser_obj.points)
        stats["num_cameras"] = len(set(parser_obj.camera_ids))
        
        # Camera intrinsics statistics
        focal_lengths = []
        image_sizes = []
        for camera_id in parser_obj.Ks_dict.keys():
            K = parser_obj.Ks_dict[camera_id]
            fx, fy = K[0, 0], K[1, 1]
            focal_lengths.append([fx, fy])
            width, height = parser_obj.imsize_dict[camera_id]
            image_sizes.append([width, height])
        
        focal_lengths = np.array(focal_lengths)
        image_sizes = np.array(image_sizes)
        
        stats["focal_length_min"] = focal_lengths.min(axis=0).tolist()
        stats["focal_length_max"] = focal_lengths.max(axis=0).tolist()
        stats["focal_length_mean"] = focal_lengths.mean(axis=0).tolist()
        stats["image_sizes"] = image_sizes.tolist()
        
        # Camera pose statistics
        positions = parser_obj.camtoworlds[:, :3, 3]  # Extract positions
        stats["camera_positions_min"] = positions.min(axis=0).tolist()
        stats["camera_positions_max"] = positions.max(axis=0).tolist()
        stats["camera_positions_mean"] = positions.mean(axis=0).tolist()
        stats["camera_positions_std"] = positions.std(axis=0).tolist()
        
        # Point cloud statistics
        stats["points_min"] = parser_obj.points.min(axis=0).tolist()
        stats["points_max"] = parser_obj.points.max(axis=0).tolist()
        stats["points_mean"] = parser_obj.points.mean(axis=0).tolist()
        stats["points_std"] = parser_obj.points.std(axis=0).tolist()
        
        if hasattr(parser_obj, "points_err"):
            stats["points_error_mean"] = float(parser_obj.points_err.mean())
            stats["points_error_std"] = float(parser_obj.points_err.std())
        
        return stats
    
    def _save_thumbnails(self, parser_obj, n_thumbs=6):
        """Save thumbnail images (like legacy 00_check_dataset.py)."""
        import imageio.v2 as imageio
        import numpy as np
        
        thumbs_dir = self.dataset_check_dir / "thumbs"
        num_images = len(parser_obj.image_paths)
        indices = np.linspace(0, num_images - 1, n_thumbs, dtype=int)
        
        for idx in indices:
            image_path = parser_obj.image_paths[idx]
            image_name = parser_obj.image_names[idx]
            
            # Load image
            image = imageio.imread(image_path)[..., :3]
            
            # Save thumbnail
            thumb_name = f"thumb_{idx:03d}_{Path(image_name).stem}.png"
            thumb_path = thumbs_dir / thumb_name
            imageio.imwrite(thumb_path, image)
    
    def _write_summary(self, stats, parser_obj):
        """Write summary file (like legacy 00_check_dataset.py)."""
        summary_path = self.dataset_check_dir / "summary.txt"
        
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET VALIDATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("BASIC INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of images: {stats['num_images']}\n")
            f.write(f"  Number of cameras: {stats['num_cameras']}\n")
            f.write(f"  Number of 3D points: {stats['num_points']}\n\n")
            
            f.write("IMAGE PROPERTIES\n")
            f.write("-" * 80 + "\n")
            unique_sizes = set(tuple(s) for s in stats['image_sizes'])
            for width, height in unique_sizes:
                count = sum(1 for s in stats['image_sizes'] if s[0] == width and s[1] == height)
                f.write(f"  Size {width}x{height}: {count} images\n")
            f.write("\n")
            
            f.write("CAMERA INTRINSICS\n")
            f.write("-" * 80 + "\n")
            fx_min, fy_min = stats['focal_length_min']
            fx_max, fy_max = stats['focal_length_max']
            fx_mean, fy_mean = stats['focal_length_mean']
            f.write(f"  Focal length (fx): min={fx_min:.2f}, max={fx_max:.2f}, mean={fx_mean:.2f}\n")
            f.write(f"  Focal length (fy): min={fy_min:.2f}, max={fy_max:.2f}, mean={fy_mean:.2f}\n")
            f.write("\n")
            
            f.write("CAMERA EXTRINSICS (Positions)\n")
            f.write("-" * 80 + "\n")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                f.write(f"  {axis}-axis: min={stats['camera_positions_min'][i]:.3f}, "
                       f"max={stats['camera_positions_max'][i]:.3f}, "
                       f"mean={stats['camera_positions_mean'][i]:.3f}, "
                       f"std={stats['camera_positions_std'][i]:.3f}\n")
            f.write("\n")
            
            f.write("POINT CLOUD STATISTICS\n")
            f.write("-" * 80 + "\n")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                f.write(f"  {axis}-axis: min={stats['points_min'][i]:.3f}, "
                       f"max={stats['points_max'][i]:.3f}, "
                       f"mean={stats['points_mean'][i]:.3f}, "
                       f"std={stats['points_std'][i]:.3f}\n")
            
            if 'points_error_mean' in stats:
                f.write(f"\n  Reprojection error: mean={stats['points_error_mean']:.4f}, "
                       f"std={stats['points_error_std']:.4f}\n")
            f.write("\n")
            
            f.write("IMAGE SAMPLES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  First image: {parser_obj.image_names[0]}\n")
            f.write(f"  Last image:  {parser_obj.image_names[-1]}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
    
    def _create_splats_with_optimizers(self, parser_obj, device, batch_size=1, scene_scale=1.0):
        """Initialize Gaussians from COLMAP points (like legacy create_splats_with_optimizers)."""
        import torch
        from utils import knn, rgb_to_sh
        
        # Initialize from SfM points
        points = torch.from_numpy(parser_obj.points).float()
        rgbs = torch.from_numpy(parser_obj.points_rgb / 255.0).float()
        
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        
        N = points.shape[0]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), 0.1))  # [N,]
        
        # Color is SH coefficients
        colors = torch.zeros((N, (self.sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        
        # Learning rates (same as legacy)
        means_lr = 1.6e-4 * scene_scale
        scales_lr = 5e-3
        quats_lr = 1e-3
        opacities_lr = 5e-2
        sh0_lr = 2.5e-3
        shN_lr = 2.5e-3 / 20
        
        params = [
            ("means", torch.nn.Parameter(points), means_lr * math.sqrt(batch_size)),
            ("scales", torch.nn.Parameter(scales), scales_lr * math.sqrt(batch_size)),
            ("quats", torch.nn.Parameter(quats), quats_lr * math.sqrt(batch_size)),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr * math.sqrt(batch_size)),
            ("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr * math.sqrt(batch_size)),
            ("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr * math.sqrt(batch_size)),
        ]
        
        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
        
        # Create optimizers with batch size scaling (like legacy)
        BS = batch_size
        optimizers = {
            name: torch.optim.Adam(
                [{"params": splats[name], "lr": lr, "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, _, lr in params
        }
        
        return splats, optimizers
    
    def _train(self, splats, optimizers, trainloader, parser_obj, device):
        """Train the 3DGS model (like legacy train function)."""
        import torch
        import torch.nn.functional as F
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
        from torchmetrics.functional import structural_similarity_index_measure
        from torch.utils.tensorboard import SummaryWriter
        from tqdm import tqdm
        
        # Setup strategy for densification
        scene_scale = parser_obj.scene_scale * 1.1
        strategy = DefaultStrategy()
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        
        # Tensorboard
        writer = SummaryWriter(log_dir=str(self.training_dir / "tb"))
        
        # Learning rate scheduler for means
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / self.iterations)
        )
        
        # Training loop
        trainloader_iter = iter(trainloader)
        pbar = tqdm(range(self.iterations), desc="Training")
        ssim_lambda = 0.2  # Same as legacy default
        
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
            
            # SH degree schedule (like legacy)
            sh_degree_to_use = min(step // 1000, self.sh_degree)
            
            # Rasterize (like legacy rasterize_splats)
            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"])
            
            # Concatenate SH coefficients up to the specified degree
            num_sh_bases = (sh_degree_to_use + 1) ** 2
            sh0 = splats["sh0"]  # [N, 1, 3]
            shN = splats["shN"]  # [N, 15, 3] for sh_degree=3
            colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
            colors = colors[:, :num_sh_bases, :]  # Truncate to current sh_degree
            
            renders, alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                packed=False,
                absgrad=True,
                sparse_grad=False,
                rasterize_mode="classic",
            )
            
            # Pre-backward step for strategy
            strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )
            
            # Loss (L1 + SSIM like legacy)
            l1loss = F.l1_loss(renders, pixels)
            ssimloss = 1.0 - structural_similarity_index_measure(
                renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), data_range=1.0
            )
            loss = l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda
            
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
    
    def _evaluate_and_render(self, splats, valset, device):
        """Evaluate and render validation views (like legacy evaluate_and_render)."""
        import torch
        import imageio.v2 as imageio
        import numpy as np
        import json
        from collections import defaultdict
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from gsplat.rendering import rasterization
        from tqdm import tqdm
        
        render_dir = self.training_dir / "renders"
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
        for i, data in enumerate(tqdm(valloader, desc="Rendering")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            
            # Render (like legacy)
            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"])
            
            num_sh_bases = (self.sh_degree + 1) ** 2
            sh0 = splats["sh0"]
            shN = splats["shN"]
            colors = torch.cat([sh0, shN], 1)
            colors = colors[:, :num_sh_bases, :]
            
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
                sh_degree=self.sh_degree,
                packed=False,
                absgrad=False,
                sparse_grad=False,
                rasterize_mode="classic",
            )
            
            renders = torch.clamp(renders, 0.0, 1.0)
            
            # Save side-by-side comparison (GT | Rendered) - like legacy
            canvas = torch.cat([pixels, renders], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(render_dir / f"view_{i:03d}.png", canvas)
            
            # Compute metrics
            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            renders_p = renders.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(psnr_fn(renders_p, pixels_p))
            metrics["ssim"].append(ssim_fn(renders_p, pixels_p))
            metrics["lpips"].append(lpips_fn(renders_p, pixels_p))
        
        # Aggregate metrics
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats["num_GS"] = len(splats["means"])
        
        # Save metrics to JSON
        metrics_path = self.training_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
