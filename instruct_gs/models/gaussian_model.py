"""3D Gaussian Splatting model wrapper for InstructGS."""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

# Add gsplat to path
project_root = Path(__file__).parent.parent.parent
gsplat_examples = project_root / "gsplat-src" / "examples"
sys.path.insert(0, str(gsplat_examples))

# Import gsplat components
from datasets.colmap import Dataset, Parser
from gsplat import rasterization
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import DefaultStrategy, MCMCStrategy


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 0.00016,
    scales_lr: float = 0.005,
    opacities_lr: float = 0.05,
    quats_lr: float = 0.001,
    sh0_lr: float = 0.0025,
    shN_lr: float = 0.000125,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create 3D Gaussians and optimizers from COLMAP data."""
    
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    N = points.shape[0]
    # Initialize Gaussian parameters
    fused_color = rgbs.float()
    fused_point_cloud = points.float()

    print(f"Initializing {N} Gaussians from {init_type}")

    dist2 = torch.clamp_min(
        torch.cdist(fused_point_cloud[None], fused_point_cloud[None])[0]
        .topk(k=4, dim=-1, largest=False)
        .values[:, 1:],
        1e-7,
    ).mean(dim=-1, keepdim=True)
    
    # Initialize scales
    scales = torch.log(torch.sqrt(dist2)).repeat(1, 3) * init_scale
    
    # Initialize rotations (quaternions)
    quats = torch.rand((N, 4))
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    # Initialize opacities
    opacities = torch.logit(torch.full((N,), init_opacity))
    
    # Initialize spherical harmonics
    def RGB2SH(rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    sh_dim = (sh_degree + 1) ** 2
    shs = torch.zeros((N, sh_dim, 3))
    if sh_dim > 0:
        shs[:, 0, :] = RGB2SH(fused_color)
    
    # Create parameters
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(fused_point_cloud.to(device)),
        "scales": torch.nn.Parameter(scales.to(device)),
        "quats": torch.nn.Parameter(quats.to(device)),
        "opacities": torch.nn.Parameter(opacities.to(device)),
        "sh0": torch.nn.Parameter(shs[:, :1, :].contiguous().to(device)),
        "shN": torch.nn.Parameter(shs[:, 1:, :].contiguous().to(device)),
    })

    # Create optimizers
    optimizers = {
        "means": torch.optim.Adam([splats["means"]], lr=means_lr),
        "scales": torch.optim.Adam([splats["scales"]], lr=scales_lr),
        "quats": torch.optim.Adam([splats["quats"]], lr=quats_lr),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=opacities_lr),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=sh0_lr),
        "shN": torch.optim.Adam([splats["shN"]], lr=shN_lr),
    }

    return splats, optimizers


class InstructGaussianModel:
    """3D Gaussian Splatting model wrapper for InstructGS editing."""
    
    def __init__(self, config, path_manager):
        """Initialize the Gaussian model."""
        self.config = config
        self.path_manager = path_manager
        self.device = config.device
        
        # Load dataset using gsplat's system
        self.parser = Parser(
            data_dir=config.data['data_dir'],
            factor=config.data['data_factor'],
            normalize=config.data['normalize_world_space'],
            test_every=config.data['test_every'],
        )
        
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=None,  # Full images for editing
            load_depths=False,
        )
        
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1
        
        # Initialize 3D Gaussians
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=config.gaussian['init_type'],
            init_num_pts=config.gaussian['init_num_pts'],
            init_extent=config.gaussian['init_extent'],
            init_opacity=config.gaussian['init_opacity'],
            init_scale=config.gaussian['init_scale'],
            means_lr=config.optimization['means_lr'],
            scales_lr=config.optimization['scales_lr'],
            opacities_lr=config.optimization['opacities_lr'],
            quats_lr=config.optimization['quats_lr'],
            sh0_lr=config.optimization['sh0_lr'],
            shN_lr=config.optimization['shN_lr'],
            scene_scale=self.scene_scale,
            sh_degree=config.gaussian['sh_degree'],
            device=self.device,
        )
        
        # Initialize densification strategy
        strategy_type = config.optimization['strategy']
        if strategy_type == "default":
            self.strategy = DefaultStrategy(
                prune_opa=config.optimization['prune_opa'],
                grow_grad2d=config.optimization['grow_grad2d'],
                grow_scale3d=config.optimization['grow_scale3d'],
                prune_scale3d=config.optimization['prune_scale3d'],
                refine_start_iter=config.optimization['refine_start_iter'],
                refine_stop_iter=config.optimization['refine_stop_iter'],
                refine_every=config.optimization['refine_every'],
                reset_every=config.optimization['reset_every'],
            )
        elif strategy_type == "mcmc":
            self.strategy = MCMCStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        self.strategy_state = self.strategy.initialize_state(self.splats)
        
        # RoI weights for object-scoped editing
        self.roi_weights = torch.ones(len(self.splats["means"]), device=self.device)
        
        print(f"✓ Gaussian model initialized with {len(self.splats['means'])} Gaussians")
        print(f"  Scene scale: {self.scene_scale:.3f}")
        print(f"  Training images: {len(self.trainset)}")
        print(f"  Validation images: {len(self.valset)}")
    
    def render_view(self, camera_params: Dict[str, torch.Tensor], sh_degree: int = None) -> Dict[str, torch.Tensor]:
        """Render a view using the current 3D Gaussians."""
        if sh_degree is None:
            sh_degree = self.config.gaussian['sh_degree']
        
        # Extract camera parameters
        camtoworld = camera_params['camtoworld']  # [4, 4]
        K = camera_params['K']  # [3, 3]
        width = camera_params['width']
        height = camera_params['height']
        
        # Render using gsplat
        renders, alphas, info = rasterization(
            means=self.splats["means"],
            scales=self.splats["scales"],
            quats=self.splats["quats"],
            opacities=self.splats["opacities"],
            sh0=self.splats["sh0"],
            shN=self.splats["shN"][:, :sh_degree**2, :] if sh_degree > 0 else None,
            camtoworlds=camtoworld[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
        )
        
        return {
            'image': renders[0],  # [H, W, 3]
            'alpha': alphas[0],   # [H, W, 1]
            'info': info,
        }
    
    def render_training_views(self, view_indices: Optional[list] = None) -> Dict[int, torch.Tensor]:
        """Render all training views (or specified subset)."""
        if view_indices is None:
            view_indices = list(range(len(self.trainset)))
        
        renders = {}
        for idx in view_indices:
            data = self.trainset[idx]
            
            camera_params = {
                'camtoworld': torch.from_numpy(data['camtoworld']).float().to(self.device),
                'K': torch.from_numpy(data['K']).float().to(self.device),
                'width': data['image'].shape[1],
                'height': data['image'].shape[0],
            }
            
            result = self.render_view(camera_params)
            renders[idx] = result['image'].detach().cpu()
        
        return renders
    
    def get_camera_params(self, view_idx: int) -> Dict[str, torch.Tensor]:
        """Get camera parameters for a specific view."""
        data = self.trainset[view_idx]
        return {
            'camtoworld': torch.from_numpy(data['camtoworld']).float().to(self.device),
            'K': torch.from_numpy(data['K']).float().to(self.device),
            'width': data['image'].shape[1],
            'height': data['image'].shape[0],
            'gt_image': torch.from_numpy(data['image']).float().to(self.device) / 255.0,
        }
    
    def optimize_step(self, loss: torch.Tensor, step: int):
        """Perform one optimization step."""
        # Backward pass
        loss.backward()
        
        # Apply densification strategy
        self.strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=getattr(self, '_last_render_info', {}),
        )
        
        # Optimizer step
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    
    def set_roi_weights(self, weights: torch.Tensor):
        """Set region-of-interest weights for object-scoped editing."""
        assert weights.shape[0] == len(self.splats["means"])
        self.roi_weights = weights.to(self.device)
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'splats': {k: v.detach().cpu() for k, v in self.splats.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'strategy_state': self.strategy_state,
            'roi_weights': self.roi_weights.detach().cpu(),
            'scene_scale': self.scene_scale,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load splats
        for k, v in checkpoint['splats'].items():
            self.splats[k].data = v.to(self.device)
        
        # Load optimizer states
        for k, state_dict in checkpoint['optimizers'].items():
            self.optimizers[k].load_state_dict(state_dict)
        
        # Load other states
        self.strategy_state = checkpoint.get('strategy_state', {})
        self.roi_weights = checkpoint.get('roi_weights', self.roi_weights).to(self.device)
        self.scene_scale = checkpoint.get('scene_scale', self.scene_scale)
        
        print(f"✓ Loaded checkpoint from {path}")
    
    def get_num_gaussians(self) -> int:
        """Get current number of Gaussians."""
        return len(self.splats["means"])
    
    def __len__(self):
        """Return number of training views."""
        return len(self.trainset)