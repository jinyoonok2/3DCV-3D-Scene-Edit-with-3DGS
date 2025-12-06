"""
Phase 2: Segmentation - Object Detection, Masking, and 3D ROI Extraction

Combines functionality from:
- legacy/02_render_training_views.py (render views from checkpoint)
- legacy/03_ground_text_to_masks.py (GroundingDINO + SAM2)
- legacy/04a_lift_masks_to_roi3d.py (2D masks → 3D ROI)

This phase:
1. Loads trained checkpoint from Phase 1
2. Renders all training views
3. Uses GroundingDINO to detect target object
4. Uses SAM2 to generate precise masks
5. Projects masks and lifts to 3D ROI

Outputs:
- phase2_segmentation/
  ├── 02_rendered_views/
  │   └── renders/          # Rendered training views
  ├── 03_generated_masks/
  │   ├── boxes/            # GroundingDINO detection boxes
  │   ├── sam_masks/        # SAM2 generated masks
  │   ├── projected_masks/  # Masks projected onto training views
  │   └── overlays/         # Visualization overlays
  ├── 04_roi_extraction/
  │   └── roi.pt           # 3D ROI binary mask
  └── manifest.json        # Metadata
"""

import sys
from pathlib import Path
from typing import Any, Dict
import torch
import numpy as np

# Add gsplat to path
gsplat_path = str(Path(__file__).parent.parent / "gsplat-src" / "examples")
if gsplat_path not in sys.path:
    sys.path.insert(0, gsplat_path)

from .base import BasePhase


class Phase2Segmentation(BasePhase):
    """Phase 2: Object segmentation and 3D ROI extraction."""
    
    def __init__(self, config):
        super().__init__(config, phase_name="segmentation", phase_number=2)
        
        # Get segmentation config
        self.text_prompt = config.get("segmentation", "text_prompt")
        self.box_threshold = config.get("segmentation", "box_threshold", default=0.35)
        self.text_threshold = config.get("segmentation", "text_threshold", default=0.25)
        self.nms_threshold = config.get("segmentation", "nms_threshold", default=0.8)
        self.sam_model = config.get("segmentation", "sam_model", default="sam2_hiera_large")
        self.dino_selection = config.get("segmentation", "dino_selection", default="largest")
        self.sam_selection = config.get("segmentation", "sam_selection", default=None)
        self.sam_thresh = config.get("segmentation", "sam_thresh", default=0.3)
        
        # Spatial filtering (optional)
        self.reference_box = config.get("segmentation", "reference_box", default=None)
        self.reference_box_normalized = config.get("segmentation", "reference_box_normalized", default=True)
        self.reference_overlap_thresh = config.get("segmentation", "reference_overlap_thresh", default=0.7)
        
        # ROI config
        self.roi_threshold = config.get("roi", "threshold", default=0.05)
        self.roi_min_views = config.get("roi", "min_views", default=3)
        
        # Dataset config
        self.dataset_root = Path(config.get("paths", "dataset_root"))
        self.factor = config.get("dataset", "factor", default=4)
        self.test_every = config.get("dataset", "test_every", default=8)
        
        # Create subdirectories matching old numbered structure
        (self.phase_dir / "02_rendered_views" / "renders").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "03_generated_masks" / "boxes").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "03_generated_masks" / "sam_masks").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "03_generated_masks" / "projected_masks").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "03_generated_masks" / "overlays").mkdir(parents=True, exist_ok=True)
        (self.phase_dir / "04_roi_extraction").mkdir(exist_ok=True)
        
        self.metadata.update({
            "text_prompt": self.text_prompt,
            "box_threshold": self.box_threshold,
            "sam_model": self.sam_model,
            "roi_threshold": self.roi_threshold,
        })
    
    def validate_inputs(self) -> bool:
        """Validate that Phase 1 checkpoint exists."""
        from rich.console import Console
        console = Console()
        
        # Check for Phase 1 checkpoint
        phase1_dir = self.output_root / "phase1_training"
        ckpt_path = phase1_dir / "01_initial_training" / "ckpt_initial.pt"
        
        if not ckpt_path.exists():
            console.print(f"[red]Phase 1 checkpoint not found:[/red] {ckpt_path}")
            console.print("[yellow]Run Phase 1 first:[/yellow] python run_pipeline.py --phase 1")
            return False
        
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute Phase 2: Segmentation and ROI extraction."""
        from rich.console import Console
        console = Console()
        
        # Import heavy dependencies only when needed
        import imageio.v2 as imageio
        from datasets.colmap import Dataset, Parser
        from gsplat.rendering import rasterization
        from tqdm import tqdm
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Step 1: Load checkpoint from Phase 1
        console.print("[bold cyan]Step 1/5: Loading Phase 1 checkpoint...[/bold cyan]")
        phase1_dir = self.output_root / "phase1_training"
        ckpt_path = phase1_dir / "01_initial_training" / "ckpt_initial.pt"
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        splats = torch.nn.ParameterDict(checkpoint["splats"])
        console.print(f"✓ Loaded checkpoint with {len(splats['means'])} Gaussians\n")
        
        # Step 2: Load dataset
        console.print("[bold cyan]Step 2/5: Loading dataset...[/bold cyan]")
        parser_obj = Parser(
            data_dir=str(self.dataset_root),
            factor=self.factor,
            normalize=True,
            test_every=self.test_every,
        )
        trainset = Dataset(parser_obj, split="train")
        console.print(f"✓ Loaded {len(trainset)} training images\n")
        
        # Step 3: Render training views
        console.print(f"[bold cyan]Step 3/5: Rendering {len(trainset)} training views...[/bold cyan]")
        renders_dir = self.phase_dir / "02_rendered_views" / "renders"
        
        for idx in tqdm(range(len(trainset)), desc="Rendering"):
            data = trainset[idx]
            camtoworlds = torch.from_numpy(data["camtoworld"]).float().to(device)[None]
            Ks = torch.from_numpy(data["K"]).float().to(device)[None]
            image = data["image"]
            height, width = image.shape[:2]
            
            # Render
            means = splats["means"]
            quats = splats["quats"]
            scales = torch.exp(splats["scales"])
            opacities = torch.sigmoid(splats["opacities"])
            sh0 = splats["sh0"]
            shN = splats["shN"]
            colors = torch.cat([sh0, shN], dim=1)
            
            with torch.no_grad():
                renders, _, _ = rasterization(
                    means=means,
                    quats=quats / quats.norm(dim=-1, keepdim=True),
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(camtoworlds),
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=3,
                    packed=False,
                )
            
            # Save render
            render_img = (renders[0].cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(renders_dir / f"{idx:04d}.png", render_img)
        
        console.print("✓ Rendering completed\n")
        
        # Step 4: Run GroundingDINO + SAM2
        console.print("[bold cyan]Step 4/5: Detecting objects and generating masks...[/bold cyan]")
        self._run_grounding_sam(renders_dir, console)
        console.print("✓ Mask generation completed\n")
        
        # Step 5: Lift masks to 3D ROI
        console.print("[bold cyan]Step 5/5: Lifting 2D masks to 3D ROI...[/bold cyan]")
        roi_mask = self._lift_masks_to_roi(splats, parser_obj, trainset, console, device)
        
        # Save ROI
        roi_path = self.phase_dir / "04_roi_extraction" / "roi.pt"
        torch.save(roi_mask, roi_path)
        console.print(f"✓ ROI saved: {roi_path}\n")
        
        results = {
            "roi_path": str(roi_path),
            "num_renders": len(trainset),
            "roi_gaussians": roi_mask.sum().item(),
            "total_gaussians": len(splats["means"]),
        }
        
        return results
    
    def _run_grounding_sam(self, renders_dir: Path, console):
        """Run GroundingDINO and SAM2 to generate masks."""
        # This is a simplified placeholder - full implementation would import and run:
        # - GroundingDINO for object detection
        # - SAM2 for mask generation
        # For now, just create dummy outputs to test the structure
        
        console.print("  [yellow]Note: Using simplified mask generation (full GroundingDINO/SAM2 coming next)[/yellow]")
        
        import cv2
        render_files = sorted(renders_dir.glob("*.png"))
        
        for img_path in render_files:
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
            
            # Create dummy mask (center region for testing)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
            # Save dummy outputs
            cv2.imwrite(str(self.phase_dir / "03_generated_masks" / "boxes" / img_path.name), image)
            cv2.imwrite(str(self.phase_dir / "03_generated_masks" / "sam_masks" / img_path.name), mask)
            
            overlay = image.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            cv2.imwrite(str(self.phase_dir / "overlays" / img_path.name), overlay)
    
    def _lift_masks_to_roi(self, splats, parser_obj, trainset, console, device):
        """Project 2D masks to 3D and compute ROI."""
        import cv2
        
        # Load all masks
        mask_files = sorted((self.phase_dir / "03_generated_masks" / "sam_masks").glob("*.png"))
        num_gaussians = len(splats["means"])
        roi_weights = torch.zeros(num_gaussians, device=device)
        
        # Simplified: mark center Gaussians as ROI for testing
        # Full implementation would project masks properly
        means = splats["means"].cpu().numpy()
        center = means.mean(axis=0)
        distances = np.linalg.norm(means - center, axis=1)
        threshold_dist = np.percentile(distances, 30)  # Inner 30% as ROI
        
        roi_mask = torch.from_numpy(distances < threshold_dist).to(device)
        
        console.print(f"  ROI: {roi_mask.sum().item()} / {num_gaussians} Gaussians ({100*roi_mask.sum().item()/num_gaussians:.1f}%)")
        
        return roi_mask
