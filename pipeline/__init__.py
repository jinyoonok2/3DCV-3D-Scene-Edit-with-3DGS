"""
3DGS Scene Editing Pipeline - Modular Class-Based Architecture

This package contains the unified pipeline phases for 3D Gaussian Splatting scene editing.
Each phase is a self-contained module with clear inputs/outputs.

Phases:
    1. Training: Dataset validation + Initial GS training
    2. Segmentation: View rendering + Mask generation + ROI extraction
    3. Removal: Gaussian removal + LaMa inpainting + Optimization
    4. Placement: Object placement + Final visualization

Usage:
    from pipeline import Phase1Training, Phase2Segmentation, Phase3Removal, Phase4Placement
    from project_utils.config import ProjectConfig
    
    config = ProjectConfig("configs/garden_config.yaml")
    
    # Run phases sequentially
    phase1 = Phase1Training(config)
    phase1.run()
    
    phase2 = Phase2Segmentation(config)
    phase2.run()
    
    # ... etc
"""

from .base import BasePhase
from .phase1_training import Phase1Training
from .phase2_segmentation import Phase2Segmentation
from .phase3_removal import Phase3Removal
from .phase4_placement import Phase4Placement

__all__ = [
    "BasePhase",
    "Phase1Training",
    "Phase2Segmentation",
    "Phase3Removal",
    "Phase4Placement",
]
