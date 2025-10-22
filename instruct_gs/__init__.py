"""
InstructGS: Text-to-3D Editing Pipeline using Iterative Dataset Update (IDU)

This package implements the InstructGS pipeline that combines InstructPix2Pix 
with 3D Gaussian Splatting for text-guided 3D scene editing.
"""

from .pipeline import InstructGSPipeline, InstructGSPipelineConfig
from .model import InstructGSModel, InstructGSModelConfig
from .config import get_instruct_gs_config, INSTRUCT_GS_CONFIGS
from .utils import (
    save_edit_comparison,
    log_idu_statistics,
    estimate_memory_usage,
    validate_dataset_for_instruct_gs,
    check_gpu_requirements,
)

__all__ = [
    # Core components
    "InstructGSPipeline",
    "InstructGSPipelineConfig", 
    "InstructGSModel",
    "InstructGSModelConfig",
    
    # Configuration
    "get_instruct_gs_config",
    "INSTRUCT_GS_CONFIGS",
    
    # Utilities
    "save_edit_comparison",
    "log_idu_statistics", 
    "estimate_memory_usage",
    "validate_dataset_for_instruct_gs",
    "check_gpu_requirements",
]

__version__ = "0.1.0"
__author__ = "InstructGS Team"
__description__ = "Text-to-3D Scene Editing with Gaussian Splatting"