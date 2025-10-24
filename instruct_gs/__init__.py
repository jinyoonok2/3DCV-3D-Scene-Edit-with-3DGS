"""
InstructGS - Text-guided 3D Scene Editing with Gaussian Splatting

A framework for editing 3D scenes using text instructions by combining
3D Gaussian Splatting with 2D diffusion models in an iterative optimization loop.
"""

__version__ = "0.1.0"
__author__ = "InstructGS Team"

from .config.config_manager import ConfigManager, InstructGSConfig, load_config_with_overrides
from .utils.path_manager import PathManager
from .core.state_manager import TrainingStateManager

__all__ = [
    "ConfigManager",
    "InstructGSConfig", 
    "load_config_with_overrides",
    "PathManager",
    "TrainingStateManager",
]