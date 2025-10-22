"""
InstructGS Model Configuration and Implementation

This module implements the InstructGS model which extends SplatfactoModel
to include image buffer management and IP2P integration for text-to-3D editing.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.colors import get_color
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackLocation


@dataclass
class InstructGSModelConfig(SplatfactoModelConfig):
    """InstructGS Model Configuration extending SplatfactoModel"""
    
    _target: Type = field(default_factory=lambda: InstructGSModel)
    
    # InstructPix2Pix Parameters
    edit_prompt: str = "Turn it into a painting"
    """Text prompt for editing instruction"""
    
    ip2p_guidance_scale: float = 7.5
    """Guidance scale for text conditioning in InstructPix2Pix"""
    
    ip2p_image_guidance_scale: float = 1.5
    """Guidance scale for image conditioning in InstructPix2Pix"""
    
    ip2p_device: str = "cuda:1"
    """Device to place InstructPix2Pix model (separate from main model if possible)"""
    
    # IDU Cycle Parameters
    cycle_steps: int = 2500
    """Number of training steps between each editing cycle"""
    
    # Image Buffer Parameters
    image_resolution: Tuple[int, int] = (512, 512)
    """Resolution for rendered and edited images"""
    
    load_original_images: bool = True
    """Whether to load and store original training images for conditioning"""


class InstructGSModel(SplatfactoModel):
    """
    InstructGS Model implementing Iterative Dataset Update (IDU) for text-to-3D editing.
    
    This model extends SplatfactoModel with:
    1. Image buffer management (original_images, edited_buffer)
    2. InstructPix2Pix integration for 2D editing
    3. IDU cycle logic for 3D consistency
    """
    
    config: InstructGSModelConfig
    
    def __init__(self, config: InstructGSModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Initialize step counter for IDU cycles
        self.global_step = 0
        
        # Image buffers - will be initialized in populate_modules
        self.original_images: Optional[Dict[str, torch.Tensor]] = None
        self.edited_buffer: Optional[Dict[str, torch.Tensor]] = None
        
        # InstructPix2Pix pipeline - will be initialized lazily
        self.ip2p_pipe = None
        self._ip2p_initialized = False
        
    def populate_modules(self):
        """Initialize the model components including image buffers and IP2P."""
        super().populate_modules()
        
        # Initialize image buffers
        self._initialize_image_buffers()
        
        # InstructPix2Pix will be initialized lazily on first use
        
    def _initialize_image_buffers(self):
        """Initialize the original images and edited buffer dictionaries."""
        # These will be populated when we have access to the datamanager
        self.original_images = {}
        self.edited_buffer = {}
        
        print(f"InstructGS: Initialized image buffers for editing")
        print(f"Edit prompt: '{self.config.edit_prompt}'")
        print(f"Cycle steps: {self.config.cycle_steps}")
        
    def _initialize_ip2p(self):
        """Lazily initialize InstructPix2Pix pipeline."""
        if self._ip2p_initialized:
            return
            
        try:
            from diffusers import StableDiffusionInstructPix2PixPipeline
            
            print(f"Loading InstructPix2Pix on device: {self.config.ip2p_device}")
            
            # Load the InstructPix2Pix pipeline
            self.ip2p_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16 if "cuda" in self.config.ip2p_device else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Move to specified device
            self.ip2p_pipe = self.ip2p_pipe.to(self.config.ip2p_device)
            
            # Enable memory efficient attention if available
            if hasattr(self.ip2p_pipe, "enable_attention_slicing"):
                self.ip2p_pipe.enable_attention_slicing()
                
            self._ip2p_initialized = True
            print("InstructPix2Pix pipeline initialized successfully")
            
        except ImportError:
            raise ImportError(
                "diffusers library is required for InstructGS. "
                "Install with: pip install diffusers transformers accelerate"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize InstructPix2Pix: {e}")
            print("Continuing without IP2P - editing will be disabled")
            
    def load_original_images_from_datamanager(self, datamanager):
        """Load original training images from the datamanager."""
        if not self.config.load_original_images:
            return
            
        print("Loading original training images...")
        
        # Get all training images from datamanager
        train_dataset = datamanager.train_dataset
        
        # Also store cameras for rendering during IDU cycles
        if hasattr(train_dataset, 'cameras'):
            self.set_training_cameras(train_dataset.cameras)
        
        for i in range(len(train_dataset)):
            try:
                # Get camera and batch data
                camera = train_dataset.cameras[i:i+1]  # Single camera
                image_idx = i
                
                # Load the original image
                if hasattr(train_dataset, "get_data"):
                    batch = train_dataset.get_data(image_idx)
                    original_image = batch["image"]  # Should be (H, W, 3)
                else:
                    # Fallback: try to access image directly
                    original_image = train_dataset[i]["image"]
                
                # Ensure proper format and device
                if isinstance(original_image, torch.Tensor):
                    original_image = original_image.clone().detach()
                else:
                    original_image = torch.from_numpy(np.array(original_image))
                
                # Ensure float32 and range [0, 1]
                if original_image.dtype != torch.float32:
                    original_image = original_image.float()
                if original_image.max() > 1.0:
                    original_image = original_image / 255.0
                    
                # Store in original_images buffer
                self.original_images[f"image_{i:06d}"] = original_image
                
                # Initialize edited_buffer with copy of original
                self.edited_buffer[f"image_{i:06d}"] = original_image.clone()
                
            except Exception as e:
                print(f"Warning: Failed to load image {i}: {e}")
                continue
                
        print(f"Loaded {len(self.original_images)} original training images")
        
    def should_run_idu_cycle(self) -> bool:
        """Check if an IDU cycle should be run at the current step."""
        return (self.global_step > 0 and 
                self.global_step % self.config.cycle_steps == 0 and
                self._ip2p_initialized and
                len(self.original_images) > 0)
        
    def run_idu_cycle(self):
        """
        Execute one complete Iterative Dataset Update (IDU) cycle:
        1. Render current 3D scene from all training views
        2. Edit each rendered image using InstructPix2Pix
        3. Update the edited_buffer with new images
        """
        if not self.should_run_idu_cycle():
            return
            
        print(f"Running IDU cycle at step {self.global_step}")
        
        # Initialize IP2P if not already done
        self._initialize_ip2p()
        
        if not self._ip2p_initialized:
            print("Skipping IDU cycle - InstructPix2Pix not available")
            return
            
        # Step 1: Render current 3D scene from all training views
        rendered_images = self._run_3d_rendering_pass()
        
        if len(rendered_images) == 0:
            print("No images rendered, skipping editing phase")
            return
            
        # Step 2: Edit all rendered images using InstructPix2Pix
        self._run_global_2d_editing(rendered_images)
        
        print(f"Completed IDU cycle at step {self.global_step}")
        
    def set_training_cameras(self, cameras):
        """Set the training cameras for rendering during IDU cycles."""
        self._training_cameras = cameras
        print(f"Set {len(cameras) if cameras else 0} training cameras for IDU rendering")
        
    def _run_3d_rendering_pass(self) -> Dict[str, torch.Tensor]:
        """
        Render the current 3D scene from all training camera poses.
        
        Returns:
            Dictionary mapping image keys to rendered images
        """
        print("Running 3D rendering pass...")
        
        rendered_images = {}
        
        if not hasattr(self, '_training_cameras') or self._training_cameras is None:
            print("Warning: No training cameras available for rendering")
            return rendered_images
            
        # Set model to eval mode for consistent rendering
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                for i, camera in enumerate(self._training_cameras):
                    image_key = f"image_{i:06d}"
                    
                    try:
                        # Render from this camera
                        outputs = self.get_outputs(camera)
                        rendered_image = outputs["rgb"]  # Should be (H, W, 3)
                        
                        # Ensure proper format
                        if rendered_image.dim() == 4:
                            rendered_image = rendered_image.squeeze(0)  # Remove batch dim
                            
                        # Clamp to valid range
                        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
                        
                        # Store rendered image
                        rendered_images[image_key] = rendered_image.detach().cpu()
                        
                    except Exception as e:
                        print(f"Warning: Failed to render camera {i}: {e}")
                        continue
                        
        finally:
            # Restore training mode
            if was_training:
                self.train()
                
        print(f"Rendered {len(rendered_images)} images from 3D scene")
        return rendered_images
        
    def _run_global_2d_editing(self, rendered_images: Dict[str, torch.Tensor]):
        """
        Edit all rendered images using InstructPix2Pix with dual conditioning.
        
        Args:
            rendered_images: Dictionary of rendered images from 3D scene
        """
        print("Running global 2D editing pass...")
        
        if not self._ip2p_initialized or self.ip2p_pipe is None:
            print("InstructPix2Pix not available, skipping editing")
            return
            
        new_edited_images = {}
        edit_count = 0
        
        for image_key, rendered_image in rendered_images.items():
            if image_key not in self.original_images:
                print(f"Warning: No original image for {image_key}")
                continue
                
            try:
                # Get original image for conditioning
                original_image = self.original_images[image_key]
                
                # Convert to PIL Images for InstructPix2Pix
                rendered_pil = self._tensor_to_pil(rendered_image)
                original_pil = self._tensor_to_pil(original_image)
                
                # Resize to target resolution if needed
                target_size = self.config.image_resolution
                rendered_pil = rendered_pil.resize(target_size, Image.LANCZOS)
                original_pil = original_pil.resize(target_size, Image.LANCZOS)
                
                # Run InstructPix2Pix with dual conditioning
                edited_pil = self.ip2p_pipe(
                    prompt=self.config.edit_prompt,
                    image=rendered_pil,  # Image to edit (current 3D render)
                    guidance_scale=self.config.ip2p_guidance_scale,
                    image_guidance_scale=self.config.ip2p_image_guidance_scale,
                    num_inference_steps=20,  # Fewer steps for faster iteration
                    generator=torch.Generator(device=self.config.ip2p_device).manual_seed(42)
                ).images[0]
                
                # Convert back to tensor and store
                edited_tensor = self._pil_to_tensor(edited_pil)
                new_edited_images[image_key] = edited_tensor
                edit_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to edit {image_key}: {e}")
                # Keep the rendered image as fallback
                new_edited_images[image_key] = rendered_image
                continue
                
        # Update the edited buffer with new images
        self.edited_buffer.update(new_edited_images)
        
        print(f"Successfully edited {edit_count}/{len(rendered_images)} images")
        
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor image to PIL Image."""
        # Ensure tensor is on CPU and in range [0, 1]
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # Convert to numpy and scale to [0, 255]
        numpy_image = (tensor.numpy() * 255).astype(np.uint8)
        
        # Convert to PIL
        if numpy_image.ndim == 3:
            return Image.fromarray(numpy_image)
        else:
            raise ValueError(f"Expected 3D tensor, got {numpy_image.ndim}D")
            
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert a PIL Image to tensor."""
        # Convert to numpy and normalize to [0, 1]
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(numpy_image)
        
        return tensor
        
    def get_train_loss_dict(self, outputs, batch, metrics_dict=None):
        """
        Override to pull images from edited_buffer instead of original images.
        """
        # For now, use the parent implementation
        # Later this will be modified to use edited_buffer images
        return super().get_loss_dict(outputs, batch, metrics_dict)
        
    def step_callback(self, step: int):
        """Called at each training step to update global step and potentially run IDU."""
        self.global_step = step
        
        # Check if we should run an IDU cycle
        if self.should_run_idu_cycle():
            with torch.no_grad():  # Don't accumulate gradients during editing
                self.run_idu_cycle()
                
    def get_training_callbacks(self, training_callback_attributes):
        """Add our step callback to the training callbacks."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        # Add our custom callback for IDU cycles - called before each training iteration
        def step_callback_wrapper(step, optimizers=None):
            self.step_callback(step)
            
        step_callback = TrainingCallback(
            where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            func=step_callback_wrapper,
        )
        callbacks.append(step_callback)
        
        return callbacks