"""
InstructGS Pipeline Implementation

This module implements the InstructGS pipeline that integrates with nerfstudio's
training system to provide text-to-3D editing capabilities through Iterative Dataset Update.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Any, Union
from pathlib import Path

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig  
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from .model import InstructGSModel, InstructGSModelConfig


@dataclass
class InstructGSPipelineConfig(VanillaPipelineConfig):
    """Configuration for InstructGS Pipeline"""
    
    _target: Type = field(default_factory=lambda: InstructGSPipeline)
    
    # Override model to use InstructGSModel
    model: InstructGSModelConfig = field(default_factory=InstructGSModelConfig)


class InstructGSPipeline(VanillaPipeline):
    """
    InstructGS Pipeline implementing text-to-3D editing via Iterative Dataset Update (IDU).
    
    This pipeline extends VanillaPipeline to:
    1. Initialize image buffers from training data
    2. Coordinate IDU cycles between 3D rendering and 2D editing  
    3. Route edited images to the training loop
    """
    
    config: InstructGSPipelineConfig
    model: InstructGSModel
    
    def __init__(
        self,
        config: InstructGSPipelineConfig,
        device: str,
        test_mode: str = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        
        # Initialize image buffers after datamanager is set up
        self._initialize_image_buffers()
        
    def _initialize_image_buffers(self):
        """Initialize image buffers from the training dataset."""
        if hasattr(self.model, 'load_original_images_from_datamanager'):
            try:
                self.model.load_original_images_from_datamanager(self.datamanager)
                print("Successfully initialized image buffers from datamanager")
            except Exception as e:
                print(f"Warning: Failed to initialize image buffers: {e}")
                print("Continuing without image buffer initialization")
                
    def get_train_loss_dict(self, step: int):
        """
        Override training loss computation to:
        1. Update model step counter for IDU cycles
        2. Use edited images when available
        3. Trigger IDU cycles at appropriate intervals
        """
        # Update the model's step counter
        if hasattr(self.model, 'global_step'):
            self.model.global_step = step
            
        # Check if we should run IDU cycle before training
        if hasattr(self.model, 'should_run_idu_cycle') and self.model.should_run_idu_cycle():
            print(f"Triggering IDU cycle at step {step}")
            with torch.no_grad():
                self.model.run_idu_cycle()
        
        # Get training data from datamanager
        ray_bundle, batch = self.datamanager.next_train(step)
        
        # TODO: Modify batch to use edited images when available
        # For now, use the standard pipeline behavior
        batch = self._get_edited_batch_if_available(batch, step)
        
        # Forward pass through model
        model_outputs = self._model(ray_bundle)
        
        # Compute metrics and losses
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        return model_outputs, loss_dict, metrics_dict
        
    def _get_edited_batch_if_available(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, torch.Tensor]:
        """
        Replace batch images with edited versions if available.
        
        Args:
            batch: Original batch from datamanager
            step: Current training step
            
        Returns:
            Modified batch with edited images if available
        """
        # Check if model has edited images available
        if (hasattr(self.model, 'edited_buffer') and 
            self.model.edited_buffer is not None and 
            len(self.model.edited_buffer) > 0):
            
            # Try to get image index from batch
            image_idx = None
            
            # Different ways to get image index depending on datamanager
            if "image_idx" in batch:
                image_idx = batch["image_idx"].item()
            elif "indices" in batch:
                # Some datamanagers use 'indices'
                indices = batch["indices"]
                if indices.numel() > 0:
                    image_idx = indices[0].item()
            else:
                # Fallback: try to infer from batch structure
                # This is a best-effort approach
                return batch
            
            if image_idx is not None:
                image_key = f"image_{image_idx:06d}"
                
                if image_key in self.model.edited_buffer:
                    # Replace with edited image
                    edited_image = self.model.edited_buffer[image_key]
                    
                    # Ensure proper format and device
                    if edited_image.device != batch["image"].device:
                        edited_image = edited_image.to(batch["image"].device)
                        
                    # Ensure same data type
                    if edited_image.dtype != batch["image"].dtype:
                        edited_image = edited_image.to(batch["image"].dtype)
                    
                    # Ensure same shape (resize if necessary)
                    if edited_image.shape != batch["image"].shape:
                        # For now, just print a warning - more sophisticated resizing could be added
                        print(f"Warning: Shape mismatch for {image_key}. "
                              f"Edited: {edited_image.shape}, Batch: {batch['image'].shape}")
                        # Try to resize to match
                        if len(edited_image.shape) == 3 and len(batch["image"].shape) == 3:
                            from torchvision.transforms.functional import resize
                            target_h, target_w = batch["image"].shape[:2]
                            edited_image = resize(edited_image.permute(2, 0, 1), 
                                                [target_h, target_w]).permute(1, 2, 0)
                    
                    # Update batch with edited image
                    batch = batch.copy()
                    batch["image"] = edited_image
                    
                    # Optional: Add flag to indicate edited image is being used
                    batch["is_edited"] = torch.tensor(True)
        
        return batch
        
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes):
        """Get training callbacks including IDU cycle management."""
        callbacks = super().get_training_callbacks(training_callback_attributes)
        
        # The model will add its own callbacks for IDU cycles
        return callbacks
        
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load pipeline state and ensure proper initialization."""
        super().load_pipeline(loaded_state, step)
        
        # Re-initialize image buffers after loading
        self._initialize_image_buffers()
        
        # Update model step counter
        if hasattr(self.model, 'global_step'):
            self.model.global_step = step
            
    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get parameter groups for optimization."""
        # Use the standard parameter groups from the model
        return self.model.get_param_groups()
        
    def export_3d_model(self, output_path: Path) -> None:
        """Export the current 3D model state."""
        # Delegate to model's export functionality if available
        if hasattr(self.model, 'export_3d_model'):
            self.model.export_3d_model(output_path)
        else:
            print("3D model export not implemented for this model type")
            
    def get_edit_status(self) -> Dict[str, Any]:
        """Get current editing status and statistics."""
        status = {
            "global_step": getattr(self.model, 'global_step', 0),
            "cycle_steps": self.config.model.cycle_steps,
            "edit_prompt": self.config.model.edit_prompt,
            "ip2p_initialized": getattr(self.model, '_ip2p_initialized', False),
            "num_original_images": len(getattr(self.model, 'original_images', {})),
            "num_edited_images": len(getattr(self.model, 'edited_buffer', {})),
        }
        
        # Calculate next IDU cycle step
        current_step = status["global_step"]
        cycle_steps = status["cycle_steps"]
        if cycle_steps > 0:
            next_cycle = ((current_step // cycle_steps) + 1) * cycle_steps
            status["next_idu_cycle_step"] = next_cycle
            status["steps_until_next_cycle"] = next_cycle - current_step
            
        return status