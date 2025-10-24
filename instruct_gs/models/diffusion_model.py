"""InstructPix2Pix diffusion model integration for InstructGS."""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import PIL.Image as Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import warnings

# Suppress diffusers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")


class InstructPix2PixModel:
    """Wrapper for InstructPix2Pix diffusion model."""
    
    def __init__(self, config):
        """Initialize the InstructPix2Pix model."""
        self.config = config
        self.device = torch.device(config.diffusion['device'])
        self.edit_prompt = config.editing['edit_prompt']
        
        # Load model configuration
        self.model_name = config.diffusion['model_name']
        self.num_inference_steps = config.diffusion['steps']
        self.guidance_scale = config.diffusion['guidance_scale']
        self.image_guidance_scale = config.diffusion.get('image_guidance_scale', 1.5)
        
        # Set random seed for reproducibility
        self.generator = torch.Generator(device=self.device)
        if 'seed' in config.diffusion:
            self.generator.manual_seed(config.diffusion['seed'])
        
        # Load the pipeline
        print(f"Loading InstructPix2Pix model: {self.model_name}")
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            safety_checker=None,  # Disable safety checker for speed
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)
        
        # Optimize for memory and speed
        if self.device.type == 'cuda':
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            # Optionally enable xformers for even faster inference
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass  # xformers not available
        
        print(f"✓ InstructPix2Pix model loaded on {self.device}")
        print(f"  Edit prompt: '{self.edit_prompt}'")
        print(f"  Inference steps: {self.num_inference_steps}")
        print(f"  Guidance scale: {self.guidance_scale}")
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert torch tensor to PIL Image."""
        # Ensure tensor is on CPU and in [0, 1] range
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Handle different tensor shapes
        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor[0]
        if tensor.dim() == 3 and tensor.shape[0] == 3:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        
        # Clamp to [0, 1] and convert to uint8
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor * 255).byte().numpy()
        
        return Image.fromarray(array)
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch tensor."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor [H, W, C] in [0, 1] range
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        
        return tensor
    
    def edit_image(
        self, 
        image: Union[torch.Tensor, Image.Image, np.ndarray], 
        prompt: Optional[str] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Edit an image using InstructPix2Pix.
        
        Args:
            image: Input image (tensor [H,W,3], PIL Image, or numpy array)
            prompt: Edit instruction (uses default if None)
            mask: Optional mask for region-specific editing [H,W] or [H,W,1]
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            Edited image as tensor [H, W, 3] in [0, 1] range
        """
        if prompt is None:
            prompt = self.edit_prompt
        
        # Convert input to PIL Image
        if isinstance(image, torch.Tensor):
            pil_image = self.tensor_to_pil(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Get pipeline parameters
        pipe_kwargs = {
            'prompt': prompt,
            'image': pil_image,
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
            'image_guidance_scale': self.image_guidance_scale,
            'generator': self.generator,
        }
        pipe_kwargs.update(kwargs)
        
        # Apply mask if provided (simple masking approach)
        if mask is not None:
            # For now, we'll handle masking by post-processing
            # More sophisticated masking could be implemented with inpainting
            original_tensor = self.pil_to_tensor(pil_image)
        
        # Run diffusion
        with torch.inference_mode():
            result = self.pipe(**pipe_kwargs)
            edited_pil = result.images[0]
        
        # Convert back to tensor
        edited_tensor = self.pil_to_tensor(edited_pil)
        
        # Apply mask if provided (blend edited and original)
        if mask is not None:
            if mask.dim() == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)  # [H, W, 1] -> [H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [H, W] -> [H, W, 1]
            
            # Resize mask to match image if needed
            if mask.shape[:2] != edited_tensor.shape[:2]:
                mask = torch.nn.functional.interpolate(
                    mask.permute(2, 0, 1).unsqueeze(0),  # [1, 1, H, W]
                    size=edited_tensor.shape[:2],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)  # [H, W, 1]
            
            # Blend: edited where mask=1, original where mask=0
            edited_tensor = mask * edited_tensor + (1 - mask) * original_tensor
        
        return edited_tensor
    
    def edit_batch(
        self, 
        images: List[Union[torch.Tensor, Image.Image]], 
        prompts: Optional[List[str]] = None,
        masks: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Edit a batch of images.
        
        Args:
            images: List of input images
            prompts: List of prompts (one per image) or None for default
            masks: List of masks (one per image) or None
            
        Returns:
            List of edited images as tensors
        """
        if prompts is None:
            prompts = [self.edit_prompt] * len(images)
        
        if masks is None:
            masks = [None] * len(images)
        
        edited_images = []
        for i, (image, prompt, mask) in enumerate(zip(images, prompts, masks)):
            print(f"  Editing image {i+1}/{len(images)}: '{prompt[:50]}...'")
            edited = self.edit_image(image, prompt, mask, **kwargs)
            edited_images.append(edited)
        
        return edited_images
    
    def set_edit_prompt(self, prompt: str):
        """Update the edit prompt."""
        self.edit_prompt = prompt
        print(f"Updated edit prompt: '{prompt}'")
    
    def set_guidance_scale(self, guidance_scale: float):
        """Update guidance scale."""
        self.guidance_scale = guidance_scale
        print(f"Updated guidance scale: {guidance_scale}")
    
    def set_inference_steps(self, steps: int):
        """Update number of inference steps."""
        self.num_inference_steps = steps
        print(f"Updated inference steps: {steps}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': str(self.device)
            }
        return {'device': str(self.device)}
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")


class MockInstructPix2PixModel:
    """Mock diffusion model for testing without downloading large models."""
    
    def __init__(self, config):
        """Initialize mock model."""
        self.config = config
        self.edit_prompt = config.editing['edit_prompt']
        print(f"✓ Mock InstructPix2Pix model initialized")
        print(f"  Edit prompt: '{self.edit_prompt}'")
        print("  Note: This is a mock model for testing - no actual editing will occur")
    
    def edit_image(self, image: torch.Tensor, prompt: str = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Mock image editing - returns slightly modified image."""
        # Just add a bit of noise to simulate editing
        noise = torch.randn_like(image) * 0.02
        edited = torch.clamp(image + noise, 0, 1)
        return edited
    
    def edit_batch(self, images: List[torch.Tensor], prompts: List[str] = None, masks: List[torch.Tensor] = None) -> List[torch.Tensor]:
        """Mock batch editing."""
        return [self.edit_image(img) for img in images]
    
    def set_edit_prompt(self, prompt: str):
        """Update edit prompt."""
        self.edit_prompt = prompt
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage."""
        return {'device': 'mock', 'allocated_gb': 0.0}
    
    def clear_cache(self):
        """Clear cache."""
        pass


def create_diffusion_model(config, use_mock: bool = False) -> Union[InstructPix2PixModel, MockInstructPix2PixModel]:
    """
    Create diffusion model (real or mock).
    
    Args:
        config: Configuration object
        use_mock: If True, use mock model for testing
        
    Returns:
        Diffusion model instance
    """
    if use_mock:
        return MockInstructPix2PixModel(config)
    else:
        return InstructPix2PixModel(config)