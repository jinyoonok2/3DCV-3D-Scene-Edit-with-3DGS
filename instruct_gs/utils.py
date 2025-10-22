"""
InstructGS Utilities

Helper functions and utilities for InstructGS training and evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import json


def save_edit_comparison(
    original_image: torch.Tensor,
    rendered_image: torch.Tensor, 
    edited_image: torch.Tensor,
    output_path: Path,
    step: int,
    camera_idx: int
):
    """
    Save a comparison image showing original, rendered, and edited versions.
    
    Args:
        original_image: Original training image
        rendered_image: Current 3D render  
        edited_image: InstructPix2Pix edited result
        output_path: Directory to save comparison
        step: Training step number
        camera_idx: Camera index
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to PIL images
    def tensor_to_pil(tensor):
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor = torch.clamp(tensor, 0.0, 1.0)
        numpy_image = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(numpy_image)
    
    orig_pil = tensor_to_pil(original_image)
    rendered_pil = tensor_to_pil(rendered_image)
    edited_pil = tensor_to_pil(edited_image)
    
    # Create side-by-side comparison
    width, height = orig_pil.size
    comparison = Image.new('RGB', (width * 3, height))
    
    comparison.paste(orig_pil, (0, 0))
    comparison.paste(rendered_pil, (width, 0))
    comparison.paste(edited_pil, (width * 2, 0))
    
    # Save comparison
    filename = f"comparison_step_{step:06d}_cam_{camera_idx:03d}.jpg"
    comparison.save(output_path / filename)


def log_idu_statistics(
    step: int,
    num_images_rendered: int,
    num_images_edited: int,
    render_time: float,
    edit_time: float,
    log_path: Optional[Path] = None
):
    """
    Log statistics from an IDU cycle.
    
    Args:
        step: Training step
        num_images_rendered: Number of images successfully rendered
        num_images_edited: Number of images successfully edited
        render_time: Time spent on 3D rendering (seconds)
        edit_time: Time spent on 2D editing (seconds)
        log_path: Optional path to save log file
    """
    stats = {
        "step": step,
        "num_images_rendered": num_images_rendered,
        "num_images_edited": num_images_edited,
        "render_time_seconds": render_time,
        "edit_time_seconds": edit_time,
        "total_time_seconds": render_time + edit_time,
        "images_per_second_render": num_images_rendered / max(render_time, 0.001),
        "images_per_second_edit": num_images_edited / max(edit_time, 0.001),
    }
    
    print(f"IDU Cycle Statistics (Step {step}):")
    print(f"  Rendered: {num_images_rendered} images in {render_time:.1f}s "
          f"({stats['images_per_second_render']:.1f} img/s)")
    print(f"  Edited: {num_images_edited} images in {edit_time:.1f}s "
          f"({stats['images_per_second_edit']:.1f} img/s)")
    print(f"  Total time: {stats['total_time_seconds']:.1f}s")
    
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to log file
        log_entries = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_entries = json.load(f)
        
        log_entries.append(stats)
        
        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)


def estimate_memory_usage(
    num_images: int,
    image_resolution: tuple,
    model_memory_gb: float = 4.0
) -> Dict[str, float]:
    """
    Estimate memory usage for InstructGS training.
    
    Args:
        num_images: Number of training images
        image_resolution: (height, width) tuple
        model_memory_gb: Estimated model memory usage
        
    Returns:
        Dictionary with memory estimates in GB
    """
    h, w = image_resolution
    
    # Image buffer memory (float32, original + edited)
    bytes_per_image = h * w * 3 * 4  # 3 channels, 4 bytes per float32
    buffer_memory_gb = (num_images * 2 * bytes_per_image) / (1024**3)
    
    # InstructPix2Pix memory (rough estimate)
    ip2p_memory_gb = 6.0  # Typical for IP2P model
    
    # Total estimate
    total_memory_gb = model_memory_gb + buffer_memory_gb + ip2p_memory_gb
    
    return {
        "model_memory_gb": model_memory_gb,
        "image_buffer_memory_gb": buffer_memory_gb,
        "ip2p_memory_gb": ip2p_memory_gb,
        "total_estimated_gb": total_memory_gb,
        "recommended_min_vram_gb": total_memory_gb * 1.5,  # Safety margin
    }


def validate_dataset_for_instruct_gs(dataset_path: Path) -> Dict[str, Union[bool, str, int]]:
    """
    Validate that a dataset is suitable for InstructGS training.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Validation results dictionary
    """
    results = {
        "valid": True,
        "messages": [],
        "num_images": 0,
        "has_colmap": False,
        "has_poses": False,
    }
    
    # Check if dataset path exists
    if not dataset_path.exists():
        results["valid"] = False
        results["messages"].append(f"Dataset path does not exist: {dataset_path}")
        return results
    
    # Check for images directory
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        results["valid"] = False
        results["messages"].append("No 'images' directory found")
        return results
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    results["num_images"] = len(image_files)
    
    if results["num_images"] < 10:
        results["messages"].append(f"Very few images found ({results['num_images']}). "
                                 "InstructGS works best with 50+ images.")
    
    # Check for COLMAP sparse reconstruction
    sparse_dir = dataset_path / "sparse" / "0"
    if sparse_dir.exists():
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        if all((sparse_dir / f).exists() for f in required_files):
            results["has_colmap"] = True
        else:
            results["messages"].append("COLMAP sparse directory exists but missing required files")
    else:
        results["messages"].append("No COLMAP sparse reconstruction found")
    
    # Check for poses_bounds.npy (LLFF format)
    poses_file = dataset_path / "poses_bounds.npy"
    if poses_file.exists():
        results["has_poses"] = True
    else:
        results["messages"].append("No poses_bounds.npy found")
    
    # Must have either COLMAP or poses
    if not (results["has_colmap"] or results["has_poses"]):
        results["valid"] = False
        results["messages"].append("Need either COLMAP reconstruction or poses_bounds.npy")
    
    # Check for nerfstudio format
    transforms_file = dataset_path / "transforms.json"
    if transforms_file.exists():
        results["messages"].append("Found transforms.json (nerfstudio format)")
    
    return results


def check_gpu_requirements(ip2p_device: str = "cuda:1") -> Dict[str, Union[bool, str, List[str]]]:
    """
    Check if GPU requirements are met for InstructGS.
    
    Args:
        ip2p_device: Device string for InstructPix2Pix
        
    Returns:
        Requirements check results
    """
    results = {
        "requirements_met": True,
        "messages": [],
        "available_gpus": [],
        "recommended_setup": "",
    }
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            results["requirements_met"] = False
            results["messages"].append("CUDA not available")
            return results
        
        num_gpus = torch.cuda.device_count()
        results["available_gpus"] = [f"cuda:{i}" for i in range(num_gpus)]
        
        # Check memory for each GPU
        for i in range(num_gpus):
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            results["messages"].append(f"GPU {i}: {memory_gb:.1f} GB VRAM")
        
        # Recommendations based on available GPUs
        if num_gpus >= 2:
            results["recommended_setup"] = ("Use cuda:0 for main model, cuda:1 for InstructPix2Pix. "
                                          "Each GPU should have 8+ GB VRAM.")
        elif num_gpus == 1:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if memory_gb >= 16:
                results["recommended_setup"] = ("Single GPU with 16+ GB should work. "
                                               "Use ip2p-device cuda:0")
            else:
                results["requirements_met"] = False
                results["messages"].append(f"Single GPU has {memory_gb:.1f} GB VRAM. "
                                         "Need 16+ GB for single GPU setup.")
        
        # Check if specified IP2P device exists
        if ip2p_device != "cpu":
            device_num = int(ip2p_device.split(":")[-1])
            if device_num >= num_gpus:
                results["requirements_met"] = False
                results["messages"].append(f"Specified IP2P device {ip2p_device} not available. "
                                         f"Only {num_gpus} GPU(s) found.")
        
    except Exception as e:
        results["requirements_met"] = False
        results["messages"].append(f"Error checking GPU requirements: {e}")
    
    return results