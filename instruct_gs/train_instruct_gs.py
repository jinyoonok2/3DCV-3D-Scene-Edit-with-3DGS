#!/usr/bin/env python3
"""
InstructGS Training Script

This script provides a command-line interface for training InstructGS models 
for text-to-3D scene editing.

Usage:
    python train_instruct_gs.py --data datasets/bicycle --edit-prompt "Turn it into a painting"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nerfstudio.scripts.train import main as train_main
from instruct_gs.config import get_instruct_gs_config, INSTRUCT_GS_CONFIGS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train InstructGS for text-to-3D editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output arguments
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset (e.g., datasets/bicycle)"
    )
    parser.add_argument(
        "--load-dir",
        type=Path,
        help="Path to pre-trained SplatfactoModel checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for training results"
    )
    
    # InstructGS specific arguments
    parser.add_argument(
        "--edit-prompt",
        type=str,
        default="Turn it into a painting",
        help="Text instruction for editing"
    )
    parser.add_argument(
        "--cycle-steps",
        type=int,
        default=2500,
        help="Number of training steps between IDU cycles"
    )
    parser.add_argument(
        "--ip2p-guidance-scale",
        type=float,
        default=7.5,
        help="Text guidance scale for InstructPix2Pix"
    )
    parser.add_argument(
        "--ip2p-image-guidance-scale",
        type=float,
        default=1.5,
        help="Image guidance scale for InstructPix2Pix"
    )
    parser.add_argument(
        "--ip2p-device",
        type=str,
        default="cuda:1",
        help="Device for InstructPix2Pix model"
    )
    
    # Training arguments
    parser.add_argument(
        "--max-num-iterations",
        type=int,
        default=30000,
        help="Maximum number of training iterations"
    )
    parser.add_argument(
        "--steps-per-save",
        type=int,
        default=2000,
        help="Steps between saving checkpoints"
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=1000,
        help="Steps between evaluation"
    )
    
    # Pre-defined configurations
    parser.add_argument(
        "--config",
        type=str,
        choices=list(INSTRUCT_GS_CONFIGS.keys()),
        help="Use a pre-defined configuration"
    )
    
    # Miscellaneous
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment (default: auto-generated)"
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=7007,
        help="Port for the web viewer"
    )
    parser.add_argument(
        "--disable-viewer",
        action="store_true",
        help="Disable the web viewer"
    )
    
    return parser.parse_args()


def setup_config(args):
    """Setup training configuration from arguments."""
    
    if args.config:
        # Use pre-defined configuration
        config = INSTRUCT_GS_CONFIGS[args.config]
        print(f"Using pre-defined configuration: {args.config}")
    else:
        # Create custom configuration
        config = get_instruct_gs_config(
            edit_prompt=args.edit_prompt,
            cycle_steps=args.cycle_steps,
            ip2p_guidance_scale=args.ip2p_guidance_scale,
            ip2p_image_guidance_scale=args.ip2p_image_guidance_scale,
            ip2p_device=args.ip2p_device,
        )
        print(f"Using custom configuration with prompt: '{args.edit_prompt}'")
    
    # Update configuration with command line arguments
    config.max_num_iterations = args.max_num_iterations
    config.steps_per_save = args.steps_per_save
    config.steps_per_eval_all_images = args.steps_per_eval
    
    # Set data path
    config.pipeline.datamanager.dataparser.data = args.data
    
    # Set load directory if provided
    if args.load_dir:
        config.load_dir = args.load_dir
        print(f"Will load pre-trained model from: {args.load_dir}")
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        # Auto-generate experiment name
        prompt_short = args.edit_prompt.replace(" ", "_").lower()[:20]
        config.experiment_name = f"instruct_gs_{prompt_short}"
    
    # Configure viewer
    if args.disable_viewer:
        config.vis = "tensorboard"
    else:
        config.viewer.websocket_port = args.viewer_port
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("InstructGS: Text-to-3D Scene Editing with Gaussian Splatting")
    print("=" * 60)
    print(f"Data path: {args.data}")
    print(f"Edit prompt: '{args.edit_prompt}'")
    print(f"Cycle steps: {args.cycle_steps}")
    print(f"IP2P device: {args.ip2p_device}")
    print(f"Max iterations: {args.max_num_iterations}")
    
    # Check if data path exists
    if not args.data.exists():
        print(f"Error: Data path does not exist: {args.data}")
        sys.exit(1)
    
    # Check if pre-trained model exists if specified
    if args.load_dir and not args.load_dir.exists():
        print(f"Error: Load directory does not exist: {args.load_dir}")
        sys.exit(1)
    
    # Setup configuration
    config = setup_config(args)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment name: {config.experiment_name}")
    
    if not args.disable_viewer:
        print(f"Viewer will be available at: http://localhost:{args.viewer_port}")
    
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Start training using nerfstudio's training infrastructure
    try:
        # We'll need to create a custom training script or modify this
        # to work with nerfstudio's training system
        print("Training would start here...")
        print("Note: Integration with nerfstudio training system needs to be completed")
        
        # For now, print the configuration that would be used
        print("Configuration summary:")
        print(f"  Pipeline: {type(config.pipeline).__name__}")
        print(f"  Model: {type(config.pipeline.model).__name__}")
        print(f"  Edit prompt: {config.pipeline.model.edit_prompt}")
        print(f"  Cycle steps: {config.pipeline.model.cycle_steps}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()