#!/usr/bin/env python3
"""
InstructGS - Text-guided 3D Scene Editing with Gaussian Splatting
Main training script that works from the project root directory.
"""

import os
import sys
import argparse
from pathlib import Path

# Add gsplat examples to path for imports
project_root = Path(__file__).parent
gsplat_examples = project_root / "gsplat-src" / "examples"
sys.path.insert(0, str(gsplat_examples))

# Add project root to path so we can import instruct_gs package
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for InstructGS."""
    parser = argparse.ArgumentParser(
        description="InstructGS - Text-guided 3D Scene Editing with Gaussian Splatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode with guided setup"
    )
    
    # Quick setup arguments
    parser.add_argument(
        "--data", 
        type=Path,
        help="Path to dataset (e.g., datasets/360_v2/garden)"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        help="Edit instruction (e.g., 'Turn it into a painting')"
    )
    parser.add_argument(
        "--name", 
        type=str,
        default="instruct_gs_experiment",
        help="Experiment name"
    )
    parser.add_argument(
        "--config", 
        type=Path,
        help="Configuration file path"
    )
    
    # Training parameters
    parser.add_argument(
        "--rounds", 
        type=int,
        default=10,
        help="Maximum number of editing rounds"
    )
    parser.add_argument(
        "--steps", 
        type=int,
        default=2500,
        help="3DGS optimization steps per round"
    )
    parser.add_argument(
        "--mode", 
        choices=["replace", "remove", "restyle"],
        default="restyle",
        help="Editing mode"
    )
    
    # System arguments
    parser.add_argument(
        "--device", 
        default="cuda",
        help="Device for training (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("outputs"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎨 InstructGS: Text-guided 3D Scene Editing with Gaussian Splatting")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not (project_root / "gsplat-src").exists():
        print("❌ Error: gsplat-src not found. Please run from the project root directory.")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Expected to find: {project_root / 'gsplat-src'}")
        sys.exit(1)
    
    # Check if datasets exist
    if not (project_root / "datasets").exists():
        print("⚠️  Warning: datasets directory not found.")
        print("   You may need to download datasets first:")
        print("   cd gsplat-src/examples")
        print("   python datasets/download_dataset.py --dataset mipnerf360 --save-dir ../../datasets")
    
    # Run interactive mode or direct execution
    if args.interactive or len(sys.argv) == 1:
        run_interactive_mode()
    else:
        run_direct_mode(args)


def run_interactive_mode():
    """Run the interactive CLI interface."""
    try:
        # Import from the instruct_gs package using absolute import
        from instruct_gs.cli.interface import InstructGSInterface
        interface = InstructGSInterface()
        interface.run_interactive_mode()
    except ImportError as e:
        print(f"❌ Error: Could not import interactive interface: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install rich pyyaml torch")
        sys.exit(1)


def run_direct_mode(args):
    """Run direct command-line execution."""
    print(f"📝 Experiment: {args.name}")
    
    # Validate required arguments
    if not args.data:
        print("❌ Error: --data argument required for direct mode")
        print("   Example: python train_instruct_gs.py --data datasets/360_v2/garden --prompt 'Turn it into a painting'")
        sys.exit(1)
    
    if not args.prompt:
        print("❌ Error: --prompt argument required for direct mode")
        print("   Example: python train_instruct_gs.py --data datasets/360_v2/garden --prompt 'Turn it into a painting'")
        sys.exit(1)
    
    # Validate data directory
    if not args.data.exists():
        print(f"❌ Error: Dataset directory not found: {args.data}")
        print("   Available datasets:")
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            for dataset in datasets_dir.iterdir():
                if dataset.is_dir():
                    print(f"     {dataset}")
        sys.exit(1)
    
    print(f"📂 Dataset: {args.data}")
    print(f"💬 Prompt: '{args.prompt}'")
    print(f"🎯 Mode: {args.mode}")
    print(f"🔄 Rounds: {args.rounds}")
    print(f"⚙️  Steps per round: {args.steps}")
    
    try:
        # Import and run trainer
        from config.config_manager import load_config_with_overrides
        from core.trainer import InstructGSTrainer
        from utils.path_manager import PathManager
        
        # Create configuration
        config_overrides = {
            "experiment": {"name": args.name},
            "data": {"data_dir": str(args.data)},
            "editing": {
                "edit_prompt": args.prompt,
                "edit_mode": args.mode,
                "max_rounds": args.rounds,
                "cycle_steps": args.steps
            },
            "system": {"device": args.device}
        }
        
        config = load_config_with_overrides(args.config, **config_overrides)
        path_manager = PathManager(config)
        
        # Setup experiment
        path_manager.setup_experiment_dirs()
        config.save(path_manager.get_config_path())
        
        print(f"📁 Output directory: {config.output_dir}")
        
        # Initialize and run trainer
        trainer = InstructGSTrainer(config, path_manager)
        trainer.train()
        
        print("✅ Training completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error: Could not import trainer components: {e}")
        print("   The trainer implementation is not yet complete.")
        print("   Currently available: configuration, CLI interface, path management")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()