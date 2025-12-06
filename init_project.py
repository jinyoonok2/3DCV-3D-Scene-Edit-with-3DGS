#!/usr/bin/env python3
"""
Initialize a new 3DGS Scene Editing project (OPTIONAL HELPER)

This is a convenience tool to quickly create a new configs/garden_config.yaml file.
You can also manually copy and edit configs/garden_config.yaml instead.

The configs/garden_config.yaml file lives in the configs/ directory.
"""

import argparse
import yaml
from pathlib import Path
from project_utils.config import ProjectConfig


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new 3DGS Scene Editing project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick create: brown plant removal (creates/updates config.yaml)
  python init_project.py --scene garden --text "brown plant" --task removal
  
  # Create a second project config (for experiments)
  python init_project.py --scene bicycle --text "old bike" --config experiment2.yaml
  
  # Or just manually edit config.yaml (no need to run this script!)
  nano config.yaml
        """
    )
    
    parser.add_argument('--scene', type=str, default='garden',
                       help='Scene name (e.g., garden, bicycle, kitchen)')
    parser.add_argument('--text', type=str, required=True,
                       help='Object to segment/remove (e.g., "brown plant")')
    parser.add_argument('--task', type=str, choices=['removal', 'replacement'], default='removal',
                       help='Task type: removal or replacement')
    parser.add_argument('--new-text', type=str,
                       help='New object description (for replacement task)')
    parser.add_argument('--dataset', type=str,
                       help='Dataset root path (default: datasets/360_v2/{scene})')
    parser.add_argument('--config', type=str, default='configs/garden_config.yaml',
                       help='Output config file path')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing config file')
    
    args = parser.parse_args()
    
    # Check if config already exists
    config_path = Path(args.config)
    if config_path.exists() and not args.force:
        print(f"❌ Config file already exists: {args.config}")
        print("   Use --force to overwrite, or specify a different --config path")
        return 1
    
    # Generate project name
    text_slug = args.text.replace(' ', '_').lower()
    project_name = f"{args.scene}_{text_slug}_{args.task}"
    
    # Build config
    config = {
        'project': {
            'name': project_name,
            'scene': args.scene,
            'task': args.task,
            'description': f"{args.task.capitalize()} {args.text} from {args.scene} scene"
        },
        'paths': {
            'dataset_root': args.dataset or f"datasets/360_v2/{args.scene}",
            'output_root': f"outputs/${{project.name}}",
            'dataset_check': "00_dataset",
            'initial_training': "01_initial_gs",
            'renders': "02_renders",
            'masks': "03_masks",
            'roi': "04_roi",
            'inpainting': "05_inpainting",
            'object_generation': "06_object_gen",
            'scene_merge': "07_merge",
            'evaluation': "08_evaluation",
            'logs': "logs"
        },
        'dataset': {
            'factor': 4,
            'test_every': 8,
            'seed': 42
        },
        'training': {
            'iterations': 30000,
            'sh_degree': 3,
            'eval_steps': [7000, 30000],
            'save_steps': [7000, 30000]
        },
        'segmentation': {
            'text_prompt': args.text,
            'box_threshold': 0.35,
            'text_threshold': 0.25,
            'nms_threshold': 0.8,
            'sam_model': "sam2_hiera_large"
        },
        'roi': {
            'threshold': 0.3,
            'min_views': 3
        },
        'inpainting': {
            'removal': {
                'roi_threshold': 0.3
            },
            'sdxl': {
                'model': "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                'prompt': f"natural {args.scene} scene without {args.text}",
                'negative_prompt': f"{args.text}, object, artifact, blur",
                'strength': 0.99,
                'guidance_scale': 7.5,
                'num_inference_steps': 50
            },
            'optimization': {
                'iterations': 1000,
                'densification': {
                    'start_iter': 100,
                    'stop_iter': 800,
                    'grad_thresh': 0.0002,
                    'densify_every': 100
                },
                'learning_rates': {
                    'means': 0.00016,
                    'scales': 0.005,
                    'quats': 0.001,
                    'opacities': 0.05,
                    'sh0': 0.0025,
                    'shN': 0.0025
                }
            }
        },
        'replacement': {
            'enabled': args.task == 'replacement',
            'object_generation': {
                'prompt': args.new_text or "new object",
                'triposr': {
                    'model': "stabilityai/TripoSR",
                    'foreground_ratio': 0.85,
                    'mc_resolution': 256
                }
            },
            'merge': {
                'depth_model': "depth-anything/Depth-Anything-V2-Large-hf",
                'alignment_method': "depth",
                'scale_factor': 1.0
            },
            'evaluation': {
                'metrics': ["clip", "lpips", "ssim", "psnr"],
                'clip_model': "ViT-B/32",
                'num_eval_views': 20
            }
        },
        'hardware': {
            'device': "cuda",
            'num_workers': 4,
            'mixed_precision': True
        },
        'logging': {
            'level': "INFO",
            'save_renders': True,
            'save_frequency': 100
        }
    }
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Created project config: {args.config}")
    print(f"{'='*80}\n")
    print(f"Project name: {project_name}")
    print(f"Scene: {args.scene}")
    print(f"Task: {args.task}")
    print(f"Target object: {args.text}")
    if args.new_text:
        print(f"Replacement: {args.new_text}")
    print(f"\nDataset: {config['paths']['dataset_root']}")
    print(f"Output: outputs/{project_name}/")
    
    # Create directory structure
    print(f"\nCreating directory structure...")
    proj_config = ProjectConfig(args.config)
    proj_config.print_structure()
    
    print(f"\n{'='*80}")
    print(f"Next steps:")
    print(f"{'='*80}\n")
    print(f"1. Review/edit config: {args.config}")
    print(f"   nano {args.config}")
    print(f"\n2. Run the pipeline (configs/garden_config.yaml is used by default):")
    if args.config == "configs/garden_config.yaml":
        print(f"   python 00_check_dataset.py")
        print(f"   python 01_train_gs_initial.py")
        print(f"   python 02_render_training_views.py")
        print(f"   ... and so on")
    else:
        print(f"   python 00_check_dataset.py --config {args.config}")
        print(f"   python 01_train_gs_initial.py --config {args.config}")
        print(f"   python 02_render_training_views.py --config {args.config}")
        print(f"   ... and so on")
    print(f"\n3. All outputs go to: outputs/{project_name}/\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
