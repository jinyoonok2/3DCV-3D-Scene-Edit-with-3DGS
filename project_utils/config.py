"""
Configuration management for 3DGS Scene Editing Pipeline
Handles loading YAML configs and managing unified output structure
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class ProjectConfig:
    """Manages project configuration and directory structure"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config YAML file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Resolve project name in paths
        self._resolve_paths()
        
        # Create directory structure
        self._create_directories()
        
    def _resolve_paths(self):
        """Resolve ${project.name} variables in paths"""
        project_name = self.config['project']['name']
        
        # Resolve output_root
        output_root = self.config['paths']['output_root']
        output_root = output_root.replace('${project.name}', project_name)
        self.config['paths']['output_root'] = output_root
        
    def _create_directories(self):
        """Create output directory structure"""
        output_root = Path(self.config['paths']['output_root'])
        
        # Create all module directories
        dirs_to_create = [
            output_root,
            self.get_path('dataset_check'),
            self.get_path('initial_training'),
            self.get_path('renders'),
            self.get_path('masks'),
            self.get_path('roi'),
            self.get_path('inpainting'),
            self.get_path('logs'),
        ]
        
        # Create replacement pipeline dirs if enabled
        if self.config.get('replacement', {}).get('enabled', False):
            dirs_to_create.extend([
                self.get_path('object_generation'),
                self.get_path('scene_merge'),
                self.get_path('evaluation'),
            ])
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_path(self, key: str, *subpaths) -> Path:
        """
        Get absolute path for a configured directory
        
        Args:
            key: Key from paths section in config
            *subpaths: Additional path components to join
            
        Returns:
            Absolute Path object
        """
        output_root = Path(self.config['paths']['output_root'])
        
        if key == 'dataset_root':
            base_path = Path(self.config['paths']['dataset_root'])
        elif key == 'output_root':
            base_path = output_root
        else:
            # Module-specific path
            rel_path = self.config['paths'].get(key, key)
            base_path = output_root / rel_path
            
        # Join with any subpaths
        for subpath in subpaths:
            base_path = base_path / subpath
            
        return base_path.resolve()
    
    def get(self, *keys, default=None):
        """
        Get nested config value using dot notation
        
        Args:
            *keys: Nested keys to traverse
            default: Default value if key not found
            
        Returns:
            Config value or default
            
        Example:
            config.get('training', 'iterations')  # Returns 30000
            config.get('segmentation', 'text_prompt')  # Returns "brown plant"
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """
        Set nested config value
        
        Args:
            *keys: Nested keys to traverse (last one is key to set)
            value: Value to set
            
        Example:
            config.set('segmentation', 'text_prompt', value='red flower')
        """
        target = self.config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    
    def save_manifest(self, module_name: str, manifest_data: Dict):
        """
        Save manifest for a completed module
        
        Args:
            module_name: Name of the module (e.g., "03_masks")
            manifest_data: Dictionary containing all manifest information
        """
        # Add project info if not present
        if 'project' not in manifest_data:
            manifest_data['project'] = self.config['project']
        
        # Save to logs directory
        log_path = self.get_path('logs') / f'{module_name}_manifest.json'
        with open(log_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
            
        print(f"✓ Saved manifest: {log_path}")
        
    def get_checkpoint_path(self, stage: str = 'initial') -> Path:
        """
        Get path to checkpoint for a specific stage
        
        Args:
            stage: Stage name (initial, holed, final)
            
        Returns:
            Path to checkpoint
        """
        if stage == 'initial':
            return self.get_path('initial_training', 'ckpt_initial.pt')
        elif stage == 'holed':
            return self.get_path('inpainting', 'holed', 'ckpt_holed.pt')
        elif stage == 'final':
            return self.get_path('inpainting', 'optimized', 'ckpt_final.pt')
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def print_structure(self):
        """Print the output directory structure"""
        output_root = Path(self.config['paths']['output_root'])
        print(f"\n{'='*80}")
        print(f"Project: {self.config['project']['name']}")
        print(f"Output Root: {output_root}")
        print(f"{'='*80}\n")
        
        print("Directory Structure:")
        for path_key, path_value in self.config['paths'].items():
            if path_key in ['dataset_root', 'output_root']:
                continue
            full_path = self.get_path(path_key)
            exists = "✓" if full_path.exists() else "✗"
            print(f"  {exists} {path_key:20} -> {full_path}")
        print()


def create_default_config(output_path: str = "config.yaml", 
                         project_name: str = "my_project",
                         scene: str = "garden",
                         text_prompt: str = "object"):
    """
    Create a default config file
    
    Args:
        output_path: Where to save config
        project_name: Project name
        scene: Scene name
        text_prompt: Object to segment
    """
    config_template = {
        'project': {
            'name': project_name,
            'scene': scene,
            'task': 'removal',
            'description': f'Remove {text_prompt} from {scene} scene'
        },
        'paths': {
            'dataset_root': f'datasets/360_v2/{scene}',
            'output_root': f'outputs/${{project.name}}',
        },
        'segmentation': {
            'text_prompt': text_prompt,
        },
        # Add other defaults...
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created default config: {output_path}")


if __name__ == "__main__":
    # Test config loading
    config = ProjectConfig("config.yaml")
    config.print_structure()
    
    # Test path retrieval
    print("Example paths:")
    print(f"  Initial checkpoint: {config.get_checkpoint_path('initial')}")
    print(f"  ROI directory: {config.get_path('roi')}")
    print(f"  Masks directory: {config.get_path('masks', 'sam')}")
