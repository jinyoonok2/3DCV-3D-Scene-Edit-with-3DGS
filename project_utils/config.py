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
                self.get_path('scene_placement'),
                self.get_path('final_optimization'),
            ])
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create module-specific subdirectories
            self._create_module_subdirs(dir_path)
            
    def _create_module_subdirs(self, module_path: Path):
        """Create subdirectories within a module based on its type"""
        module_name = module_path.name
        output_config = self.config.get('output', {})
        subdirs = output_config.get('subdirs', {})
        
        # Create renders subdirs for relevant modules (specific modules only, not parent dirs)
        if any(x in module_name for x in ['render', 'training', 'optimization']) and not module_name.endswith('_scene_editing'):
            for subdir in subdirs.get('renders', ['train', 'val']):
                (module_path / 'renders' / subdir).mkdir(parents=True, exist_ok=True)
                
        # Create masks subdirs for mask-related modules
        if 'mask' in module_name or 'roi' in module_name:
            for subdir in subdirs.get('masks', ['sam_masks', 'projected_masks']):
                (module_path / subdir).mkdir(parents=True, exist_ok=True)
                
        # Create model subdirs for training modules
        if any(x in module_name for x in ['training', 'optimization']):
            for subdir in subdirs.get('models', ['checkpoints', 'metrics']):
                (module_path / subdir).mkdir(parents=True, exist_ok=True)
            
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
        
        # Save to central logs directory
        log_path = self.get_path('logs') / f'{module_name}_manifest.json'
        with open(log_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        # Also save locally if enabled
        output_config = self.config.get('output', {})
        if output_config.get('save_manifests_local', True):
            # Try to determine module directory from name
            module_dir = self._get_module_dir_from_name(module_name)
            if module_dir:
                local_manifest = module_dir / 'manifest.json'
                with open(local_manifest, 'w') as f:
                    json.dump(manifest_data, f, indent=2)
                    
        # Create module summary if enabled
        if output_config.get('create_summaries', True):
            self._create_module_summary(module_name, manifest_data)
            
        print(f"✓ Saved manifest: {log_path}")
        
    def get_checkpoint_path(self, stage: str = 'initial') -> Path:
        """
        Get standardized checkpoint path using unified naming
        
        Args:
            stage: Checkpoint stage ('initial', 'holed', 'patched', 'merged', 'final', etc.)
            
        Returns:
            Path to checkpoint
        """
        prefix = self.config.get('output', {}).get('checkpoint_prefix', 'ckpt_')
        
        if stage == 'initial':
            return self.get_path('initial_training') / f'{prefix}initial.pt'
        elif stage == 'holed':
            return self.get_path('inpainting') / f'{prefix}holed.pt'
        elif stage == 'patched':
            return self.get_path('inpainting') / f'{prefix}patched.pt'
        elif stage == 'merged':
            return self.get_path('scene_placement') / f'{prefix}merged.pt'
        elif stage == 'final':
            return self.get_path('final_optimization') / f'{prefix}final.pt'
        else:
            # Generic checkpoint in appropriate module
            return self.get_path('initial_training') / f'{prefix}{stage}.pt'
    
    def _get_module_dir_from_name(self, module_name: str):
        """Get module directory path from module name"""
        # Map module names to config path keys
        module_mapping = {
            '00_check_dataset': 'dataset_check',
            '01_train_gs_initial': 'initial_training',
            '02_render_training_views': 'renders',
            '03_ground_text_to_masks': 'masks',
            '04a_lift_masks_to_roi3d': 'roi',
            '04b_visualize_roi': 'roi',
            '05a_remove_and_render_holes': 'inpainting',
            '05b_inpaint_holes': 'inpainting',
            '05c_optimize_to_targets': 'inpainting',
            '06_object_generation': 'object_generation',
            '07_place_object_at_roi': 'scene_placement',
            '08_final_optimization': 'final_optimization',
        }
        
        path_key = module_mapping.get(module_name)
        return self.get_path(path_key) if path_key else None
        
    def _create_module_summary(self, module_name: str, manifest_data: Dict):
        """Create a README.md summary for the module"""
        module_dir = self._get_module_dir_from_name(module_name)
        if not module_dir:
            return
            
        readme_path = module_dir / 'README.md'
        
        # Generate summary content
        content = f"# {module_name.replace('_', ' ').title()}\n\n"
        content += f"**Generated:** {manifest_data.get('timestamp', 'Unknown')}\n\n"
        
        if 'inputs' in manifest_data:
            content += "## Inputs\n"
            for key, value in manifest_data['inputs'].items():
                content += f"- **{key}**: `{value}`\n"
            content += "\n"
            
        if 'parameters' in manifest_data:
            content += "## Parameters\n"
            for key, value in manifest_data['parameters'].items():
                content += f"- **{key}**: {value}\n"
            content += "\n"
            
        if 'outputs' in manifest_data:
            content += "## Outputs\n"
            for key, value in manifest_data['outputs'].items():
                if isinstance(value, list):
                    content += f"- **{key}**: {len(value)} files\n"
                else:
                    content += f"- **{key}**: `{value}`\n"
            content += "\n"
            
        if 'metrics' in manifest_data or 'results' in manifest_data:
            content += "## Results\n"
            results = manifest_data.get('metrics', manifest_data.get('results', {}))
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    content += f"- **{key}**: {value:.4f}\n"
                else:
                    content += f"- **{key}**: {value}\n"
            content += "\n"
            
        # List actual files in directory
        content += "## Files in Directory\n"
        try:
            for item in sorted(module_dir.iterdir()):
                if item.name != 'README.md':
                    if item.is_dir():
                        file_count = len(list(item.rglob('*.*')))
                        content += f"- 📁 `{item.name}/` ({file_count} files)\n"
                    else:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        content += f"- 📄 `{item.name}` ({size_mb:.1f} MB)\n"
        except Exception:
            content += "Directory listing unavailable\n"
            
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def get_render_filename(self, index: int) -> str:
        """
        Get standardized render filename
        
        Args:
            index: View index
        """
        render_format = self.config.get('output', {}).get('render_format', '05d')
        return f"{index:{render_format}}.png"

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
