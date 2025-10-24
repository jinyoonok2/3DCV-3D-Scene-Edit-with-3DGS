"""Configuration management for InstructGS."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class InstructGSConfig:
    """Main configuration class for InstructGS."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        # Flatten nested config for easier access
        self.experiment = config_dict.get('experiment', {})
        self.data = config_dict.get('data', {})
        self.editing = config_dict.get('editing', {})
        self.diffusion = config_dict.get('diffusion', {})
        self.gaussian = config_dict.get('gaussian', {})
        self.optimization = config_dict.get('optimization', {})
        self.losses = config_dict.get('losses', {})
        self.roi = config_dict.get('roi', {})
        self.evaluation = config_dict.get('evaluation', {})
        self.system = config_dict.get('system', {})
        self.logging = config_dict.get('logging', {})
        self.paths = config_dict.get('paths', {})
        
        # Store original dict for saving
        self._config_dict = config_dict
    
    @property
    def device(self) -> torch.device:
        """Get PyTorch device."""
        device_str = self.system.get('device', 'cuda')
        return torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    @property
    def output_dir(self) -> Path:
        """Get experiment output directory."""
        base_dir = Path(self.experiment.get('output_dir', 'outputs'))
        exp_name = self.experiment.get('name', 'instruct_gs_experiment')
        return base_dir / exp_name
    
    def get_checkpoint_path(self, round_num: Optional[int] = None) -> Path:
        """Get checkpoint path for specific round or latest."""
        checkpoint_dir = self.output_dir / self.paths.get('checkpoints', 'checkpoints')
        if round_num is not None:
            return checkpoint_dir / f"round_{round_num:03d}.pt"
        else:
            return checkpoint_dir / "latest.pt"
    
    def get_round_dir(self, round_num: int) -> Path:
        """Get directory for specific round artifacts."""
        return self.output_dir / "rounds" / f"round_{round_num:03d}"
    
    def save(self, path: Path):
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self._config_dict, updates)
        # Reinitialize from updated dict
        self.__init__(self._config_dict)


class ConfigManager:
    """Manager for loading and validating InstructGS configurations."""
    
    @staticmethod
    def load_config(config_path: Optional[Path] = None) -> InstructGSConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            # Load default config
            config_path = Path(__file__).parent / "base_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate configuration
        ConfigManager.validate_config(config_dict)
        
        return InstructGSConfig(config_dict)
    
    @staticmethod
    def create_config_from_args(**kwargs) -> InstructGSConfig:
        """Create configuration from command line arguments."""
        # Load base config
        base_config = ConfigManager.load_config()
        
        # Convert flat kwargs to nested dict
        updates = {}
        for key, value in kwargs.items():
            if value is not None:
                # Parse nested keys like "data.data_dir" 
                parts = key.split('.')
                nested_dict = updates
                for part in parts[:-1]:
                    nested_dict = nested_dict.setdefault(part, {})
                nested_dict[parts[-1]] = value
        
        # Update base config
        base_config.update(updates)
        return base_config
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any]):
        """Validate configuration dictionary."""
        required_sections = [
            'experiment', 'data', 'editing', 'diffusion', 
            'gaussian', 'optimization', 'losses'
        ]
        
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate data directory exists
        data_dir = Path(config_dict['data']['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Validate edit mode
        valid_modes = ['replace', 'remove', 'restyle']
        edit_mode = config_dict['editing']['edit_mode']
        if edit_mode not in valid_modes:
            raise ValueError(f"Invalid edit mode: {edit_mode}. Must be one of {valid_modes}")
        
        # Validate device
        device = config_dict.get('system', {}).get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Using CPU.")
        
        print("✓ Configuration validation passed")


def load_config_with_overrides(config_path: Optional[Path] = None, **overrides) -> InstructGSConfig:
    """Load configuration with command line overrides."""
    config = ConfigManager.load_config(config_path)
    
    if overrides:
        config.update(overrides)
    
    return config


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager.load_config()
    print("Configuration loaded successfully!")
    print(f"Experiment: {config.experiment['name']}")
    print(f"Data directory: {config.data['data_dir']}")
    print(f"Edit prompt: '{config.editing['edit_prompt']}'")
    print(f"Output directory: {config.output_dir}")