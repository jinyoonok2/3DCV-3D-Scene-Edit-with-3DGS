"""Unified path management system for InstructGS."""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import datetime


class PathManager:
    """Manages all file paths and directory structure for InstructGS experiments."""
    
    def __init__(self, config):
        self.config = config
        self.base_output_dir = config.output_dir
        
        # Define directory structure
        self.dir_structure = {
            'checkpoints': config.paths.get('checkpoints', 'checkpoints'),
            'renders': config.paths.get('renders', 'renders'),
            'logs': config.paths.get('logs', 'logs'),
            'metrics': config.paths.get('metrics', 'metrics'),
            'artifacts': config.paths.get('artifacts', 'artifacts'),
        }
    
    def setup_experiment_dirs(self):
        """Create the complete directory structure for the experiment."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main subdirectories
        for dir_name in self.dir_structure.values():
            (self.base_output_dir / dir_name).mkdir(exist_ok=True)
        
        # Create round-specific structure
        (self.base_output_dir / "rounds").mkdir(exist_ok=True)
        
        print(f"✓ Created experiment directory structure at: {self.base_output_dir}")
    
    def get_config_path(self) -> Path:
        """Get path for configuration file."""
        return self.base_output_dir / "config.yaml"
    
    def get_checkpoints_dir(self) -> Path:
        """Get checkpoints directory."""
        return self.base_output_dir / self.dir_structure['checkpoints']
    
    def get_renders_dir(self) -> Path:
        """Get renders directory."""
        return self.base_output_dir / self.dir_structure['renders']
    
    def get_logs_dir(self) -> Path:
        """Get logs directory."""
        return self.base_output_dir / self.dir_structure['logs']
    
    def get_metrics_dir(self) -> Path:
        """Get metrics directory."""
        return self.base_output_dir / self.dir_structure['metrics']
    
    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory.""" 
        return self.base_output_dir / self.dir_structure['artifacts']
    
    def get_round_dir(self, round_num: int) -> Path:
        """Get directory for specific round."""
        round_dir = self.base_output_dir / "rounds" / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        return round_dir
    
    def get_round_renders_dir(self, round_num: int) -> Path:
        """Get renders directory for specific round."""
        renders_dir = self.get_round_dir(round_num) / "renders"
        renders_dir.mkdir(exist_ok=True)
        return renders_dir
    
    def get_round_artifacts_dir(self, round_num: int) -> Path:
        """Get artifacts directory for specific round."""
        artifacts_dir = self.get_round_dir(round_num) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        return artifacts_dir
    
    def get_checkpoint_path(self, round_num: Optional[int] = None, name: str = "checkpoint") -> Path:
        """Get checkpoint file path."""
        checkpoints_dir = self.get_checkpoints_dir()
        
        if round_num is not None:
            return checkpoints_dir / f"{name}_round_{round_num:03d}.pt"
        else:
            return checkpoints_dir / f"{name}_latest.pt"
    
    def get_training_state_path(self) -> Path:
        """Get training state file path."""
        return self.base_output_dir / "training_state.yaml"
    
    def get_render_paths(self, round_num: int, view_idx: int) -> Dict[str, Path]:
        """Get render file paths for a specific round and view."""
        render_dir = self.get_round_renders_dir(round_num)
        
        return {
            'pre_edit': render_dir / f"pre_edit_view_{view_idx:03d}.png",
            'edited': render_dir / f"edited_view_{view_idx:03d}.png", 
            'post_opt': render_dir / f"post_opt_view_{view_idx:03d}.png",
            'mask': render_dir / f"mask_view_{view_idx:03d}.png",
        }
    
    def get_metrics_path(self, round_num: int) -> Path:
        """Get metrics file path for specific round."""
        return self.get_metrics_dir() / f"round_{round_num:03d}.yaml"
    
    def get_log_path(self, log_type: str = "training") -> Path:
        """Get log file path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.get_logs_dir() / f"{log_type}_{timestamp}.log"
    
    def get_video_path(self, round_num: Optional[int] = None, video_type: str = "trajectory") -> Path:
        """Get video file path."""
        if round_num is not None:
            return self.get_round_dir(round_num) / f"{video_type}_round_{round_num:03d}.mp4"
        else:
            return self.base_output_dir / f"{video_type}_final.mp4"
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints."""
        checkpoints_dir = self.get_checkpoints_dir()
        if not checkpoints_dir.exists():
            return []
        
        return sorted(checkpoints_dir.glob("*.pt"))
    
    def list_rounds(self) -> List[int]:
        """List all completed rounds."""
        rounds_dir = self.base_output_dir / "rounds"
        if not rounds_dir.exists():
            return []
        
        round_dirs = [d for d in rounds_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]
        round_nums = []
        
        for round_dir in round_dirs:
            try:
                round_num = int(round_dir.name.split("_")[1])
                round_nums.append(round_num)
            except (ValueError, IndexError):
                continue
        
        return sorted(round_nums)
    
    def clean_round(self, round_num: int):
        """Clean up files from a specific round."""
        round_dir = self.get_round_dir(round_num)
        if round_dir.exists():
            shutil.rmtree(round_dir)
        
        # Remove round-specific checkpoint
        checkpoint_path = self.get_checkpoint_path(round_num)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Remove round metrics
        metrics_path = self.get_metrics_path(round_num)
        if metrics_path.exists():
            metrics_path.unlink()
    
    def clean_all_rounds(self):
        """Clean up all round data."""
        rounds_dir = self.base_output_dir / "rounds"
        if rounds_dir.exists():
            shutil.rmtree(rounds_dir)
            rounds_dir.mkdir()
        
        # Clean checkpoints (keep latest)
        checkpoints = self.list_checkpoints()
        for checkpoint in checkpoints:
            if "latest" not in checkpoint.name:
                checkpoint.unlink()
        
        # Clean metrics
        metrics_dir = self.get_metrics_dir()
        if metrics_dir.exists():
            for metrics_file in metrics_dir.glob("round_*.yaml"):
                metrics_file.unlink()
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics for the experiment."""
        def get_dir_size(path: Path) -> float:
            """Get directory size in MB."""
            if not path.exists():
                return 0.0
            
            total = 0
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
            return total / (1024 * 1024)  # Convert to MB
        
        usage = {}
        for name, dir_path in self.dir_structure.items():
            full_path = self.base_output_dir / dir_path
            usage[name] = get_dir_size(full_path)
        
        # Add rounds directory
        usage['rounds'] = get_dir_size(self.base_output_dir / "rounds")
        usage['total'] = sum(usage.values())
        
        return usage
    
    def create_archive(self, archive_path: Optional[Path] = None) -> Path:
        """Create an archive of the entire experiment."""
        if archive_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = self.config.experiment['name']
            archive_path = Path(f"{exp_name}_{timestamp}.tar.gz")
        
        import tarfile
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.base_output_dir, arcname=self.base_output_dir.name)
        
        return archive_path
    
    def __str__(self) -> str:
        """String representation of path structure."""
        lines = [f"InstructGS Experiment: {self.config.experiment['name']}"]
        lines.append(f"Base Directory: {self.base_output_dir}")
        lines.append("Directory Structure:")
        
        for name, path in self.dir_structure.items():
            full_path = self.base_output_dir / path
            exists = "✓" if full_path.exists() else "✗"
            lines.append(f"  {exists} {name}: {path}")
        
        return "\n".join(lines)


# Utility functions for path operations
def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    import re
    # Replace invalid characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name.strip('_')


def get_available_experiments(base_dir: Path = Path("outputs")) -> List[str]:
    """Get list of available experiment names."""
    if not base_dir.exists():
        return []
    
    experiments = []
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
            experiments.append(exp_dir.name)
    
    return sorted(experiments)