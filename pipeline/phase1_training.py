"""
Phase 1: Training - Dataset Validation + Initial 3DGS Training

Combines functionality from:
- legacy/00_check_dataset.py (dataset validation)
- legacy/01_train_gs_initial.py (initial GS training)

This phase:
1. Validates dataset paths and structure
2. Loads COLMAP data
3. Trains initial 3D Gaussian Splatting model
4. Saves checkpoint and metrics

Outputs:
- phase1_training/
  ├── ckpt_initial.pt          # Trained GS model
  ├── dataset_summary.txt       # Dataset statistics
  ├── metrics.json             # Training metrics
  ├── renders/                 # Sample renders
  └── thumbnails/              # Dataset preview images
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add gsplat to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gsplat-src" / "examples"))

from .base import BasePhase


class Phase1Training(BasePhase):
    """Phase 1: Dataset validation and initial GS training."""
    
    def __init__(self, config):
        super().__init__(config, phase_name="training", phase_number=1)
        
        # Get training config
        self.dataset_root = Path(config.get("paths", "dataset_root"))
        self.factor = config.get("dataset", "factor", 4)
        self.test_every = config.get("dataset", "test_every", 8)
        self.seed = config.get("dataset", "seed", 42)
        
        self.iterations = config.get("training", "iterations", 30000)
        self.sh_degree = config.get("training", "sh_degree", 3)
        self.eval_steps = config.get("training", "eval_steps", [7000, 30000])
        self.save_steps = config.get("training", "save_steps", [7000, 30000])
        
        self.metadata.update({
            "dataset_root": str(self.dataset_root),
            "factor": self.factor,
            "iterations": self.iterations,
            "sh_degree": self.sh_degree,
        })
    
    def validate_inputs(self) -> bool:
        """Validate that dataset exists."""
        if not self.dataset_root.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]Dataset not found:[/red] {self.dataset_root}")
            return False
        
        # Check for required COLMAP files
        sparse_dir = self.dataset_root / "sparse" / "0"
        if not sparse_dir.exists():
            from rich.console import Console
            console = Console()
            console.print(f"[red]COLMAP sparse reconstruction not found:[/red] {sparse_dir}")
            return False
        
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute Phase 1: Dataset validation + training."""
        from rich.console import Console
        console = Console()
        
        # Import here to avoid loading heavy dependencies at module import time
        import subprocess
        
        console.print("[bold cyan]Step 1/2: Validating dataset...[/bold cyan]")
        
        # Run legacy dataset validation script with proper environment
        legacy_check = Path(__file__).parent.parent / "legacy" / "00_check_dataset.py"
        
        # Set up environment to include gsplat path AND project root
        import os
        env = os.environ.copy()
        project_root = str(Path(__file__).parent.parent)
        gsplat_path = str(Path(__file__).parent.parent / "gsplat-src" / "examples")
        
        # Build PYTHONPATH: project_root:gsplat_path:existing
        existing_path = env.get('PYTHONPATH', '')
        paths = [project_root, gsplat_path]
        if existing_path:
            paths.append(existing_path)
        env['PYTHONPATH'] = ':'.join(paths)
        
        cmd_check = [
            sys.executable,
            str(legacy_check),
            "--config", str(self.config.config_path),
            "--output_dir", str(self.phase_dir / "dataset_validation")
        ]
        
        result = subprocess.run(cmd_check, capture_output=True, text=True, env=env, cwd=project_root)
        if result.returncode != 0:
            console.print(f"[red]Dataset validation failed:[/red]\n{result.stderr}")
            raise RuntimeError("Dataset validation failed")
        
        console.print("✓ Dataset validated\n")
        
        console.print("[bold cyan]Step 2/2: Training initial 3DGS model...[/bold cyan]")
        
        # Run legacy training script
        legacy_train = Path(__file__).parent.parent / "legacy" / "01_train_gs_initial.py"
        cmd_train = [
            sys.executable,
            str(legacy_train),
            "--config", str(self.config.config_path),
            "--output_dir", str(self.phase_dir)
        ]
        
        result = subprocess.run(cmd_train, capture_output=True, text=True, env=env, cwd=project_root)
        if result.returncode != 0:
            console.print(f"[red]Training failed:[/red]\n{result.stderr}")
            raise RuntimeError("Training failed")
        
        console.print("✓ Training completed\n")
        
        # Check outputs
        ckpt_path = self.get_checkpoint_path("ckpt_initial")
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not created: {ckpt_path}")
        
        results = {
            "checkpoint": str(ckpt_path),
            "dataset_root": str(self.dataset_root),
            "num_iterations": self.iterations,
        }
        
        return results
