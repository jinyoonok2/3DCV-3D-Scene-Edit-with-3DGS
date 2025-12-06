"""
Base Phase Class - Foundation for all pipeline phases

Provides common functionality:
- Configuration management
- Input/output validation
- Progress tracking
- Error handling
- Logging and metadata
"""

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class BasePhase(ABC):
    """Base class for all pipeline phases."""
    
    def __init__(self, config, phase_name: str, phase_number: int):
        """
        Initialize a pipeline phase.
        
        Args:
            config: ProjectConfig instance
            phase_name: Human-readable name (e.g., "Training")
            phase_number: Phase number (1-4)
        """
        self.config = config
        self.phase_name = phase_name
        self.phase_number = phase_number
        self.project_name = config.get("project", "name")
        
        # Setup output directories
        self.output_root = Path(config.get("paths", "output_root").replace("${project.name}", self.project_name))
        self.phase_dir = self.output_root / f"phase{phase_number}_{phase_name.lower()}"
        self.logs_dir = self.output_root / "logs"
        
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.end_time = None
        self.metadata = {}
    
    def print_header(self):
        """Print phase header."""
        console.print("\n" + "=" * 80)
        console.print(f"Phase {self.phase_number}: {self.phase_name}")
        console.print(f"Project: {self.project_name}")
        console.print("=" * 80 + "\n")
    
    def print_footer(self):
        """Print phase footer with timing."""
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            console.print(f"\n✓ Phase {self.phase_number} completed in {duration:.1f}s\n")
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """
        Validate that all required inputs exist.
        
        Returns:
            bool: True if all inputs are valid
        """
        pass
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the phase logic.
        
        Returns:
            dict: Phase results/outputs
        """
        pass
    
    def save_metadata(self, results: Dict[str, Any]):
        """Save phase metadata to JSON."""
        metadata = {
            "phase": self.phase_number,
            "phase_name": self.phase_name,
            "project": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "config_file": str(self.config.config_path),
            "results": results,
            **self.metadata
        }
        
        metadata_path = self.logs_dir / f"phase{self.phase_number}_manifest.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"[dim]Metadata saved to: {metadata_path}[/dim]")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete phase pipeline.
        
        Returns:
            dict: Phase results
        """
        self.print_header()
        self.start_time = datetime.now()
        
        try:
            # Validate inputs
            console.print("[bold]Validating inputs...[/bold]")
            if not self.validate_inputs():
                console.print("[red]Input validation failed![/red]")
                sys.exit(1)
            console.print("✓ All inputs validated\n")
            
            # Execute phase
            console.print(f"[bold]Executing {self.phase_name}...[/bold]\n")
            results = self.execute()
            
            # Save metadata
            self.end_time = datetime.now()
            self.save_metadata(results)
            
            self.print_footer()
            return results
            
        except Exception as e:
            console.print(f"[red]Error in Phase {self.phase_number}:[/red] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def get_checkpoint_path(self, name: str) -> Path:
        """Get path for a checkpoint file in this phase."""
        return self.phase_dir / f"{name}.pt"
    
    def get_previous_phase_dir(self, phase_num: int) -> Optional[Path]:
        """Get directory of a previous phase."""
        if phase_num < 1 or phase_num >= self.phase_number:
            return None
        
        # Map phase numbers to names
        phase_names = {
            1: "training",
            2: "segmentation",
            3: "removal",
            4: "placement"
        }
        
        prev_dir = self.output_root / f"phase{phase_num}_{phase_names[phase_num]}"
        return prev_dir if prev_dir.exists() else None
