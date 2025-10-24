"""Interactive CLI interface for InstructGS training."""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt
import yaml

from ..config.config_manager import ConfigManager, load_config_with_overrides
from ..core.trainer import InstructGSTrainer
from ..core.state_manager import TrainingStateManager
from ..utils.path_manager import PathManager


class InstructGSInterface:
    """Interactive command-line interface for InstructGS."""
    
    def __init__(self):
        self.console = Console()
        self.trainer: Optional[InstructGSTrainer] = None
        self.config: Optional[Any] = None
        self.state_manager: Optional[TrainingStateManager] = None
        self.path_manager: Optional[PathManager] = None
        
    def print_banner(self):
        """Print InstructGS banner."""
        banner = """
[bold blue]╔══════════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║                        InstructGS                            ║[/bold blue]
[bold blue]║            Text-Guided 3D Scene Editing                      ║[/bold blue]
[bold blue]║                with Gaussian Splatting                       ║[/bold blue]
[bold blue]╚══════════════════════════════════════════════════════════════╝[/bold blue]
"""
        self.console.print(banner)
    
    def run_interactive_mode(self):
        """Run interactive mode with menu system."""
        self.print_banner()
        
        while True:
            self.print_main_menu()
            choice = Prompt.ask(
                "\n[bold cyan]Choose an option[/bold cyan]", 
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"],
                default="1"
            )
            
            try:
                if choice == "1":
                    self.setup_experiment()
                elif choice == "2":
                    self.load_experiment()
                elif choice == "3":
                    self.start_training()
                elif choice == "4":
                    self.resume_training()
                elif choice == "5":
                    self.view_progress()
                elif choice == "6":
                    self.modify_config()
                elif choice == "7":
                    self.reset_experiment()
                elif choice == "8":
                    self.export_results()
                elif choice == "9":
                    self.show_help()
                elif choice == "q":
                    self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
    
    def print_main_menu(self):
        """Print the main menu options."""
        if self.config:
            status = f"[green]Loaded: {self.config.experiment['name']}[/green]"
        else:
            status = "[yellow]No experiment loaded[/yellow]"
        
        menu = Table(title="Main Menu", show_header=False, box=None)
        menu.add_column("Option", style="cyan", width=3)
        menu.add_column("Description", style="white")
        
        menu.add_row("1", "🚀 Setup New Experiment")
        menu.add_row("2", "📂 Load Existing Experiment")
        menu.add_row("3", "▶️  Start Training")
        menu.add_row("4", "⏯️  Resume Training")
        menu.add_row("5", "📊 View Progress")
        menu.add_row("6", "⚙️  Modify Configuration")
        menu.add_row("7", "🔄 Reset Experiment")
        menu.add_row("8", "💾 Export Results")
        menu.add_row("9", "❓ Help")
        menu.add_row("q", "🚪 Quit")
        
        self.console.print(f"\n{status}")
        self.console.print(menu)
    
    def setup_experiment(self):
        """Setup a new experiment with guided configuration."""
        self.console.print("\n[bold green]🚀 Setting up new experiment[/bold green]")
        
        # Get basic experiment info
        exp_name = Prompt.ask("Experiment name", default="my_instruct_gs")
        data_dir = Prompt.ask("Dataset directory", default="datasets/360_v2/garden")
        edit_prompt = Prompt.ask("Edit instruction", default="Turn it into a painting")
        
        # Validate data directory
        if not Path(data_dir).exists():
            self.console.print(f"[red]Error: Dataset directory not found: {data_dir}[/red]")
            return
        
        # Create configuration overrides
        overrides = {
            "experiment": {"name": exp_name},
            "data": {"data_dir": data_dir},
            "editing": {"edit_prompt": edit_prompt}
        }
        
        # Load config with overrides
        try:
            self.config = load_config_with_overrides(**overrides)
            self.path_manager = PathManager(self.config)
            self.state_manager = TrainingStateManager(self.config, self.path_manager)
            
            # Create experiment directory structure
            self.path_manager.setup_experiment_dirs()
            
            # Save config
            self.config.save(self.path_manager.get_config_path())
            
            self.console.print(f"[green]✓ Experiment '{exp_name}' created successfully![/green]")
            self.console.print(f"Output directory: {self.config.output_dir}")
            
        except Exception as e:
            self.console.print(f"[red]Error setting up experiment: {e}[/red]")
    
    def load_experiment(self):
        """Load an existing experiment."""
        self.console.print("\n[bold blue]📂 Loading existing experiment[/bold blue]")
        
        # List available experiments
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            self.console.print("[yellow]No experiments found in outputs directory.[/yellow]")
            return
        
        experiments = [d for d in outputs_dir.iterdir() if d.is_dir()]
        if not experiments:
            self.console.print("[yellow]No experiments found.[/yellow]")
            return
        
        # Show available experiments
        table = Table(title="Available Experiments")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Last Modified")
        
        for exp_dir in experiments:
            config_file = exp_dir / "config.yaml"
            if config_file.exists():
                status = "✓ Complete"
                modified = time.ctime(config_file.stat().st_mtime)
            else:
                status = "⚠ Incomplete"  
                modified = "Unknown"
            table.add_row(exp_dir.name, status, modified)
        
        self.console.print(table)
        
        # Select experiment
        exp_name = Prompt.ask("Enter experiment name to load")
        config_path = outputs_dir / exp_name / "config.yaml"
        
        if not config_path.exists():
            self.console.print(f"[red]Config file not found: {config_path}[/red]")
            return
        
        try:
            self.config = ConfigManager.load_config(config_path)
            self.path_manager = PathManager(self.config)
            self.state_manager = TrainingStateManager(self.config, self.path_manager)
            
            self.console.print(f"[green]✓ Loaded experiment: {exp_name}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error loading experiment: {e}[/red]")
    
    def start_training(self):
        """Start training from scratch."""
        if not self.config:
            self.console.print("[red]No experiment loaded. Please setup or load an experiment first.[/red]")
            return
        
        self.console.print("\n[bold green]▶️ Starting training[/bold green]")
        
        # Check if training already exists
        if self.state_manager.has_existing_training():
            if not Confirm.ask("Training already exists. Reset and start fresh?"):
                return
            self.state_manager.reset_training()
        
        try:
            # Initialize trainer
            self.trainer = InstructGSTrainer(self.config, self.path_manager)
            
            # Start training with progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Initializing training...", total=None)
                
                # Run training (this will be the main training loop)
                self.trainer.train()
                
            self.console.print("[green]✓ Training completed successfully![/green]")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted by user.[/yellow]")
            if self.trainer:
                self.trainer.save_checkpoint("interrupted")
        except Exception as e:
            self.console.print(f"[red]Training failed: {e}[/red]")
    
    def resume_training(self):
        """Resume training from checkpoint."""
        if not self.config:
            self.console.print("[red]No experiment loaded.[/red]")
            return
        
        if not self.state_manager.has_existing_training():
            self.console.print("[yellow]No existing training found. Use 'Start Training' instead.[/yellow]")
            return
        
        self.console.print("\n[bold yellow]⏯️ Resuming training[/bold yellow]")
        
        try:
            # Load training state
            state = self.state_manager.load_training_state()
            current_round = state.get('current_round', 0)
            
            self.console.print(f"Resuming from round {current_round}")
            
            # Initialize trainer and resume
            self.trainer = InstructGSTrainer(self.config, self.path_manager)
            self.trainer.load_checkpoint(state['checkpoint_path'])
            self.trainer.train(start_round=current_round)
            
            self.console.print("[green]✓ Training resumed successfully![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error resuming training: {e}[/red]")
    
    def view_progress(self):
        """View training progress and metrics."""
        if not self.config:
            self.console.print("[red]No experiment loaded.[/red]")
            return
        
        self.console.print("\n[bold blue]📊 Training Progress[/bold blue]")
        
        # Show training state
        if self.state_manager.has_existing_training():
            state = self.state_manager.load_training_state()
            
            # Create progress table
            table = Table(title="Training Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Current Round", str(state.get('current_round', 0)))
            table.add_row("Total Rounds", str(self.config.editing['max_rounds']))
            table.add_row("Steps per Round", str(self.config.editing['cycle_steps']))
            table.add_row("Last Updated", str(state.get('last_updated', 'Unknown')))
            
            self.console.print(table)
            
            # Show round metrics if available
            self.show_round_metrics()
        else:
            self.console.print("[yellow]No training progress found.[/yellow]")
    
    def show_round_metrics(self):
        """Show metrics from completed rounds."""
        metrics_dir = self.path_manager.get_metrics_dir()
        if not metrics_dir.exists():
            return
        
        metric_files = list(metrics_dir.glob("round_*.yaml"))
        if not metric_files:
            return
        
        # Load and display recent metrics
        table = Table(title="Recent Round Metrics")
        table.add_column("Round", style="cyan")
        table.add_column("PSNR", style="green")
        table.add_column("SSIM", style="green") 
        table.add_column("LPIPS", style="green")
        
        for metric_file in sorted(metric_files)[-5:]:  # Show last 5 rounds
            with open(metric_file, 'r') as f:
                metrics = yaml.safe_load(f)
            
            round_num = metric_file.stem.split('_')[1]
            psnr = f"{metrics.get('psnr', 0):.2f}"
            ssim = f"{metrics.get('ssim', 0):.3f}"
            lpips = f"{metrics.get('lpips', 0):.3f}"
            
            table.add_row(round_num, psnr, ssim, lpips)
        
        self.console.print(table)
    
    def modify_config(self):
        """Modify experiment configuration interactively."""
        if not self.config:
            self.console.print("[red]No experiment loaded.[/red]")
            return
        
        self.console.print("\n[bold yellow]⚙️ Configuration Editor[/bold yellow]")
        
        # Show current key settings
        current_settings = Table(title="Current Settings")
        current_settings.add_column("Setting", style="cyan")
        current_settings.add_column("Value", style="white")
        
        current_settings.add_row("Edit Prompt", self.config.editing['edit_prompt'])
        current_settings.add_row("Edit Mode", self.config.editing['edit_mode'])
        current_settings.add_row("Cycle Steps", str(self.config.editing['cycle_steps']))
        current_settings.add_row("Max Rounds", str(self.config.editing['max_rounds']))
        
        self.console.print(current_settings)
        
        # Allow modifications
        if Confirm.ask("Modify edit prompt?"):
            new_prompt = Prompt.ask("New edit prompt", default=self.config.editing['edit_prompt'])
            self.config.editing['edit_prompt'] = new_prompt
        
        if Confirm.ask("Modify training parameters?"):
            new_cycles = IntPrompt.ask("Cycle steps", default=self.config.editing['cycle_steps'])
            new_rounds = IntPrompt.ask("Max rounds", default=self.config.editing['max_rounds'])
            
            self.config.editing['cycle_steps'] = new_cycles
            self.config.editing['max_rounds'] = new_rounds
        
        # Save updated config
        self.config.save(self.path_manager.get_config_path())
        self.console.print("[green]✓ Configuration updated![/green]")
    
    def reset_experiment(self):
        """Reset experiment to start over."""
        if not self.config:
            self.console.print("[red]No experiment loaded.[/red]")
            return
        
        if not Confirm.ask("[red]Are you sure you want to reset all training progress?[/red]"):
            return
        
        try:
            self.state_manager.reset_training()
            self.console.print("[green]✓ Experiment reset successfully![/green]")
        except Exception as e:
            self.console.print(f"[red]Error resetting experiment: {e}[/red]")
    
    def export_results(self):
        """Export training results."""
        if not self.config:
            self.console.print("[red]No experiment loaded.[/red]")
            return
        
        self.console.print("\n[bold green]💾 Exporting results[/bold green]")
        
        export_dir = self.config.output_dir / "export"
        export_dir.mkdir(exist_ok=True)
        
        # TODO: Implement result export (renders, metrics, final model)
        self.console.print(f"[green]Results would be exported to: {export_dir}[/green]")
    
    def show_help(self):
        """Show help information."""
        help_text = """
[bold cyan]InstructGS Help[/bold cyan]

[bold]Getting Started:[/bold]
1. Setup a new experiment or load an existing one
2. Configure your edit prompt and parameters  
3. Start training to begin the round-based editing process

[bold]Training Process:[/bold]
- Each round renders the current 3D scene
- Applies 2D diffusion edits to the renders  
- Optimizes the 3D Gaussians to match edited targets
- Saves progress and artifacts for each round

[bold]Key Features:[/bold]
- Resume training from any checkpoint
- Modify configuration during training
- Real-time progress monitoring
- Automatic result organization

[bold]Tips:[/bold]
- Use descriptive edit prompts for better results
- Monitor metrics to track editing progress
- Experiment with different cycle steps and rounds
- Save important results before resetting
"""
        panel = Panel(help_text, title="Help", border_style="blue")
        self.console.print(panel)


def main():
    """Main entry point for InstructGS CLI."""
    parser = argparse.ArgumentParser(description="InstructGS - Text-guided 3D Scene Editing")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--data", type=str, help="Dataset directory")
    parser.add_argument("--prompt", type=str, help="Edit instruction")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--rounds", type=int, help="Maximum rounds")
    parser.add_argument("--steps", type=int, help="Steps per round")
    
    args = parser.parse_args()
    
    if args.interactive or len(sys.argv) == 1:
        # Run interactive mode
        interface = InstructGSInterface()
        interface.run_interactive_mode()
    else:
        # Direct command line execution
        # TODO: Implement direct CLI execution
        print("Direct CLI execution not yet implemented. Use --interactive mode.")


if __name__ == "__main__":
    main()