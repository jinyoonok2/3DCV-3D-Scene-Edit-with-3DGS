#!/usr/bin/env python3
"""
Run Pipeline - Execute 3DGS Scene Editing Pipeline

Usage:
    # Run single phase
    python run_pipeline.py --config configs/garden_config.yaml --phase 1
    
    # Run all phases
    python run_pipeline.py --config configs/garden_config.yaml --all
    
    # Run phases 1-3
    python run_pipeline.py --config configs/garden_config.yaml --phases 1 2 3
"""

import argparse
import sys
from pathlib import Path

from project_utils.config import ProjectConfig
from pipeline import Phase1Training, Phase2Segmentation, Phase3Removal, Phase4Placement
from rich.console import Console

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3DGS Scene Editing Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/garden_config.yaml)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run single phase (1-4)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        help="Run specific phases (e.g., --phases 1 2 3)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all phases (1-4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine which phases to run
    if args.all:
        phases_to_run = [1, 2, 3, 4]
    elif args.phases:
        phases_to_run = sorted(args.phases)
    elif args.phase:
        phases_to_run = [args.phase]
    else:
        console.print("[red]Error: Specify --phase, --phases, or --all[/red]")
        sys.exit(1)
    
    # Load config
    config = ProjectConfig(args.config)
    project_name = config.get("project", "name")
    
    console.print("\n" + "="*80)
    console.print(f"3DGS Scene Editing Pipeline")
    console.print(f"Project: {project_name}")
    console.print(f"Config: {args.config}")
    console.print(f"Phases: {phases_to_run}")
    console.print("="*80 + "\n")
    
    # Phase classes
    phase_classes = {
        1: Phase1Training,
        2: Phase2Segmentation,
        3: Phase3Removal,
        4: Phase4Placement,
    }
    
    # Run phases
    results = {}
    for phase_num in phases_to_run:
        try:
            PhaseClass = phase_classes[phase_num]
            phase = PhaseClass(config)
            
            # Override is_complete if --force flag is set
            if args.force:
                phase.is_complete = lambda: False
            
            result = phase.run()
            results[phase_num] = result
        except Exception as e:
            console.print(f"\n[red]Pipeline failed at Phase {phase_num}:[/red] {e}")
            sys.exit(1)
    
    console.print("\n" + "="*80)
    console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
    console.print("="*80 + "\n")


if __name__ == "__main__":
    main()
