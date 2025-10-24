"""Training state management for InstructGS round-based editing."""

import yaml
import torch
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class TrainingStateManager:
    """Manages training state, checkpoints, and round progression."""
    
    def __init__(self, config, path_manager):
        self.config = config
        self.path_manager = path_manager
        self.state_file = path_manager.get_training_state_path()
    
    def initialize_training_state(self) -> Dict[str, Any]:
        """Initialize a new training state."""
        state = {
            'experiment_name': self.config.experiment['name'],
            'created_at': datetime.datetime.now().isoformat(),
            'current_round': 0,
            'max_rounds': self.config.editing['max_rounds'],
            'cycle_steps': self.config.editing['cycle_steps'],
            'edit_prompt': self.config.editing['edit_prompt'],
            'edit_mode': self.config.editing['edit_mode'],
            'completed_rounds': [],
            'round_metrics': {},
            'training_status': 'initialized',
            'last_updated': datetime.datetime.now().isoformat(),
            'total_iterations': 0,
            'best_round': None,
            'best_metric': None,
        }
        
        self.save_training_state(state)
        return state
    
    def load_training_state(self) -> Dict[str, Any]:
        """Load existing training state."""
        if not self.state_file.exists():
            return self.initialize_training_state()
        
        with open(self.state_file, 'r') as f:
            state = yaml.safe_load(f)
        
        return state
    
    def save_training_state(self, state: Dict[str, Any]):
        """Save training state to file."""
        state['last_updated'] = datetime.datetime.now().isoformat()
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.state_file, 'w') as f:
            yaml.dump(state, f, default_flow_style=False, indent=2)
    
    def update_round_start(self, round_num: int) -> Dict[str, Any]:
        """Update state at the start of a new round."""
        state = self.load_training_state()
        
        state['current_round'] = round_num
        state['training_status'] = 'training'
        state['round_start_time'] = datetime.datetime.now().isoformat()
        
        self.save_training_state(state)
        return state
    
    def update_round_completion(self, round_num: int, metrics: Dict[str, float], 
                              checkpoint_path: str) -> Dict[str, Any]:
        """Update state when a round is completed."""
        state = self.load_training_state()
        
        # Update completed rounds
        if round_num not in state['completed_rounds']:
            state['completed_rounds'].append(round_num)
            state['completed_rounds'].sort()
        
        # Save round metrics
        state['round_metrics'][f'round_{round_num:03d}'] = {
            'metrics': metrics,
            'checkpoint': checkpoint_path,
            'completed_at': datetime.datetime.now().isoformat(),
        }
        
        # Update total iterations
        state['total_iterations'] += self.config.editing['cycle_steps']
        
        # Check if this is the best round (based on PSNR)
        if 'psnr' in metrics:
            if state['best_metric'] is None or metrics['psnr'] > state['best_metric']:
                state['best_round'] = round_num
                state['best_metric'] = metrics['psnr']
        
        # Update training status
        if round_num >= self.config.editing['max_rounds'] - 1:
            state['training_status'] = 'completed'
        else:
            state['training_status'] = 'paused'
        
        self.save_training_state(state)
        return state
    
    def mark_training_interrupted(self, round_num: int):
        """Mark training as interrupted."""
        state = self.load_training_state()
        state['current_round'] = round_num
        state['training_status'] = 'interrupted'
        state['interrupted_at'] = datetime.datetime.now().isoformat()
        self.save_training_state(state)
    
    def mark_training_completed(self):
        """Mark training as fully completed."""
        state = self.load_training_state()
        state['training_status'] = 'completed'
        state['completed_at'] = datetime.datetime.now().isoformat()
        self.save_training_state(state)
    
    def has_existing_training(self) -> bool:
        """Check if there is existing training state."""
        return self.state_file.exists()
    
    def get_next_round(self) -> int:
        """Get the next round number to train."""
        state = self.load_training_state()
        return state.get('current_round', 0)
    
    def can_resume_training(self) -> bool:
        """Check if training can be resumed."""
        if not self.has_existing_training():
            return False
        
        state = self.load_training_state()
        status = state.get('training_status', '')
        current_round = state.get('current_round', 0)
        max_rounds = state.get('max_rounds', self.config.editing['max_rounds'])
        
        return status in ['paused', 'interrupted'] and current_round < max_rounds
    
    def is_training_completed(self) -> bool:
        """Check if training is completed."""
        if not self.has_existing_training():
            return False
        
        state = self.load_training_state()
        return state.get('training_status') == 'completed'
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress information."""
        state = self.load_training_state()
        
        current_round = state.get('current_round', 0)
        max_rounds = state.get('max_rounds', self.config.editing['max_rounds'])
        completed_rounds = len(state.get('completed_rounds', []))
        
        progress = {
            'current_round': current_round,
            'max_rounds': max_rounds,
            'completed_rounds': completed_rounds,
            'progress_percentage': (completed_rounds / max_rounds) * 100 if max_rounds > 0 else 0,
            'training_status': state.get('training_status', 'not_started'),
            'total_iterations': state.get('total_iterations', 0),
            'best_round': state.get('best_round'),
            'best_metric': state.get('best_metric'),
        }
        
        return progress
    
    def get_round_metrics(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics for a specific round or all rounds."""
        state = self.load_training_state()
        round_metrics = state.get('round_metrics', {})
        
        if round_num is not None:
            round_key = f'round_{round_num:03d}'
            return round_metrics.get(round_key, {})
        
        return round_metrics
    
    def save_checkpoint(self, round_num: int, model_state: Dict[str, Any], 
                       optimizer_states: Dict[str, Any], additional_data: Dict[str, Any] = None):
        """Save training checkpoint."""
        checkpoint_path = self.path_manager.get_checkpoint_path(round_num, "model")
        
        checkpoint = {
            'round_num': round_num,
            'model_state': model_state,
            'optimizer_states': optimizer_states,
            'config': self.config._config_dict,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.path_manager.get_checkpoint_path(None, "model")
        shutil.copy2(checkpoint_path, latest_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """Load checkpoint from specific round or latest."""
        if round_num is not None:
            checkpoint_path = self.path_manager.get_checkpoint_path(round_num, "model")
        else:
            checkpoint_path = self.path_manager.get_checkpoint_path(None, "model")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path, map_location='cpu')
    
    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for checkpoint_path in self.path_manager.list_checkpoints():
            try:
                # Try to load checkpoint metadata without loading full model
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                info = {
                    'path': str(checkpoint_path),
                    'filename': checkpoint_path.name,
                    'round_num': checkpoint.get('round_num'),
                    'timestamp': checkpoint.get('timestamp'),
                    'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                }
                
                checkpoints.append(info)
                
            except Exception as e:
                # Skip corrupted checkpoints
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
                continue
        
        # Sort by round number
        checkpoints.sort(key=lambda x: x['round_num'] if x['round_num'] is not None else -1)
        return checkpoints
    
    def reset_training(self, keep_checkpoints: bool = False):
        """Reset training state and optionally remove checkpoints."""
        # Remove state file
        if self.state_file.exists():
            self.state_file.unlink()
        
        # Clean up round data
        self.path_manager.clean_all_rounds()
        
        # Optionally remove checkpoints
        if not keep_checkpoints:
            for checkpoint_path in self.path_manager.list_checkpoints():
                checkpoint_path.unlink()
        
        # Initialize fresh state
        self.initialize_training_state()
        
        print("✓ Training state reset successfully")
    
    def backup_current_state(self, backup_name: Optional[str] = None) -> Path:
        """Create a backup of current training state."""
        if backup_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_dir = self.path_manager.base_output_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy state file
        if self.state_file.exists():
            shutil.copy2(self.state_file, backup_dir / "training_state.yaml")
        
        # Copy latest checkpoint
        latest_checkpoint = self.path_manager.get_checkpoint_path(None, "model")
        if latest_checkpoint.exists():
            shutil.copy2(latest_checkpoint, backup_dir / "model_latest.pt")
        
        # Copy config
        config_path = self.path_manager.get_config_path()
        if config_path.exists():
            shutil.copy2(config_path, backup_dir / "config.yaml")
        
        return backup_dir
    
    def restore_from_backup(self, backup_path: Path):
        """Restore training state from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Restore state file
        backup_state = backup_path / "training_state.yaml"
        if backup_state.exists():
            shutil.copy2(backup_state, self.state_file)
        
        # Restore checkpoint
        backup_checkpoint = backup_path / "model_latest.pt"
        if backup_checkpoint.exists():
            latest_checkpoint = self.path_manager.get_checkpoint_path(None, "model")
            shutil.copy2(backup_checkpoint, latest_checkpoint)
        
        print(f"✓ Restored training state from backup: {backup_path}")


def create_training_summary(state_manager: TrainingStateManager) -> Dict[str, Any]:
    """Create a comprehensive training summary."""
    state = state_manager.load_training_state()
    progress = state_manager.get_training_progress()
    metrics = state_manager.get_round_metrics()
    
    # Compute summary statistics
    all_psnr = [round_data['metrics'].get('psnr', 0) 
                for round_data in metrics.values() 
                if 'metrics' in round_data and 'psnr' in round_data['metrics']]
    
    summary = {
        'experiment_name': state.get('experiment_name'),
        'training_status': progress['training_status'],
        'progress': {
            'completed_rounds': progress['completed_rounds'],
            'total_rounds': progress['max_rounds'],
            'percentage': progress['progress_percentage'],
        },
        'metrics_summary': {
            'best_psnr': max(all_psnr) if all_psnr else None,
            'latest_psnr': all_psnr[-1] if all_psnr else None,
            'avg_psnr': sum(all_psnr) / len(all_psnr) if all_psnr else None,
        },
        'timing': {
            'created_at': state.get('created_at'),
            'last_updated': state.get('last_updated'),
            'total_iterations': state.get('total_iterations', 0),
        },
        'configuration': {
            'edit_prompt': state.get('edit_prompt'),
            'edit_mode': state.get('edit_mode'),
            'cycle_steps': state.get('cycle_steps'),
        }
    }
    
    return summary