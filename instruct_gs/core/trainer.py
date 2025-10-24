"""Main training implementation for InstructGS."""

from instruct_gs.utils.path_manager import PathManager
from instruct_gs.core.state_manager import TrainingStateManager
from instruct_gs.training.round_trainer import create_trainer


class InstructGSTrainer:
    """Main trainer for InstructGS."""
    
    def __init__(self, config, path_manager: PathManager, state_manager: TrainingStateManager):
        """Initialize trainer."""
        self.config = config
        self.path_manager = path_manager
        self.state_manager = state_manager
        
        # Initialize round-based trainer
        self.round_trainer = create_trainer(config, path_manager, state_manager)
        
        print(f"✓ InstructGS trainer initialized")
    
    def train(self, start_round=0):
        """Run training."""
        print("🚀 Starting InstructGS training...")
        
        try:
            # Run round-based training
            training_history = self.round_trainer.train()
            
            # Update final state
            final_round = training_history['rounds'][-1] if training_history['rounds'] else {}
            final_loss = final_round.get('final_loss', 0.0)
            
            self.state_manager.update_state({
                'training_completed': True,
                'final_loss': final_loss,
                'total_rounds': len(training_history['rounds']),
                'total_steps': training_history['total_steps']
            })
            
            print("✓ InstructGS training completed successfully!")
            return {
                'status': 'completed', 
                'loss': final_loss,
                'rounds': len(training_history['rounds']),
                'steps': training_history['total_steps']
            }
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            self.state_manager.update_state({
                'training_failed': True,
                'error': str(e)
            })
            raise
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        if hasattr(self.round_trainer, 'load_checkpoint'):
            self.round_trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Would load checkpoint from: {checkpoint_path}")
    
    def save_checkpoint(self, name):
        """Save checkpoint."""
        if hasattr(self.round_trainer, 'save_checkpoint'):
            # Extract round number from current state
            current_round = self.state_manager.get_state().get('current_round', 0)
            self.round_trainer.save_checkpoint(current_round)
        else:
            print(f"Would save checkpoint: {name}")