"""Round-based training implementation for InstructGS."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import logging

from instruct_gs.models.gaussian_model import InstructGaussianModel
from instruct_gs.models.diffusion_model import InstructPix2PixModel
from instruct_gs.models.losses import create_loss_function
from instruct_gs.utils.path_manager import PathManager
from instruct_gs.core.state_manager import TrainingStateManager


class RoundBasedTrainer:
    """Round-based trainer implementing iterative 2D→3D optimization."""
    
    def __init__(self, config, path_manager: PathManager, state_manager: TrainingStateManager):
        """Initialize round-based trainer."""
        self.config = config
        self.path_manager = path_manager
        self.state_manager = state_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.max_rounds = config.editing['max_rounds']
        self.cycle_steps = config.editing['cycle_steps']
        self.edit_prompt = config.editing['edit_prompt']
        self.guidance_scale = config.diffusion.get('guidance_scale', 7.5)
        
        # View selection parameters
        self.views_per_round = config.editing.get('views_per_round', 4)
        self.view_selection_strategy = config.editing.get('view_selection_strategy', 'random')
        
        # Initialize models
        self.gaussian_model = None
        self.diffusion_model = None
        self.loss_function = None
        
        # Training state
        self.current_round = 0
        self.total_steps = 0
        self.round_history = []
        
        # Camera data
        self.cameras = {}
        self.camera_indices = []
        
        print(f"✓ Round-based trainer initialized")
        print(f"  Max rounds: {self.max_rounds}")
        print(f"  Steps per cycle: {self.cycle_steps}")
        print(f"  Edit prompt: '{self.edit_prompt}'")
    
    def initialize_models(self):
        """Initialize all models for training."""
        try:
            # Initialize Gaussian model
            print("Initializing Gaussian model...")
            self.gaussian_model = InstructGaussianModel(self.config, self.device)
            
            # Load camera data
            self._load_camera_data()
            
            # Initialize Gaussian splats
            self.gaussian_model.initialize_from_dataset(
                self.path_manager.get_dataset_path(),
                self.cameras
            )
            
            # Initialize diffusion model
            print("Initializing diffusion model...")
            self.diffusion_model = InstructPix2PixModel(self.config, self.device)
            
            # Initialize loss function
            print("Initializing loss function...")
            self.loss_function = create_loss_function(self.config, self.device)
            
            print("✓ All models initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing models: {e}")
            raise
    
    def _load_camera_data(self):
        """Load camera parameters from dataset."""
        try:
            cameras_file = self.path_manager.get_dataset_path() / "cameras.npz"
            if cameras_file.exists():
                camera_data = np.load(cameras_file)
                
                # Load camera parameters
                self.cameras = {}
                for i, key in enumerate(camera_data.files):
                    if key.startswith('camera_'):
                        idx = int(key.split('_')[1])
                        self.cameras[idx] = {
                            'K': torch.from_numpy(camera_data[key][:3, :3]).float().to(self.device),
                            'c2w': torch.from_numpy(camera_data[key]).float().to(self.device),
                            'image_size': (camera_data.get(f'size_{idx}', [800, 800]))
                        }
                        self.camera_indices.append(idx)
                
                print(f"✓ Loaded {len(self.cameras)} cameras")
            else:
                print("⚠ No cameras.npz found, using default camera")
                self._create_default_camera()
                
        except Exception as e:
            print(f"⚠ Error loading cameras: {e}, using default")
            self._create_default_camera()
    
    def _create_default_camera(self):
        """Create a default camera for testing."""
        # Simple front-facing camera
        self.cameras = {0: {
            'K': torch.tensor([
                [800.0, 0.0, 400.0],
                [0.0, 800.0, 400.0],
                [0.0, 0.0, 1.0]
            ]).to(self.device),
            'c2w': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0]
            ]).to(self.device),
            'image_size': (800, 800)
        }}
        self.camera_indices = [0]
    
    def select_views_for_round(self, round_num: int) -> List[int]:
        """Select camera views for current round."""
        if self.view_selection_strategy == 'random':
            # Random selection
            n_views = min(self.views_per_round, len(self.camera_indices))
            selected = np.random.choice(self.camera_indices, size=n_views, replace=False)
            return selected.tolist()
        
        elif self.view_selection_strategy == 'sequential':
            # Sequential selection
            start_idx = (round_num * self.views_per_round) % len(self.camera_indices)
            selected = []
            for i in range(self.views_per_round):
                idx = (start_idx + i) % len(self.camera_indices)
                selected.append(self.camera_indices[idx])
            return selected
        
        elif self.view_selection_strategy == 'all':
            # Use all views
            return self.camera_indices
        
        else:
            # Default to first few views
            return self.camera_indices[:self.views_per_round]
    
    def generate_roi_mask(self, image: torch.Tensor, method: str = 'center') -> torch.Tensor:
        """Generate region of interest mask for editing."""
        H, W = image.shape[:2]
        mask = torch.zeros((H, W), device=self.device)
        
        if method == 'center':
            # Center region (50% of image)
            center_h, center_w = H // 2, W // 2
            size_h, size_w = H // 4, W // 4
            mask[center_h-size_h:center_h+size_h, center_w-size_w:center_w+size_w] = 1.0
        
        elif method == 'full':
            # Full image
            mask[:, :] = 1.0
        
        elif method == 'bottom_half':
            # Bottom half of image
            mask[H//2:, :] = 1.0
        
        return mask
    
    def save_round_results(self, round_num: int, round_data: Dict):
        """Save results from current round."""
        round_dir = self.path_manager.get_round_dir(round_num)
        round_dir.mkdir(exist_ok=True)
        
        # Save images
        for view_idx, data in round_data['views'].items():
            # Original render
            if 'original_render' in data:
                self._save_image(
                    data['original_render'],
                    round_dir / f"view_{view_idx:03d}_original.png"
                )
            
            # Edited image
            if 'edited_image' in data:
                self._save_image(
                    data['edited_image'],
                    round_dir / f"view_{view_idx:03d}_edited.png"
                )
            
            # Final render
            if 'final_render' in data:
                self._save_image(
                    data['final_render'],
                    round_dir / f"view_{view_idx:03d}_final.png"
                )
            
            # RoI mask
            if 'roi_mask' in data:
                mask_img = (data['roi_mask'].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(round_dir / f"view_{view_idx:03d}_mask.png"), mask_img)
        
        # Save loss history
        if 'losses' in round_data:
            losses_file = round_dir / "losses.txt"
            with open(losses_file, 'w') as f:
                for step, loss_dict in enumerate(round_data['losses']):
                    f.write(f"Step {step}: {loss_dict}\n")
        
        # Save round summary
        summary_file = round_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Round {round_num} Summary\n")
            f.write(f"Edit prompt: {self.edit_prompt}\n")
            f.write(f"Selected views: {round_data.get('selected_views', [])}\n")
            f.write(f"Training steps: {self.cycle_steps}\n")
            if 'final_loss' in round_data:
                f.write(f"Final loss: {round_data['final_loss']}\n")
    
    def _save_image(self, tensor: torch.Tensor, path: Path):
        """Save tensor as image."""
        if tensor.dim() == 3:
            # Convert from [H, W, 3] to [3, H, W] if needed
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
        
        # Clamp to [0, 1] and convert to numpy
        img_np = torch.clamp(tensor, 0, 1).cpu().numpy()
        
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))  # [3, H, W] -> [H, W, 3]
        
        # Convert to PIL and save
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)
    
    def train_round(self, round_num: int) -> Dict:
        """Train one round of editing."""
        print(f"\n🔄 Starting Round {round_num + 1}/{self.max_rounds}")
        
        # Select views for this round
        selected_views = self.select_views_for_round(round_num)
        print(f"  Selected views: {selected_views}")
        
        # Store round data
        round_data = {
            'round_num': round_num,
            'selected_views': selected_views,
            'views': {},
            'losses': []
        }
        
        # Step 1: Render original images
        print("  Step 1: Rendering original images...")
        original_renders = {}
        for view_idx in selected_views:
            camera = self.cameras[view_idx]
            render = self.gaussian_model.render_view(camera['K'], camera['c2w'], camera['image_size'])
            original_renders[view_idx] = render
            
            round_data['views'][view_idx] = {'original_render': render}
        
        # Step 2: Apply diffusion editing
        print("  Step 2: Applying diffusion editing...")
        edited_images = {}
        roi_masks = {}
        
        for view_idx in selected_views:
            original = original_renders[view_idx]
            
            # Generate RoI mask
            roi_mask = self.generate_roi_mask(original, method='center')
            roi_masks[view_idx] = roi_mask
            
            # Apply diffusion editing
            edited = self.diffusion_model.edit_image(
                original,
                self.edit_prompt,
                guidance_scale=self.guidance_scale
            )
            edited_images[view_idx] = edited
            
            round_data['views'][view_idx].update({
                'edited_image': edited,
                'roi_mask': roi_mask
            })
        
        # Step 3: Optimize Gaussians
        print(f"  Step 3: Optimizing Gaussians ({self.cycle_steps} steps)...")
        
        for step in range(self.cycle_steps):
            step_losses = {}
            total_step_loss = 0.0
            
            # Train on all selected views
            for view_idx in selected_views:
                camera = self.cameras[view_idx]
                target_image = edited_images[view_idx]
                original_render = original_renders[view_idx]
                roi_mask = roi_masks[view_idx]
                
                # Render current view
                current_render = self.gaussian_model.render_view(
                    camera['K'], camera['c2w'], camera['image_size']
                )
                
                # Compute losses
                loss, loss_dict = self.loss_function.compute_total_loss(
                    pred_render=current_render,
                    target_image=target_image,
                    original_render=original_render,
                    roi_mask=roi_mask
                )
                
                # Optimize
                self.gaussian_model.optimize_step(loss)
                
                total_step_loss += loss.item()
                for k, v in loss_dict.items():
                    if k not in step_losses:
                        step_losses[k] = 0.0
                    step_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
            
            # Average losses across views
            for k in step_losses:
                step_losses[k] /= len(selected_views)
            
            round_data['losses'].append(step_losses)
            
            # Log progress
            if step % (self.cycle_steps // 10) == 0 or step == self.cycle_steps - 1:
                print(f"    Step {step:4d}/{self.cycle_steps}: Loss = {total_step_loss:.6f}")
            
            self.total_steps += 1
        
        # Step 4: Final renders
        print("  Step 4: Generating final renders...")
        for view_idx in selected_views:
            camera = self.cameras[view_idx]
            final_render = self.gaussian_model.render_view(
                camera['K'], camera['c2w'], camera['image_size']
            )
            round_data['views'][view_idx]['final_render'] = final_render
        
        # Store final loss
        if round_data['losses']:
            round_data['final_loss'] = round_data['losses'][-1]['total']
        
        # Save round results
        self.save_round_results(round_num, round_data)
        
        print(f"✓ Round {round_num + 1} completed")
        return round_data
    
    def train(self) -> Dict:
        """Run full round-based training."""
        print(f"\n🚀 Starting InstructGS training")
        print(f"  Edit prompt: '{self.edit_prompt}'")
        print(f"  Max rounds: {self.max_rounds}")
        print(f"  Steps per round: {self.cycle_steps}")
        
        try:
            # Initialize models
            self.initialize_models()
            
            # Training loop
            training_history = {
                'rounds': [],
                'config': self.config,
                'total_steps': 0
            }
            
            for round_num in range(self.max_rounds):
                # Train one round
                round_data = self.train_round(round_num)
                training_history['rounds'].append(round_data)
                
                # Update state
                self.current_round = round_num + 1
                self.state_manager.update_state({
                    'current_round': self.current_round,
                    'total_steps': self.total_steps,
                    'last_round_loss': round_data.get('final_loss', 0.0)
                })
                
                # Save checkpoint
                self.save_checkpoint(round_num)
            
            training_history['total_steps'] = self.total_steps
            
            print(f"\n✓ Training completed!")
            print(f"  Total rounds: {self.max_rounds}")
            print(f"  Total steps: {self.total_steps}")
            
            return training_history
            
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            raise
    
    def save_checkpoint(self, round_num: int):
        """Save training checkpoint."""
        checkpoint_dir = self.path_manager.get_checkpoints_path()
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'round_num': round_num,
            'total_steps': self.total_steps,
            'config': self.config,
            'gaussian_state': self.gaussian_model.get_state_dict() if self.gaussian_model else None,
        }
        
        checkpoint_file = checkpoint_dir / f"round_{round_num:03d}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # Also save as latest
        latest_file = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_file)
        
        print(f"  Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_round = checkpoint['round_num']
        self.total_steps = checkpoint['total_steps']
        
        if self.gaussian_model and checkpoint['gaussian_state']:
            self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resumed at round: {self.current_round}")
        print(f"  Total steps: {self.total_steps}")


def create_trainer(config, path_manager: PathManager, state_manager: TrainingStateManager) -> RoundBasedTrainer:
    """Create round-based trainer."""
    return RoundBasedTrainer(config, path_manager, state_manager)