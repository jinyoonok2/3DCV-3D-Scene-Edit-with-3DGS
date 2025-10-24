"""Loss functions for InstructGS training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import lpips


class InstructGSLoss:
    """Combined loss function for InstructGS training."""
    
    def __init__(self, config, device='cuda'):
        """Initialize loss functions."""
        self.config = config
        self.device = device
        
        # Loss weights
        self.photometric_weight = config.losses['photometric_weight']
        self.preservation_weight = config.losses['preservation_weight']
        self.consistency_weight = config.losses['consistency_weight']
        self.perceptual_weight = config.losses.get('perceptual_weight', 0.0)
        self.depth_weight = config.losses.get('depth_weight', 0.0)
        
        # Initialize perceptual loss if needed
        self.lpips_loss = None
        if self.perceptual_weight > 0:
            self.lpips_loss = lpips.LPIPS(net='alex').to(device)
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
        
        print(f"✓ InstructGS loss initialized")
        print(f"  Photometric weight: {self.photometric_weight}")
        print(f"  Preservation weight: {self.preservation_weight}")
        print(f"  Consistency weight: {self.consistency_weight}")
        if self.perceptual_weight > 0:
            print(f"  Perceptual weight: {self.perceptual_weight}")
    
    def photometric_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        loss_type: str = 'l1'
    ) -> torch.Tensor:
        """
        Photometric loss between predicted and target images.
        
        Args:
            pred: Predicted image [H, W, 3]
            target: Target image [H, W, 3]
            mask: Optional mask [H, W] or [H, W, 1] (1=include, 0=exclude)
            loss_type: 'l1', 'l2', or 'charbonnier'
            
        Returns:
            Loss scalar
        """
        if loss_type == 'l1':
            loss = F.l1_loss(pred, target, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(pred, target, reduction='none')
        elif loss_type == 'charbonnier':
            diff = pred - target
            loss = torch.sqrt(diff * diff + 1e-6)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [H, W] -> [H, W, 1]
            loss = loss * mask
            # Normalize by masked area
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def perceptual_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perceptual loss using LPIPS.
        
        Args:
            pred: Predicted image [H, W, 3]
            target: Target image [H, W, 3]
            mask: Optional mask [H, W] or [H, W, 1]
            
        Returns:
            Loss scalar
        """
        if self.lpips_loss is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert to [1, 3, H, W] and normalize to [-1, 1]
        pred_lpips = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        target_lpips = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        
        # Apply mask if provided (crop to masked region)
        if mask is not None:
            if mask.dim() == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)  # [H, W, 1] -> [H, W]
            
            # Find bounding box of mask
            mask_indices = torch.nonzero(mask > 0.5)
            if len(mask_indices) > 0:
                y_min, x_min = mask_indices.min(0)[0]
                y_max, x_max = mask_indices.max(0)[0]
                
                # Crop images to masked region
                pred_lpips = pred_lpips[:, :, y_min:y_max+1, x_min:x_max+1]
                target_lpips = target_lpips[:, :, y_min:y_max+1, x_min:x_max+1]
        
        # Resize to minimum size for LPIPS (usually 64x64)
        min_size = 64
        if pred_lpips.shape[-1] < min_size or pred_lpips.shape[-2] < min_size:
            pred_lpips = F.interpolate(pred_lpips, size=(min_size, min_size), mode='bilinear', align_corners=False)
            target_lpips = F.interpolate(target_lpips, size=(min_size, min_size), mode='bilinear', align_corners=False)
        
        loss = self.lpips_loss(pred_lpips, target_lpips)
        return loss.mean()
    
    def consistency_loss(
        self, 
        renders: Dict[int, torch.Tensor], 
        gaussian_positions: torch.Tensor,
        camera_params: Dict[int, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Cross-view consistency loss to encourage coherent appearance.
        
        Args:
            renders: Dictionary of rendered images {view_idx: image [H, W, 3]}
            gaussian_positions: 3D positions of Gaussians [N, 3]
            camera_params: Camera parameters for each view
            
        Returns:
            Loss scalar
        """
        if len(renders) < 2:
            return torch.tensor(0.0, device=gaussian_positions.device)
        
        total_loss = 0.0
        num_pairs = 0
        
        view_indices = list(renders.keys())
        
        # Compare pairs of views
        for i in range(len(view_indices)):
            for j in range(i + 1, len(view_indices)):
                view_i, view_j = view_indices[i], view_indices[j]
                
                # Project 3D points to both views
                # This is a simplified consistency check
                # More sophisticated version would track corresponding pixels
                
                # For now, use a simple global consistency metric
                render_i = renders[view_i]
                render_j = renders[view_j]
                
                # Compare global statistics (mean color, variance)
                mean_i = render_i.mean(dim=[0, 1])  # [3]
                mean_j = render_j.mean(dim=[0, 1])  # [3]
                
                var_i = render_i.var(dim=[0, 1])   # [3]
                var_j = render_j.var(dim=[0, 1])   # [3]
                
                # Consistency loss on global statistics
                mean_loss = F.mse_loss(mean_i, mean_j)
                var_loss = F.mse_loss(var_i, var_j)
                
                total_loss += mean_loss + var_loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def preservation_loss(
        self, 
        current_render: torch.Tensor, 
        original_render: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Preservation loss to keep non-edited regions unchanged.
        
        Args:
            current_render: Current rendered image [H, W, 3]
            original_render: Original rendered image [H, W, 3]
            roi_mask: Region of interest mask [H, W] (1=edit region, 0=preserve)
            
        Returns:
            Loss scalar
        """
        # Invert mask to get preservation region
        preserve_mask = 1.0 - roi_mask
        if preserve_mask.dim() == 2:
            preserve_mask = preserve_mask.unsqueeze(-1)  # [H, W, 1]
        
        # Photometric loss in preservation region
        loss = self.photometric_loss(current_render, original_render, preserve_mask)
        return loss
    
    def compute_total_loss(
        self,
        pred_render: torch.Tensor,
        target_image: torch.Tensor,
        original_render: Optional[torch.Tensor] = None,
        roi_mask: Optional[torch.Tensor] = None,
        view_renders: Optional[Dict[int, torch.Tensor]] = None,
        gaussian_positions: Optional[torch.Tensor] = None,
        camera_params: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss combining all components.
        
        Args:
            pred_render: Predicted render [H, W, 3]
            target_image: Target edited image [H, W, 3]
            original_render: Original render before editing
            roi_mask: Region of interest mask [H, W]
            view_renders: Multi-view renders for consistency
            gaussian_positions: 3D Gaussian positions
            camera_params: Camera parameters
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        
        # 1. Photometric loss in RoI
        if roi_mask is not None:
            photo_loss = self.photometric_loss(pred_render, target_image, roi_mask)
        else:
            photo_loss = self.photometric_loss(pred_render, target_image)
        losses['photometric'] = photo_loss
        
        # 2. Perceptual loss
        if self.perceptual_weight > 0:
            perc_loss = self.perceptual_loss(pred_render, target_image, roi_mask)
            losses['perceptual'] = perc_loss
        
        # 3. Preservation loss
        if self.preservation_weight > 0 and original_render is not None and roi_mask is not None:
            pres_loss = self.preservation_loss(pred_render, original_render, roi_mask)
            losses['preservation'] = pres_loss
        
        # 4. Consistency loss
        if self.consistency_weight > 0 and view_renders is not None and gaussian_positions is not None:
            cons_loss = self.consistency_loss(view_renders, gaussian_positions, camera_params)
            losses['consistency'] = cons_loss
        
        # Combine losses
        total_loss = self.photometric_weight * losses['photometric']
        
        if 'perceptual' in losses:
            total_loss += self.perceptual_weight * losses['perceptual']
        
        if 'preservation' in losses:
            total_loss += self.preservation_weight * losses['preservation']
        
        if 'consistency' in losses:
            total_loss += self.consistency_weight * losses['consistency']
        
        losses['total'] = total_loss
        
        return total_loss, losses


class SSIMLoss(nn.Module):
    """SSIM loss implementation."""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        
    def gaussian_window(self, window_size, sigma=1.5):
        """Create Gaussian window."""
        coords = torch.arange(window_size, dtype=torch.float)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)
    
    def forward(self, img1, img2):
        """Compute SSIM loss."""
        # Convert to grayscale if needed
        if img1.dim() == 3 and img1.shape[-1] == 3:
            img1 = img1.mean(dim=-1, keepdim=True)
            img2 = img2.mean(dim=-1, keepdim=True)
        
        # Add batch and channel dimensions
        if img1.dim() == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
            img2 = img2.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
        
        # Create Gaussian window
        window = self.gaussian_window(self.window_size).to(img1.device)
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
        
        # Compute SSIM
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=[1, 2, 3])


def create_loss_function(config, device='cuda') -> InstructGSLoss:
    """Create loss function."""
    return InstructGSLoss(config, device)