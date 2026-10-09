"""
Loss functions for Road PCI Intelligence.

Implements masked multi-task loss formulation:
- Masked Binary Cross-Entropy with Logits
- Soft Dice Loss
- Selective backpropagation based on sample-level supervision capability [labels_crack, labels_pothole].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Computes soft Dice loss for a batch of 2D binary masks.
    
    Args:
        probs: Predicted probabilities in [0, 1], shape (N, H, W)
        targets: Ground truth binary masks in {0, 1}, shape (N, H, W)
        smooth: Laplace smoothing constant to prevent division by zero
        
    Returns:
        Scalar soft Dice loss in [0, 1]
    """
    if probs.numel() == 0:
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        
    # Flatten spatial dimensions per sample: (N, H*W)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    intersection = (probs_flat * targets_flat).sum(dim=1)
    cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    dice_loss = 1.0 - dice_score
    return dice_loss.mean()


class MaskedMultiTaskLoss(nn.Module):
    """
    Multi-channel masked loss supporting partial supervision.
    
    Each channel (0: Crack, 1: Pothole) is supervised independently only
    over samples where label_mask[:, c] == 1.
    If no samples in a batch label channel c, the loss for that channel is exactly 0.0.
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        channel_weights: Tuple[float, float] = (1.0, 1.0),
        smooth: float = 1.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.channel_weights = channel_weights
        self.smooth = smooth
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: Predicted raw logits of shape (B, 2, H, W)
            targets: Binary ground-truth masks of shape (B, 2, H, W)
            label_mask: Binary supervision indicators of shape (B, 2)
            
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary containing component losses for logging
        """
        B, C, H, W = logits.shape
        device = logits.device
        dtype = logits.dtype
        
        channel_losses = []
        loss_dict = {}
        
        for c in range(C):
            # Select valid samples for this channel
            valid_idx = torch.where(label_mask[:, c] > 0.5)[0]
            
            if len(valid_idx) == 0:
                # No supervision for this channel in this batch
                c_bce = torch.tensor(0.0, device=device, dtype=dtype)
                c_dice = torch.tensor(0.0, device=device, dtype=dtype)
                c_loss = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                c_logits = logits[valid_idx, c, :, :]
                c_targets = targets[valid_idx, c, :, :].to(dtype=dtype)
                
                c_bce = F.binary_cross_entropy_with_logits(c_logits, c_targets)
                c_probs = torch.sigmoid(c_logits)
                c_dice = soft_dice_loss(c_probs, c_targets, smooth=self.smooth)
                c_loss = self.bce_weight * c_bce + self.dice_weight * c_dice
                
            ch_name = 'crack' if c == 0 else 'pothole'
            loss_dict[f'loss_{ch_name}_bce'] = c_bce.item()
            loss_dict[f'loss_{ch_name}_dice'] = c_dice.item()
            loss_dict[f'loss_{ch_name}'] = c_loss.item()
            
            channel_losses.append(self.channel_weights[c] * c_loss)
            
        total_loss = sum(channel_losses)
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
