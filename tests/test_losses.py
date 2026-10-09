"""
Unit tests for MaskedMultiTaskLoss and Dice calculation.

Assertions:
(a) A pothole-only sample with the crack channel target set to all ones gives exactly
    zero gradient on the crack logits.
(b) Loss equals the hand-computed loss on labeled channels.
(c) A batch with no labels for a channel gives 0, not NaN.
(d) Dice of a perfect prediction is ~1 and of an all-zero prediction on a non-empty mask is ~0.
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import MaskedMultiTaskLoss, soft_dice_loss


class TestMaskedLosses(unittest.TestCase):
    
    def test_01_pothole_sample_zero_gradient_on_crack_logits(self):
        """
        (a) A pothole-only sample with the crack channel target set to all ones
        must give exactly zero gradient on the crack logits.
        """
        loss_fn = MaskedMultiTaskLoss()
        
        # Batch size 1, 2 channels, 16x16
        logits = torch.randn(1, 2, 16, 16, requires_grad=True)
        # Crack target set to all 1s intentionally
        targets = torch.zeros(1, 2, 16, 16)
        targets[0, 0, :, :] = 1.0  # Crack all 1s
        targets[0, 1, 4:8, 4:8] = 1.0  # Some pothole distress
        
        # Pothole-only sample: crack=0, pothole=1
        label_mask = torch.tensor([[0.0, 1.0]])
        
        loss, _ = loss_fn(logits, targets, label_mask)
        loss.backward()
        
        crack_grad = logits.grad[0, 0, :, :]
        pothole_grad = logits.grad[0, 1, :, :]
        
        # Assert crack grad is identically 0.0 everywhere
        self.assertTrue(
            torch.all(crack_grad == 0.0).item(),
            f"Expected crack grad to be all zeros, but max absolute grad was {crack_grad.abs().max().item()}"
        )
        # Assert pothole grad has non-zero values
        self.assertGreater(
            pothole_grad.abs().max().item(), 0.0,
            "Expected non-zero gradient on the supervised pothole channel."
        )
        print("[PASS] (a) Zero gradient on un-supervised crack channel verified.")

    def test_02_loss_equals_hand_computed_loss(self):
        """
        (b) Loss equals the hand-computed loss on labeled channels.
        """
        loss_fn = MaskedMultiTaskLoss(bce_weight=1.0, dice_weight=1.0, channel_weights=(1.0, 1.0), smooth=1.0)
        
        logits = torch.tensor([[[[1.5, -0.5], [0.0, 2.0]], [[-1.0, 1.0], [0.5, -0.5]]]])
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
        label_mask = torch.tensor([[1.0, 0.0]])  # Only crack labeled
        
        loss, loss_dict = loss_fn(logits, targets, label_mask)
        
        # Hand-compute crack channel (channel 0)
        c0_logits = logits[0, 0, :, :]
        c0_targets = targets[0, 0, :, :]
        
        expected_bce = F.binary_cross_entropy_with_logits(c0_logits, c0_targets).item()
        
        c0_probs = torch.sigmoid(c0_logits)
        inter = (c0_probs * c0_targets).sum().item()
        card = c0_probs.sum().item() + c0_targets.sum().item()
        expected_dice_score = (2.0 * inter + 1.0) / (card + 1.0)
        expected_dice_loss = 1.0 - expected_dice_score
        
        expected_total = expected_bce + expected_dice_loss
        
        self.assertAlmostEqual(loss.item(), expected_total, places=5)
        self.assertAlmostEqual(loss_dict['loss_pothole'], 0.0, places=5)
        print(f"[PASS] (b) Hand-computed loss matches exactly: {loss.item():.6f} == {expected_total:.6f}")

    def test_03_batch_with_no_labels_gives_zero_not_nan(self):
        """
        (c) A batch with no labels for a channel gives 0, not NaN.
        """
        loss_fn = MaskedMultiTaskLoss()
        
        logits = torch.randn(2, 2, 8, 8, requires_grad=True)
        targets = torch.zeros(2, 2, 8, 8)
        
        # Sample 0: crack only, Sample 1: crack only -> pothole channel has NO labels
        label_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        loss, loss_dict = loss_fn(logits, targets, label_mask)
        
        self.assertFalse(torch.isnan(loss).item())
        self.assertEqual(loss_dict['loss_pothole'], 0.0)
        self.assertFalse(torch.isnan(torch.tensor(loss_dict['loss_pothole'])).item())
        
        # Completely unlabelled batch
        empty_mask = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        loss_empty, dict_empty = loss_fn(logits, targets, empty_mask)
        self.assertEqual(loss_empty.item(), 0.0)
        self.assertFalse(torch.isnan(loss_empty).item())
        print("[PASS] (c) Zero loss and no NaN for unlabeled channels verified.")

    def test_04_dice_perfect_and_zero_predictions(self):
        """
        (d) Dice of a perfect prediction is ~1 and of an all-zero prediction
        on a non-empty mask is ~0.
        """
        # Non-empty target mask
        targets = torch.zeros(1, 32, 32)
        targets[0, 8:24, 8:24] = 1.0  # 256 non-empty pixels
        
        # Perfect prediction (prob = 1.0 on targets, 0.0 elsewhere)
        perfect_probs = targets.clone()
        perfect_loss = soft_dice_loss(perfect_probs, targets, smooth=1e-5)
        perfect_dice = 1.0 - perfect_loss.item()
        
        self.assertAlmostEqual(perfect_dice, 1.0, places=4)
        
        # All-zero prediction
        zero_probs = torch.zeros(1, 32, 32)
        zero_loss = soft_dice_loss(zero_probs, targets, smooth=1e-5)
        zero_dice = 1.0 - zero_loss.item()
        
        self.assertLess(zero_dice, 1e-3)
        self.assertAlmostEqual(zero_dice, 0.0, places=3)
        print(f"[PASS] (d) Dice metric: perfect = {perfect_dice:.5f} (~1), all-zero = {zero_dice:.5f} (~0).")


if __name__ == '__main__':
    unittest.main()
