"""
Evaluation metrics for Road PCI Intelligence.

Computes:
- Dataset-level IoU and Dice (accumulated TP/FP/FN over all pixels)
- Mean per-image Dice (averaged across individual images)
- Strictly per-channel evaluation based on source supervision capability:
  Crack channel for crack500 and road_crack; Pothole channel for pothole.
- Never computes cross-channel numbers.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_binary_metrics(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
    eps: float = 1e-6
) -> Tuple[int, int, int, float]:
    """
    Computes TP, FP, FN and per-image Dice score for a single binary mask pair.
    
    Args:
        pred_bin: Binary prediction array {0, 1}
        target_bin: Binary ground truth array {0, 1}
        eps: Small constant for numerical stability
        
    Returns:
        tp, fp, fn, per_image_dice
    """
    p = (pred_bin > 0.5)
    t = (target_bin > 0.5)
    
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, np.logical_not(t)).sum())
    fn = int(np.logical_and(np.logical_not(p), t).sum())
    
    # Per-image Dice (1.0 if both pred and target empty)
    if (2 * tp + fp + fn) == 0:
        per_img_dice = 1.0
    else:
        per_img_dice = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        
    return tp, fp, fn, per_img_dice


class DistressMetricTracker:
    """
    Tracks and accumulates evaluation metrics across validation/test batches.
    Maintains separate accumulators per source.
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.stats = {
            'crack500': {'tp': 0, 'fp': 0, 'fn': 0, 'image_dices': [], 'non_empty_image_dices': [], 'empty_image_dices': []},
            'road_crack': {'tp': 0, 'fp': 0, 'fn': 0, 'image_dices': [], 'non_empty_image_dices': [], 'empty_image_dices': []},
            'pothole': {'tp': 0, 'fp': 0, 'fn': 0, 'image_dices': [], 'non_empty_image_dices': [], 'empty_image_dices': []}
        }

    def update(
        self,
        source: str,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        eval_unpad: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Updates metrics for a single prediction.
        
        Args:
            source: Source dataset key ('crack500', 'road_crack', 'pothole')
            pred_probs: Sigmoid probability tensor of shape (2, H, W)
            targets: Ground-truth target tensor of shape (2, H, W)
            eval_unpad: Tuple (y0, x0, h, w) to crop padded predictions back to original size
        """
        if source not in self.stats:
            return

        # Select relevant channel
        ch = 1 if source == 'pothole' else 0

        p = pred_probs[ch].detach().cpu().numpy()
        t = targets[ch].detach().cpu().numpy()

        # Unpad if evaluation padding was applied
        if eval_unpad is not None:
            y0, x0, h, w = eval_unpad
            p = p[y0:y0 + h, x0:x0 + w]
            t = t[y0:y0 + h, x0:x0 + w]

        pred_bin = (p >= self.threshold).astype(np.uint8)
        target_bin = (t >= 0.5).astype(np.uint8)

        tp, fp, fn, dice_i = compute_binary_metrics(pred_bin, target_bin, eps=self.eps)

        self.stats[source]['tp'] += tp
        self.stats[source]['fp'] += fp
        self.stats[source]['fn'] += fn
        self.stats[source]['image_dices'].append(dice_i)

        if target_bin.sum() > 0:
            self.stats[source]['non_empty_image_dices'].append(dice_i)
        else:
            self.stats[source]['empty_image_dices'].append(dice_i)

    def compute(self) -> Dict[str, float]:
        """
        Computes final dataset-level IoU and Dice, plus mean per-image Dice.
        """
        results = {}
        all_src_dices = []

        for src, data in self.stats.items():
            tp = data['tp']
            fp = data['fp']
            fn = data['fn']
            img_dices = data['image_dices']
            n_images = len(img_dices)

            if n_images == 0:
                continue

            # Dataset-level IoU and Dice (accumulated over all pixels)
            denom_iou = tp + fp + fn
            ds_iou = float(tp / denom_iou) if denom_iou > 0 else 1.0

            denom_dice = 2 * tp + fp + fn
            ds_dice = float((2 * tp) / denom_dice) if denom_dice > 0 else 1.0

            # Mean per-image Dice
            mean_img_dice = float(np.mean(img_dices))

            # Non-empty images (images with at least one target foreground pixel)
            non_empty_dices = data.get('non_empty_image_dices', [])
            n_non_empty = len(non_empty_dices)
            mean_non_empty_dice = float(np.mean(non_empty_dices)) if n_non_empty > 0 else 0.0

            defect = 'Pothole' if src == 'pothole' else 'Crack'
            results[f'{src}_channel'] = defect
            results[f'{src}_images'] = n_images
            results[f'{src}_non_empty_images'] = n_non_empty
            results[f'{src}_dataset_iou'] = ds_iou
            results[f'{src}_dataset_dice'] = ds_dice
            results[f'{src}_mean_image_dice'] = mean_img_dice
            results[f'{src}_non_empty_mean_image_dice'] = mean_non_empty_dice

            all_src_dices.append(ds_dice)

        if all_src_dices:
            results['mean_source_dice'] = float(np.mean(all_src_dices))

        return results
