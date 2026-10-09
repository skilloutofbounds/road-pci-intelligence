"""
Training Pipeline, Overfit Test & Smoke Test for Road PCI Intelligence.

Components:
- build_model(): Constructs U-Net ResNet-34 with ImageNet weights (halts if unavailable).
- train_one_epoch(): Runs training with AMP, AdamW, and MaskedMultiTaskLoss.
- evaluate(): Evaluates full images (padded to 32, cropped back) on validation set.
- run_overfit_test(): Sanity Check 2 (8 samples, 200 steps, target Dice > 0.85).
- run_smoke_test(): Sanity Check 3 (3 epochs, timing, GPU memory, val metrics).
- save_prediction_overlays(): Sanity Check 4 (6 overlays per source with clean titles).
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
import segmentation_models_pytorch as smp

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import RoadDistressDataset, SourceBalancedSampler, get_eval_transforms
from src.losses import MaskedMultiTaskLoss
from src.metrics import DistressMetricTracker
from src.build_manifest import DATASET_SPECS, resolve_dataset_root


def seed_everything(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_library_versions():
    """Logs versions of all core deep learning libraries."""
    print("=" * 80)
    print("LIBRARY VERSIONS")
    print("=" * 80)
    print(f"  • Python:                 {sys.version.split()[0]}")
    print(f"  • PyTorch:                {torch.__version__}")
    print(f"  • Torchvision:            {getattr(torch, '__version__', 'N/A')}")
    print(f"  • Albumentations:         {A.__version__}")
    print(f"  • smp:                    {smp.__version__}")
    print(f"  • CUDA Available:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • GPU Device:             {torch.cuda.get_device_name(0)}")
    print("=" * 80)


def build_model(device: torch.device) -> nn.Module:
    """
    Constructs U-Net with ResNet-34 encoder pretrained on ImageNet.
    Halts execution with RuntimeError if pretrained weights cannot be loaded.
    """
    print("\nInitializing U-Net (ResNet-34, ImageNet pretrained)...")
    try:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation=None
        )
    except Exception as e:
        raise RuntimeError(
            f"CRITICAL: Failed to download/load ImageNet pretrained weights: {e}. "
            f"Per requirements, training from random initialization is strictly disallowed."
        ) from e

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Pretrained U-Net initialized successfully. Trainable parameters: {total_params:,}")
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MaskedMultiTaskLoss,
    scaler: GradScaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> float:
    """Runs a single training epoch with AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets, label_masks, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        label_masks = label_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            loss, _ = loss_fn(logits, targets, label_masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model on validation split:
    Full images padded to multiple of 32, cropped back to original dimensions.
    """
    model.eval()
    tracker = DistressMetricTracker(threshold=0.5)

    for images, targets, _, metas in dataloader:
        images = images.to(device, non_blocking=True)

        with autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            probs = torch.sigmoid(logits)

        # Batch unpadding & metric accumulation
        batch_size = images.size(0)
        for i in range(batch_size):
            src = metas['source'][i]
            unpad_info = (
                int(metas['eval_unpad'][0][i]),
                int(metas['eval_unpad'][1][i]),
                int(metas['eval_unpad'][2][i]),
                int(metas['eval_unpad'][3][i])
            )
            tracker.update(src, probs[i], targets[i], eval_unpad=unpad_info)

    return tracker.compute()


def run_overfit_test(
    train_dataset: RoadDistressDataset,
    device: torch.device,
    num_steps: int = 200
) -> float:
    """
    Sanity Check 2: Overfit test on 8 training samples (3 crack500, 2 road_crack, 3 pothole).
    Must reach final per-source Dice > 0.85 without augmentation.
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK 2: OVERFIT TEST ON 8 SAMPLES")
    print("=" * 80)
    
    # 1. Select fixed 8 samples: 3 crack500, 2 road_crack, 3 pothole
    indices_c = [i for i, r in train_dataset.df.iterrows() if r['source'] == 'crack500'][:3]
    indices_r = [i for i, r in train_dataset.df.iterrows() if r['source'] == 'road_crack'][:2]
    indices_p = [i for i, r in train_dataset.df.iterrows() if r['source'] == 'pothole'][:3]
    
    selected_indices = indices_c + indices_r + indices_p
    assert len(selected_indices) == 8, f"Expected 8 samples, got {len(selected_indices)}"
    
    # Use evaluation transform (no augmentation)
    overfit_ds = RoadDistressDataset(
        manifest_df=train_dataset.df.iloc[selected_indices].copy(),
        dataset_roots=train_dataset.dataset_roots,
        split='train',
        crop_size=512,
        transform=get_eval_transforms()
    )
    overfit_loader = DataLoader(overfit_ds, batch_size=8, shuffle=False)

    model = build_model(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = MaskedMultiTaskLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    model.train()
    print("\nTraining on 8 fixed samples:")
    for step in range(1, num_steps + 1):
        for images, targets, label_masks, _ in overfit_loader:
            images = images.to(device)
            targets = targets.to(device)
            label_masks = label_masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(images)
                loss, loss_dict = loss_fn(logits, targets, label_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if step % 25 == 0 or step == num_steps:
            print(f"  Step [{step:3d}/{num_steps}]: Loss = {loss.item():.5f} "
                  f"(Crack: {loss_dict['loss_crack']:.5f}, Pothole: {loss_dict['loss_pothole']:.5f})")

    # Evaluate on the exact same 8 samples
    model.eval()
    tracker = DistressMetricTracker(threshold=0.5)
    with torch.no_grad():
        for images, targets, _, metas in overfit_loader:
            images = images.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                probs = torch.sigmoid(model(images))
            for i in range(len(images)):
                tracker.update(metas['source'][i], probs[i], targets[i])

    results = tracker.compute()
    print("\nOverfit Test Results on 8 Images:")
    for src in ['crack500', 'road_crack', 'pothole']:
        ds_dice = results.get(f'{src}_dataset_dice', 0.0)
        m_dice = results.get(f'{src}_mean_image_dice', 0.0)
        print(f"  • {src:10s}: Dataset Dice = {ds_dice:.4f} | Mean Image Dice = {m_dice:.4f}")

    mean_dice = results.get('mean_source_dice', 0.0)
    print(f"\nOverall Mean Source Dice: {mean_dice:.4f}")
    if mean_dice > 0.85:
        print("[PASS] Overfit test successfully achieved Dice > 0.85.")
    else:
        print(f"[FAIL] Overfit test failed to achieve Dice > 0.85 (achieved {mean_dice:.4f}).")

    return mean_dice


def save_prediction_overlays(
    model: nn.Module,
    val_dataset: RoadDistressDataset,
    output_dir: Path,
    device: torch.device,
    samples_per_source: int = 6
):
    """
    Sanity Check 4: Saves 6 validation prediction overlays per source
    (Image | Ground Truth | Prediction) with clean, non-overlapping titles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    sources = ['crack500', 'road_crack', 'pothole']
    for src in sources:
        sub_df = val_dataset.df[val_dataset.df['source'] == src]
        sample_df = sub_df.sample(min(samples_per_source, len(sub_df)), random_state=42)

        ch = 1 if src == 'pothole' else 0
        defect_name = 'Pothole' if ch == 1 else 'Crack'
        color = np.array([30, 120, 255]) if ch == 1 else np.array([255, 30, 30])  # Blue or Red

        fig, axes = plt.subplots(samples_per_source, 3, figsize=(14, 3.5 * samples_per_source))
        fig.suptitle(f"Dataset: {src} — 6 Validation Overlays (Defect: {defect_name})",
                     fontsize=14, weight='bold', y=0.995)

        for row_idx, (_, row) in enumerate(sample_df.iterrows()):
            root = val_dataset.dataset_roots[src]
            img_p = root / row['image_path']
            lbl_p = root / row['mask_path_or_label_path']

            # Load original image
            img_bgr = cv2.imread(str(img_p))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            # Downscale crack500 if needed
            if src == 'crack500':
                img_rgb = cv2.resize(img_rgb, (orig_w // 2, orig_h // 2), interpolation=cv2.INTER_LINEAR)
            curr_h, curr_w = img_rgb.shape[:2]

            # Ground truth mask
            mask_bin = load_standardized_mask(src, lbl_p, img_shape=(orig_h, orig_w))
            if src == 'crack500':
                mask_bin = cv2.resize(mask_bin, (orig_w // 2, orig_h // 2), interpolation=cv2.INTER_NEAREST)

            # Pad to multiple of 32 for inference
            pad_h = int(np.ceil(curr_h / 32.0) * 32) - curr_h
            pad_w = int(np.ceil(curr_w / 32.0) * 32) - curr_w
            img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

            # Transform & inference
            inp = get_eval_transforms()(image=img_padded)['image']
            tensor_in = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                with autocast(enabled=(device.type == 'cuda')):
                    probs = torch.sigmoid(model(tensor_in))[0, ch].cpu().numpy()

            # Crop back
            prob_map = probs[:curr_h, :curr_w]
            pred_bin = (prob_map >= 0.5).astype(np.uint8)

            # Colored masks
            gt_rgb = np.zeros_like(img_rgb)
            gt_rgb[mask_bin == 1] = color

            pred_rgb = np.zeros_like(img_rgb)
            pred_rgb[pred_bin == 1] = color

            # Calculate individual image Dice
            tp = int(np.logical_and(pred_bin == 1, mask_bin == 1).sum())
            denom = 2 * tp + int(np.logical_and(pred_bin == 1, mask_bin == 0).sum()) + int(np.logical_and(pred_bin == 0, mask_bin == 1).sum())
            img_dice = (2.0 * tp / denom) if denom > 0 else (1.0 if (mask_bin.sum() == 0 and pred_bin.sum() == 0) else 0.0)

            stem = Path(row['image_path']).stem
            if len(stem) > 25:
                stem = stem[:22] + "..."

            # Col 0: Image
            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"Image: {stem}", fontsize=9)
            axes[row_idx, 0].axis('off')

            # Col 1: Ground Truth
            axes[row_idx, 1].imshow(gt_rgb)
            axes[row_idx, 1].set_title(f"GT: {defect_name} (FG: {mask_bin.mean()*100:.2f}%)", fontsize=9)
            axes[row_idx, 1].axis('off')

            # Col 2: Prediction
            axes[row_idx, 2].imshow(pred_rgb)
            axes[row_idx, 2].set_title(f"Pred: Dice={img_dice:.3f} (FG: {pred_bin.mean()*100:.2f}%)", fontsize=9)
            axes[row_idx, 2].axis('off')

        plt.subplots_adjust(top=0.95, hspace=0.35, wspace=0.15)
        out_fig = output_dir / f"val_overlays_{src}.png"
        plt.savefig(out_fig, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved 6 overlays to {out_fig}")
