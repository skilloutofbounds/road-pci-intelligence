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
import torchvision

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
from src.data_rules import load_standardized_mask


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
    print(f"  • Torchvision:            {getattr(torchvision, '__version__', 'N/A')}")
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
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates model on validation split:
    Full images padded to multiple of 32, cropped back to original dimensions.
    """
    model.eval()
    tracker = DistressMetricTracker(threshold=threshold)

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


@torch.no_grad()
def run_threshold_sweep(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> List[Dict[str, float]]:
    """
    Sweeps decision threshold across {0.3, 0.4, 0.5, 0.6, 0.7} on validation split.
    Uses precomputed sigmoid probabilities to evaluate all thresholds efficiently in one pass.
    """
    model.eval()
    trackers = {th: DistressMetricTracker(threshold=th) for th in thresholds}

    for images, targets, _, metas in dataloader:
        images = images.to(device, non_blocking=True)

        with autocast(enabled=(device.type == 'cuda')):
            logits = model(images)
            probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        for i in range(batch_size):
            src = metas['source'][i]
            unpad_info = (
                int(metas['eval_unpad'][0][i]),
                int(metas['eval_unpad'][1][i]),
                int(metas['eval_unpad'][2][i]),
                int(metas['eval_unpad'][3][i])
            )
            for th in thresholds:
                trackers[th].update(src, probs[i], targets[i], eval_unpad=unpad_info)

    results = []
    for th in thresholds:
        th_metrics = trackers[th].compute()
        th_metrics['threshold'] = th
        results.append(th_metrics)

    return results


def print_threshold_table(sweep_results: List[Dict[str, float]]):
    """Prints a clean, formatted table of threshold sweep results per source."""
    print("\n" + "=" * 90)
    print("VALIDATION THRESHOLD SWEEP (Trained Model 'best.pt' on Val Split)")
    print("=" * 90)
    print(f"{'Source':<12} | {'Threshold':<9} | {'Dataset IoU':<12} | {'Dataset Dice':<12} | {'Mean Img Dice':<14}")
    print("-" * 90)
    for res in sweep_results:
        th = res['threshold']
        for src in ['crack500', 'road_crack', 'pothole']:
            ds_iou = res.get(f'{src}_dataset_iou', 0.0)
            ds_dice = res.get(f'{src}_dataset_dice', 0.0)
            m_dice = res.get(f'{src}_mean_image_dice', 0.0)
            print(f"{src:<12} | {th:<9.1f} | {ds_iou:<12.4f} | {ds_dice:<12.4f} | {m_dice:<14.4f}")
        print("-" * 90)


def plot_learning_curves(history: List[dict], output_path: Path):
    """
    Plots and saves learning curves:
    - Left: Training loss vs Epoch
    - Right: Validation Dataset Dice per source (crack500, road_crack, pothole) vs Epoch
    """
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]

    c500_dice = [h['val_metrics'].get('crack500_dataset_dice', 0.0) for h in history]
    rc_dice = [h['val_metrics'].get('road_crack_dataset_dice', 0.0) for h in history]
    pot_dice = [h['val_metrics'].get('pothole_dataset_dice', 0.0) for h in history]
    mean_dice = [h['val_metrics'].get('mean_source_dice', 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, marker='o', color='#2563EB', linewidth=2, label='Train Loss')
    ax1.set_title('Training Loss vs. Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Validation Dice curve
    ax2.plot(epochs, c500_dice, marker='s', color='#DC2626', linewidth=2, label='CRACK500 (Crack)')
    ax2.plot(epochs, rc_dice, marker='^', color='#F59E0B', linewidth=2, label='Road Crack (Crack)')
    ax2.plot(epochs, pot_dice, marker='o', color='#10B981', linewidth=2, label='Pothole (Pothole)')
    ax2.plot(epochs, mean_dice, marker='D', color='#8B5CF6', linewidth=2, linestyle='--', label='Mean Source Dice')
    ax2.set_title('Validation Dataset Dice per Source vs. Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Dataset Dice', fontsize=11)
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved learning curves figure to {output_path}")


def run_full_training(
    manifest_df: pd.DataFrame,
    dataset_roots: Dict[str, Path],
    epochs: int = 30,
    loss_mode: str = 'masked',
    loss_terms: str = 'bce+dice',
    seed: int = 42,
    output_dir: Path = Path('/kaggle/working'),
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    samples_per_epoch: int = 600,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, List[dict], List[dict]]:
    """
    Runs full semantic segmentation baseline training with ablation capabilities:
    - epochs: Number of training epochs (default 30)
    - loss_mode: 'masked' (multi-task partial supervision) or 'naive' (all-zero negative targets)
    - loss_terms: 'bce+dice', 'bce', or 'dice'
    - seed: Random seed for reproducibility
    - output_dir: Destination for best.pt, last.pt, history.json, learning_curves.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    log_library_versions()

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Configuration:")
    print(f"  • Epochs:            {epochs}")
    print(f"  • Loss Mode:         {loss_mode}")
    print(f"  • Loss Terms:        {loss_terms}")
    print(f"  • Random Seed:       {seed}")
    print(f"  • Learning Rate:     {lr}")
    print(f"  • Weight Decay:      {weight_decay}")
    print(f"  • Batch Size:        {batch_size}")
    print(f"  • Samples / Epoch:   {samples_per_epoch} (balanced: 200 per source)")
    print(f"  • Device:            {device}")
    print(f"  • Output Directory:  {output_dir}")

    # Initialize model
    model = build_model(device)

    # Initialize data loaders
    train_dataset = RoadDistressDataset(
        manifest_df=manifest_df,
        dataset_roots=dataset_roots,
        split='train',
        crop_size=512
    )
    train_sampler = SourceBalancedSampler(
        dataset=train_dataset,
        samples_per_epoch=samples_per_epoch,
        seed=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    val_dataset = RoadDistressDataset(
        manifest_df=manifest_df,
        dataset_roots=dataset_roots,
        split='val',
        crop_size=512
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\nDatasets Initialized:")
    print(f"  • Train dataset: {len(train_dataset):,} images | Sampler: {len(train_sampler)} images/epoch ({len(train_loader)} batches)")
    print(f"  • Val dataset:   {len(val_dataset):,} images ({len(val_loader)} batches)")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    loss_fn = MaskedMultiTaskLoss(loss_mode=loss_mode, loss_terms=loss_terms)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    history = []
    best_score = -1.0
    best_epoch = -1
    t_start = time.time()

    print("\n" + "=" * 90)
    print(f"STARTING TRAINING ({epochs} Epochs, Loss Mode: '{loss_mode}', Terms: '{loss_terms}')")
    print("=" * 90)

    for epoch in range(1, epochs + 1):
        t0_epoch = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            scheduler=scheduler
        )

        t_epoch = time.time() - t0_epoch
        peak_gpu_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == 'cuda' else 0.0

        # Evaluate on validation split
        val_results = evaluate(model=model, dataloader=val_loader, device=device)

        # Selection metric: mean(crack500 Dice, pothole Dice)
        c500_dice = val_results.get('crack500_dataset_dice', 0.0)
        pot_dice = val_results.get('pothole_dataset_dice', 0.0)
        selection_score = (c500_dice + pot_dice) / 2.0

        # Save last checkpoint
        torch.save(model.state_dict(), output_dir / 'last.pt')

        is_best = False
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            is_best = True
            torch.save(model.state_dict(), output_dir / 'best.pt')

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'epoch_time_s': t_epoch,
            'peak_gpu_mb': peak_gpu_mb,
            'selection_score': selection_score,
            'is_best': is_best,
            'val_metrics': val_results
        }
        history.append(record)

        best_marker = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.5f} | "
              f"CRACK500: Dice={c500_dice:.4f}, IoU={val_results.get('crack500_dataset_iou', 0.0):.4f} | "
              f"RoadCrack: Dice={val_results.get('road_crack_dataset_dice', 0.0):.4f}, IoU={val_results.get('road_crack_dataset_iou', 0.0):.4f} | "
              f"Pothole: Dice={pot_dice:.4f}, IoU={val_results.get('pothole_dataset_iou', 0.0):.4f} | "
              f"Score: {selection_score:.4f}{best_marker} ({t_epoch:.1f}s)")

    total_runtime_s = time.time() - t_start
    print("\n" + "=" * 90)
    print(f"TRAINING COMPLETE: Total Runtime: {total_runtime_s:.2f}s ({total_runtime_s/60:.2f} min)")
    print(f"Best Epoch: {best_epoch} with Mean(CRACK500, Pothole) Dice = {best_score:.4f}")
    print("=" * 90)

    # Save history JSON
    history_path = output_dir / 'history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training history to {history_path}")

    # Plot learning curves
    plot_learning_curves(history, output_dir / 'learning_curves.png')

    # Print summary table for every 5th epoch + best epoch
    print("\n" + "=" * 90)
    print("EPOCH VALIDATION SUMMARY (Every 5th Epoch + Best Epoch)")
    print("=" * 90)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'CRACK500 Dice':<14} | {'RoadCrack Dice':<15} | {'Pothole Dice':<13} | {'Mean Score':<11} | {'Notes'}")
    print("-" * 90)
    summary_epochs = sorted(list(set([e for e in range(5, epochs + 1, 5)] + [1, best_epoch])))
    for ep in summary_epochs:
        rec = history[ep - 1]
        c500_d = rec['val_metrics'].get('crack500_dataset_dice', 0.0)
        rc_d = rec['val_metrics'].get('road_crack_dataset_dice', 0.0)
        pot_d = rec['val_metrics'].get('pothole_dataset_dice', 0.0)
        note = "★ BEST" if ep == best_epoch else ""
        print(f"{ep:<6d} | {rec['train_loss']:<11.5f} | {c500_d:<14.4f} | {rc_d:<15.4f} | {pot_d:<13.4f} | {rec['selection_score']:<11.4f} | {note}")
    print("-" * 90)

    # Threshold sweep using best.pt on validation split only
    print("\nLoading best model checkpoint ('best.pt') for validation threshold sweep...")
    model.load_state_dict(torch.load(output_dir / 'best.pt', map_location=device))
    sweep_results = run_threshold_sweep(model, val_loader, device, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
    print_threshold_table(sweep_results)

    # Save sweep results
    sweep_path = output_dir / 'threshold_sweep.json'
    with open(sweep_path, 'w', encoding='utf-8') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[OK] Saved threshold sweep results to {sweep_path}")

    return model, history, sweep_results
