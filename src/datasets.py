"""
Dataset and Sampler implementations for Road PCI Intelligence.

Features:
- RoadDistressDataset: unified multi-source pavement distress dataset.
- Mask standardization via src/data_rules.py.
- Albumentations pipeline (512x512 random crop & augmentation for train, pad to multiple of 32 for eval).
- SourceBalancedSampler: 600 samples per epoch (200 per source: crack500, road_crack, pothole).
"""

import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, Sampler
import albumentations as A

try:
    from src.data_rules import load_standardized_mask
    from src.build_manifest import DATASET_SPECS, resolve_dataset_root
except ImportError:
    from data_rules import load_standardized_mask
    from build_manifest import DATASET_SPECS, resolve_dataset_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms() -> A.Compose:
    """
    Augmentation pipeline for training:
    Flips, 90-degree rotations, mild brightness/contrast, mild blur, ImageNet normalization.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> A.Compose:
    """Evaluation preprocessing pipeline (normalization only, no augmentation)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class RoadDistressDataset(Dataset):
    """
    Multi-source road distress segmentation dataset.
    
    Reads manifest CSV and loads images and masks dynamically.
    Returns:
        image: torch.Tensor of shape (3, H, W) normalized to ImageNet mean/std
        target: torch.Tensor of shape (2, H, W) [channel 0: crack, channel 1: pothole]
        label_mask: torch.Tensor of shape (2,) [labels_crack, labels_pothole]
        meta: dict containing metadata (source, original dimensions, paths)
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_roots: Dict[str, Path],
        split: str = 'train',
        crop_size: int = 512,
        transform: Optional[A.Compose] = None
    ):
        self.split = split
        self.crop_size = crop_size
        self.dataset_roots = dataset_roots
        
        # Filter by split if column present
        if 'split' in manifest_df.columns:
            self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        else:
            self.df = manifest_df.reset_index(drop=True)
            
        self.transform = transform if transform is not None else (
            get_train_transforms() if split == 'train' else get_eval_transforms()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[idx]
        src_name = row['source']
        root = self.dataset_roots[src_name]
        
        img_p = root / row['image_path']
        lbl_rel = row['mask_path_or_label_path']
        lbl_p = root / lbl_rel if (pd.notna(lbl_rel) and lbl_rel != '') else None
        
        # 1. Load image (RGB)
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            # Fallback PIL load
            with Image.open(img_p) as pil_img:
                img_rgb = np.array(pil_img.convert('RGB'))
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        orig_h, orig_w = img_rgb.shape[:2]

        # 2. Load standardized 2D binary mask
        if lbl_p and lbl_p.exists():
            mask_binary = load_standardized_mask(src_name, lbl_p, img_shape=(orig_h, orig_w))
        else:
            mask_binary = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # 3. Handle CRACK500 downscale by 0.5
        if src_name == 'crack500':
            new_w = max(1, orig_w // 2)
            new_h = max(1, orig_h // 2)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_binary = cv2.resize(mask_binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        curr_h, curr_w = img_rgb.shape[:2]

        # 4. Construct 2-channel ground truth mask [crack, pothole]
        # Channel 0: Crack, Channel 1: Pothole
        labels_crack = bool(row['labels_crack'])
        labels_pothole = bool(row['labels_pothole'])
        
        target_mask = np.zeros((curr_h, curr_w, 2), dtype=np.uint8)
        if labels_crack:
            target_mask[:, :, 0] = mask_binary
        if labels_pothole:
            target_mask[:, :, 1] = mask_binary

        # 5. Spatial Cropping / Padding
        if self.split == 'train':
            # Training: random 512x512 crop (pad if smaller)
            pad_h = max(0, self.crop_size - curr_h)
            pad_w = max(0, self.crop_size - curr_w)
            if pad_h > 0 or pad_w > 0:
                img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                target_mask = cv2.copyMakeBorder(target_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                curr_h, curr_w = img_rgb.shape[:2]

            # Random crop
            y0 = random.randint(0, curr_h - self.crop_size) if curr_h > self.crop_size else 0
            x0 = random.randint(0, curr_w - self.crop_size) if curr_w > self.crop_size else 0
            img_crop = img_rgb[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            target_crop = target_mask[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            
            # Apply Albumentations transform
            transformed = self.transform(image=img_crop, mask=target_crop)
            img_out = transformed['image']
            target_out = transformed['mask']
            
            unpad_info = (0, 0, self.crop_size, self.crop_size)
        else:
            # Eval / Test: full image padded to multiple of 32
            pad_multiple = 32
            target_h = int(math.ceil(curr_h / pad_multiple) * pad_multiple)
            target_w = int(math.ceil(curr_w / pad_multiple) * pad_multiple)
            
            pad_bottom = target_h - curr_h
            pad_right = target_w - curr_w
            
            if pad_bottom > 0 or pad_right > 0:
                img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
                target_padded = cv2.copyMakeBorder(target_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
            else:
                img_padded = img_rgb
                target_padded = target_mask

            transformed = self.transform(image=img_padded, mask=target_padded)
            img_out = transformed['image']
            target_out = transformed['mask']
            
            # Stores original unpadded boundaries (h, w) to crop prediction back
            unpad_info = (0, 0, curr_h, curr_w)

        # 6. Convert to PyTorch tensors
        # Image: (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img_out).permute(2, 0, 1).float()
        # Target: (H, W, 2) -> (2, H, W)
        target_tensor = torch.from_numpy(target_out).permute(2, 0, 1).float()
        # Label mask: [crack_supervision, pothole_supervision]
        label_mask_tensor = torch.tensor(
            [1.0 if labels_crack else 0.0, 1.0 if labels_pothole else 0.0],
            dtype=torch.float32
        )

        meta = {
            'source': src_name,
            'image_path': row['image_path'],
            'group_id': row['group_id'],
            'orig_size': (orig_h, orig_w),
            'eval_unpad': unpad_info
        }

        return img_tensor, target_tensor, label_mask_tensor, meta


class SourceBalancedSampler(Sampler):
    """
    Sampler that yields a fixed number of samples per epoch (default 600),
    drawing equal samples from each of the three sources (crack500, road_crack, pothole).
    """

    def __init__(
        self,
        dataset: RoadDistressDataset,
        samples_per_epoch: int = 600,
        seed: int = 42
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        
        # Partition dataset indices by source
        self.source_indices = {'crack500': [], 'road_crack': [], 'pothole': []}
        for idx in range(len(dataset)):
            src = dataset.df.iloc[idx]['source']
            if src in self.source_indices:
                self.source_indices[src].append(idx)
            else:
                raise ValueError(f"Unknown source: {src}")

        self.samples_per_source = samples_per_epoch // 3

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000)
        selected_indices = []
        
        for src, idxs in self.source_indices.items():
            if not idxs:
                continue
            # Sample with replacement if dataset smaller than quota, else without
            if len(idxs) >= self.samples_per_source:
                chosen = rng.sample(idxs, self.samples_per_source)
            else:
                chosen = [rng.choice(idxs) for _ in range(self.samples_per_source)]
            selected_indices.extend(chosen)
            
        rng.shuffle(selected_indices)
        return iter(selected_indices)

    def __len__(self) -> int:
        return self.samples_per_epoch
