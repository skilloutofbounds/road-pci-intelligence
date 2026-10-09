"""
Data Conversion Rules, Assumptions, and Label Processing for Road PCI Intelligence.

This module defines deterministic conversion rules from raw dataset annotations
into standardized segmentation masks (crack and pothole channels) and bounding
box representations.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image

# Channel identification constants
CHANNEL_CRACK = 0
CHANNEL_POTHOLE = 1
NUM_SEGMENTATION_CHANNELS = 2

# Source-specific annotation capabilities
SOURCE_CAPABILITIES = {
    'road_crack': {
        'labels_crack': True,
        'labels_pothole': False,
        'annotation_type': 'raster_mask'
    },
    'crack500': {
        'labels_crack': True,
        'labels_pothole': False,
        'annotation_type': 'raster_mask'
    },
    'pothole': {
        'labels_crack': False,
        'labels_pothole': True,
        'annotation_type': 'yolo_polygon'
    },
    'rdd2022es': {
        'labels_crack': False,
        'labels_pothole': False,
        'annotation_type': 'yolo_bbox'
    }
}


def convert_road_crack_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Converts raw road-crack-dataset multi-class mask to a binary crack mask (0 or 1).

    RULE & ASSUMPTION:
    - Value 1 -> 1 (Crack).
    - Value 2 -> 0 (Background).
    - Value 3 -> 0 (Background).
    - Value 0 -> 0 (Background).

    ASSUMPTION DOCUMENTATION:
    The dataset author's Kaggle metadata lists classes in order:
    1. Concrete Crack
    2. Construction Joints
    3. Drilling
    However, the on-disk repository contains an empty README.md and no config file.
    Empirical inspection reveals:
    - Value 1 features elongated, tortuous crack networks (72.4% touch image borders).
    - Value 2 features straight, artificial construction slab joint seams (94.8% touch borders).
    - Value 3 features isolated, small circular core-drilling holes (0.0% touch borders).
    Therefore, Value 1 is mapped to 'crack', while construction joints (2) and drill holes (3)
    are mapped to background. This is a recorded assumption:
    "value-to-class mapping inferred from the Kaggle description order and mask geometry,
    not documented in the files".

    Args:
        raw_mask: 2D numpy array with integer pixel values in {0, 1, 2, 3}.

    Returns:
        binary_mask: 2D uint8 numpy array with values in {0, 1}.
    """
    binary_mask = np.zeros(raw_mask.shape[:2], dtype=np.uint8)
    binary_mask[raw_mask == 1] = 1
    return binary_mask


def convert_crack500_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Converts raw CRACK500 ground-truth mask to a binary crack mask (0 or 1).

    RULE:
    - Any non-zero pixel (> 0) -> 1 (Crack).
    - Zero (0) -> 0 (Background).

    CRACK500 benchmark stores masks as binary grayscale PNGs (0 and 255).
    Any non-zero defect pixel is converted to standard class index 1.

    Args:
        raw_mask: 2D numpy array.

    Returns:
        binary_mask: 2D uint8 numpy array with values in {0, 1}.
    """
    binary_mask = np.zeros(raw_mask.shape[:2], dtype=np.uint8)
    binary_mask[raw_mask > 0] = 1
    return binary_mask


def rasterize_pothole_polygons(
    img_shape: Tuple[int, int],
    label_path: Path
) -> np.ndarray:
    """
    Rasterizes YOLOv8 normalized polygon annotations into a binary pothole mask (0 or 1).

    RULE:
    - Reads line-by-line normalized coordinates (class_id x1 y1 x2 y2 ... xn yn).
    - Denormalizes polygon vertices to image coordinates (width, height).
    - Fills polygons using cv2.fillPoly to generate binary mask (1 for pothole, 0 for background).

    Args:
        img_shape: (height, width) of the target image.
        label_path: Path to the YOLOv8 .txt polygon annotation file.

    Returns:
        binary_mask: 2D uint8 numpy array with values in {0, 1}.
    """
    h, w = img_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if not label_path.exists() or label_path.stat().st_size == 0:
        return binary_mask

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) > 4:  # At least class_id + 2 coordinate pairs
            coords = [float(x) for x in parts[1:]]
            pts = []
            for i in range(0, len(coords), 2):
                px = int(round(coords[i] * w))
                py = int(round(coords[i + 1] * h))
                pts.append([px, py])
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(binary_mask, [pts_arr], color=1)

    return binary_mask


def load_standardized_mask(
    source: str,
    mask_or_label_path: Path,
    img_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Loads and applies the corresponding source conversion rule to return a
    standardized binary mask (values in {0, 1}).

    Args:
        source: Name of the source ('road_crack', 'crack500', 'pothole').
        mask_or_label_path: Absolute or relative Path to mask image or label text file.
        img_shape: (height, width) tuple required for polygon rasterization.

    Returns:
        2D uint8 mask with values in {0, 1}.
    """
    if source == 'road_crack':
        raw = np.array(Image.open(mask_or_label_path))
        return convert_road_crack_mask(raw)

    elif source == 'crack500':
        raw = np.array(Image.open(mask_or_label_path))
        return convert_crack500_mask(raw)

    elif source == 'pothole':
        if img_shape is None:
            raise ValueError("img_shape (height, width) is required for rasterizing pothole polygons.")
        return rasterize_pothole_polygons(img_shape, mask_or_label_path)

    else:
        raise ValueError(f"Unknown segmentation source: {source}")
