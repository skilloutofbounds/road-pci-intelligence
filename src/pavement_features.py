"""
Pavement feature extraction module for automated pavement condition assessment.

Computes interpretable geometric and morphological distress features from binary
segmentation masks (crack and pothole channels) normalized to a standardized
resolution (shorter side = 512 px).
"""

from typing import Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2
from skimage.morphology import skeletonize


def rescale_mask(mask: np.ndarray, target_shorter_side: int = 512) -> np.ndarray:
    """
    Rescales a binary mask so that its shorter side equals `target_shorter_side`
    using nearest-neighbor interpolation, followed by binary thresholding.
    """
    if mask is None or mask.size == 0:
        return np.zeros((target_shorter_side, target_shorter_side), dtype=np.uint8)

    h, w = mask.shape[:2]
    shorter = min(h, w)
    if shorter <= 0:
        return np.zeros((target_shorter_side, target_shorter_side), dtype=np.uint8)

    if shorter != target_shorter_side:
        scale = float(target_shorter_side) / float(shorter)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        rescaled = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        rescaled = mask.astype(np.uint8)

    return (rescaled > 0).astype(np.uint8)


def compute_skeleton_length(skel_binary: np.ndarray) -> float:
    """
    Calculates skeleton length in pixels using 8-connected steps.
    Orthogonal (horizontal/vertical) steps count as 1.0 px.
    Diagonal steps count as sqrt(2) ≈ 1.41421356 px.
    """
    if not np.any(skel_binary):
        return 0.0

    sk = (skel_binary > 0).astype(np.int32)
    # Sum undirected edges by checking the 4 forward neighbors
    h_edges = np.sum(sk[:, :-1] & sk[:, 1:]) * 1.0
    v_edges = np.sum(sk[:-1, :] & sk[1:, :]) * 1.0
    d1_edges = np.sum(sk[:-1, :-1] & sk[1:, 1:]) * np.sqrt(2.0)
    d2_edges = np.sum(sk[:-1, 1:] & sk[1:, :-1]) * np.sqrt(2.0)

    return float(h_edges + v_edges + d1_edges + d2_edges)


def compute_features(
    crack_mask: Optional[np.ndarray],
    pothole_mask: Optional[np.ndarray]
) -> Dict[str, Any]:
    """
    Computes pavement distress metrics from crack and pothole binary masks.

    Parameters
    ----------
    crack_mask : np.ndarray or None
        Binary or boolean 2D array indicating crack distress.
    pothole_mask : np.ndarray or None
        Binary or boolean 2D array indicating pothole distress.

    Returns
    -------
    dict
        Extracted features containing:
        - crack_area_frac: float (crack area / total image area)
        - crack_length_px: float (skeleton length in pixels, diagonal = sqrt(2))
        - crack_length_density: float (crack_length_px / 512, length per 512 px)
        - mean_width_px: float (crack_area / skeleton_length)
        - relative_width: float (mean_width_px / 512)
        - max_width_px: float (max local diameter from Euclidean distance transform)
        - width_p90_px: float (90th percentile of local width along skeleton)
        - n_crack_components: int (connected components >= 30 px)
        - n_junctions: int (skeleton pixels with >= 3 neighbors, merged when adjacent)
        - cells: int (enclosed background regions inside crack mask >= 100 px)
        - network_index: float ((n_junctions + cells) / (crack_length_px / 100))
        - pothole_count: int (pothole connected components >= 50 px)
        - pothole_area_frac: float (valid pothole area / total image area)
        - largest_pothole_frac: float (area of largest pothole / total image area)
    """
    # Fallback to zero arrays if None
    if crack_mask is None and pothole_mask is None:
        crack_mask = np.zeros((512, 512), dtype=np.uint8)
        pothole_mask = np.zeros((512, 512), dtype=np.uint8)
    elif crack_mask is None:
        crack_mask = np.zeros_like(pothole_mask, dtype=np.uint8)
    elif pothole_mask is None:
        pothole_mask = np.zeros_like(crack_mask, dtype=np.uint8)

    # 1. Rescale so shorter side is 512 px
    crack_rescaled = rescale_mask(crack_mask, target_shorter_side=512)
    pothole_rescaled = rescale_mask(pothole_mask, target_shorter_side=512)

    img_h, img_w = crack_rescaled.shape[:2]
    total_img_area = float(img_h * img_w)

    # -------------------------------------------------------------------------
    # CRACK FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    # Filter connected components smaller than 30 px
    n_crack_lbl, crack_lbls, crack_stats, _ = cv2.connectedComponentsWithStats(
        crack_rescaled, connectivity=8
    )

    clean_crack = np.zeros_like(crack_rescaled, dtype=np.uint8)
    n_crack_components = 0
    for lbl in range(1, n_crack_lbl):
        area = crack_stats[lbl, cv2.CC_STAT_AREA]
        if area >= 30:
            clean_crack[crack_lbls == lbl] = 1
            n_crack_components += 1

    crack_area_px = int(np.sum(clean_crack))
    crack_area_frac = float(crack_area_px / total_img_area) if total_img_area > 0 else 0.0

    if crack_area_px == 0:
        crack_length_px = 0.0
        crack_length_density = 0.0
        mean_width_px = 0.0
        relative_width = 0.0
        max_width_px = 0.0
        width_p90_px = 0.0
        n_junctions = 0
        cells = 0
        network_index = 0.0
    else:
        # Morphological skeleton
        skel = skeletonize(clean_crack > 0)
        crack_length_px = compute_skeleton_length(skel)

        # crack_length_density = crack_length_px / (512 * 512) * 512 = crack_length_px / 512
        crack_length_density = float(crack_length_px / 512.0)

        # Width metrics
        if crack_length_px > 0:
            mean_width_px = float(crack_area_px / crack_length_px)
        else:
            mean_width_px = 0.0
        relative_width = float(mean_width_px / 512.0)

        # Euclidean distance transform
        dist_transform = cv2.distanceTransform(clean_crack, cv2.DIST_L2, 5)
        skel_mask = (skel > 0)
        if np.any(skel_mask):
            local_widths = 2.0 * dist_transform[skel_mask]
            max_width_px = float(np.max(local_widths))
            width_p90_px = float(np.percentile(local_widths, 90))
        else:
            max_width_px = float(2.0 * np.max(dist_transform))
            width_p90_px = max_width_px

        # Junctions: skeleton pixels with >= 3 8-neighbors, merged when adjacent
        skel_u8 = skel_mask.astype(np.uint8)
        k_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel_u8, -1, k_neighbors) * skel_u8
        junction_pixels = ((neighbor_count >= 3) & (skel_u8 > 0)).astype(np.uint8)

        if np.any(junction_pixels):
            num_j_lbl, _ = cv2.connectedComponents(junction_pixels, connectivity=8)
            n_junctions = int(num_j_lbl - 1)
        else:
            n_junctions = 0

        # Cells: enclosed background regions inside closed crack mask >= 100 px
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_crack = cv2.morphologyEx(clean_crack, cv2.MORPH_CLOSE, k3)
        bg = (closed_crack == 0).astype(np.uint8)
        num_bg_lbl, bg_lbls, bg_stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)

        # Identify boundary components
        border_mask = np.zeros((img_h, img_w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_labels: Set[int] = set(np.unique(bg_lbls[border_mask]))

        cells = 0
        for lbl in range(1, num_bg_lbl):
            if lbl not in border_labels:
                cell_area = bg_stats[lbl, cv2.CC_STAT_AREA]
                if cell_area >= 100:
                    cells += 1

        # Network Index: (n_junctions + cells) per 100 px of crack length
        if crack_length_px > 0:
            network_index = float((n_junctions + cells) / (crack_length_px / 100.0))
        else:
            network_index = 0.0

    # -------------------------------------------------------------------------
    # POTHOLE FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    num_p_lbl, p_lbls, p_stats, _ = cv2.connectedComponentsWithStats(
        pothole_rescaled, connectivity=8
    )

    pothole_areas = []
    for lbl in range(1, num_p_lbl):
        area = p_stats[lbl, cv2.CC_STAT_AREA]
        if area >= 50:
            pothole_areas.append(area)

    pothole_count = int(len(pothole_areas))
    total_pothole_area = float(sum(pothole_areas))
    pothole_area_frac = float(total_pothole_area / total_img_area) if total_img_area > 0 else 0.0
    largest_pothole_frac = float(max(pothole_areas) / total_img_area) if (pothole_areas and total_img_area > 0) else 0.0

    return {
        'crack_area_frac': float(crack_area_frac),
        'crack_length_px': float(crack_length_px),
        'crack_length_density': float(crack_length_density),
        'mean_width_px': float(mean_width_px),
        'relative_width': float(relative_width),
        'max_width_px': float(max_width_px),
        'width_p90_px': float(width_p90_px),
        'n_crack_components': int(n_crack_components),
        'n_junctions': int(n_junctions),
        'cells': int(cells),
        'network_index': float(network_index),
        'pothole_count': int(pothole_count),
        'pothole_area_frac': float(pothole_area_frac),
        'largest_pothole_frac': float(largest_pothole_frac),
    }
