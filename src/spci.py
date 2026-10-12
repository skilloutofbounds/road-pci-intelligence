"""
Simplified Pavement Condition Index (sPCI) Module.

Translates geometric distress features and semantic segmentation masks
(crack and pothole channels) into objective, continuous pavement health
ratings on a [0, 100] scale according to ASTM D6433 rating principles.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import numpy as np
import cv2
from skimage.morphology import skeletonize

from src.pavement_features import (
    rescale_mask,
    compute_skeleton_length,
    prune_skeleton,
    compute_features,
)

DEFAULT_CONFIG_PATH = Path("configs/spci_params.yaml")


def load_spci_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads sPCI configuration dictionary from YAML or defaults."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Hardcoded fallbacks if YAML is missing
    return {
        "crack_severity": {
            "low_threshold_px": 5.0,
            "high_threshold_px": 9.0,
        },
        "network_factor": {
            "n_ref": 1.933,
            "beta": 0.5,
        },
        "pothole_severity": {
            "low_threshold_norm": 0.08,
            "high_threshold_norm": 0.20,
        },
        "deduct_curves": {
            "crack_low": {"a": 5.0, "b": 0.75},
            "crack_medium": {"a": 12.0, "b": 0.75},
            "crack_high": {"a": 25.0, "b": 0.75},
            "pothole_low": {"a": 100.0, "b": 0.60},
            "pothole_medium": {"a": 160.0, "b": 0.60},
            "pothole_high": {"a": 240.0, "b": 0.60},
        },
        "condition_classes": {
            "good": [86, 100],
            "satisfactory": [71, 85],
            "fair": [56, 70],
            "poor": [41, 55],
            "very_poor": [26, 40],
            "serious": [11, 25],
            "failed": [0, 10],
        },
        "noise_floor": {
            "tau_short_sides": 0.0,
        },
    }


def get_condition_class(spci: float, classes_cfg: Optional[Dict[str, list]] = None) -> str:
    """
    Maps continuous sPCI in [0, 100] to an ASTM D6433 condition category.
    Per ASTM D6433 Figure 1, the standard non-overlapping integer rating bands are:
      86-100: Good
      71-85:  Satisfactory
      56-70:  Fair
      41-55:  Poor
      26-40:  Very Poor
      11-25:  Serious
       0-10:  Failed
    Continuous sPCI is mapped by rounding to the nearest integer per ASTM D6433 Section 10.
    """
    val = int(np.clip(np.round(spci), 0, 100))
    if val >= 86:
        return "Good"
    elif val >= 71:
        return "Satisfactory"
    elif val >= 56:
        return "Fair"
    elif val >= 41:
        return "Poor"
    elif val >= 26:
        return "Very Poor"
    elif val >= 11:
        return "Serious"
    else:
        return "Failed"



def compute_spci(
    crack_mask: Optional[np.ndarray],
    pothole_mask: Optional[np.ndarray],
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    noise_floor_tau: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes simplified Pavement Condition Index (sPCI) from binary masks.

    Parameters
    ----------
    crack_mask : np.ndarray or None
        Binary mask indicating crack distress.
    pothole_mask : np.ndarray or None
        Binary mask indicating pothole distress.
    config : dict, optional
        Pre-loaded configuration dictionary.
    config_path : str or Path, optional
        Path to YAML configuration file.
    noise_floor_tau : float, optional
        Noise floor threshold tau in short sides (length / 512). If total crack
        skeleton length is below tau, crack deducts are set to 0.0.


    Returns
    -------
    dict
        sPCI rating result:
        - spci: float [0, 100]
        - condition_class: str
        - cdv: float [0, 100] (Corrected Total Deduct Value)
        - deduct_values: dict of individual DVs per distress & severity
        - quantities: dict of quantities per severity tier
        - network_factor: float multiplier m
    """
    cfg = config if config is not None else load_spci_config(config_path)

    # Fallback to zero arrays if None
    if crack_mask is None and pothole_mask is None:
        crack_mask = np.zeros((512, 512), dtype=np.uint8)
        pothole_mask = np.zeros((512, 512), dtype=np.uint8)
    elif crack_mask is None:
        crack_mask = np.zeros_like(pothole_mask, dtype=np.uint8)
    elif pothole_mask is None:
        pothole_mask = np.zeros_like(crack_mask, dtype=np.uint8)

    # Rescale to 512 px shorter side
    crack_rescaled = rescale_mask(crack_mask, target_shorter_side=512)
    pothole_rescaled = rescale_mask(pothole_mask, target_shorter_side=512)

    img_h, img_w = crack_rescaled.shape[:2]
    total_img_area = float(img_h * img_w)

    # -------------------------------------------------------------------------
    # 1. CRACK DISTRESS ANALYSIS
    # -------------------------------------------------------------------------
    n_crack_lbl, crack_lbls, crack_stats, _ = cv2.connectedComponentsWithStats(
        crack_rescaled, connectivity=8
    )
    clean_crack = np.zeros_like(crack_rescaled, dtype=np.uint8)
    for lbl in range(1, n_crack_lbl):
        if crack_stats[lbl, cv2.CC_STAT_AREA] >= 30:
            clean_crack[crack_lbls == lbl] = 1

    crack_area_px = int(np.sum(clean_crack))
    crack_length_px = 0.0
    q_crack_low = 0.0
    q_crack_med = 0.0
    q_crack_high = 0.0
    network_factor = 1.0

    if crack_area_px > 0:
        skel_raw = skeletonize(clean_crack > 0)
        skel = prune_skeleton(skel_raw, min_branch_len=12)
        crack_length_px = compute_skeleton_length(skel)


        # Distance transform for local width
        dist_transform = cv2.distanceTransform(clean_crack, cv2.DIST_L2, 5)
        skel_mask = (skel > 0)
        num_skel_px = int(np.sum(skel_mask))

        if num_skel_px > 0 and crack_length_px > 0:
            local_widths = 2.0 * dist_transform[skel_mask]

            t_low = float(cfg["crack_severity"]["low_threshold_px"])
            t_high = float(cfg["crack_severity"]["high_threshold_px"])

            mask_low = (local_widths <= t_low)
            mask_med = ((local_widths > t_low) & (local_widths <= t_high))
            mask_high = (local_widths > t_high)

            f_low = float(np.sum(mask_low)) / float(num_skel_px)
            f_med = float(np.sum(mask_med)) / float(num_skel_px)
            f_high = float(np.sum(mask_high)) / float(num_skel_px)

            norm_len = crack_length_px / 512.0
            q_crack_low = f_low * norm_len
            q_crack_med = f_med * norm_len
            q_crack_high = f_high * norm_len

            # Junctions and cells on pruned skeleton
            skel_u8 = skel.astype(np.uint8)
            k_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
            nc = cv2.filter2D(skel_u8, -1, k_neighbors) * skel_u8
            junc_px = ((nc >= 3) & (skel_u8 > 0)).astype(np.uint8)
            if np.any(junc_px):
                num_j_lbl, _ = cv2.connectedComponents(junc_px, connectivity=8)
                n_junctions = max(0, num_j_lbl - 1)
            else:
                n_junctions = 0

            # Cells
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_crack = cv2.morphologyEx(clean_crack, cv2.MORPH_CLOSE, k3)
            bg = (closed_crack == 0).astype(np.uint8)
            num_bg_lbl, bg_lbls, bg_stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)

            border_mask = np.zeros((img_h, img_w), dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            border_labels = set(np.unique(bg_lbls[border_mask]))

            cells = 0
            for lbl in range(1, num_bg_lbl):
                if lbl not in border_labels:
                    if bg_stats[lbl, cv2.CC_STAT_AREA] >= 100:
                        cells += 1

            network_index = float((n_junctions + cells) / (crack_length_px / 100.0))

            # Network factor m = 1 + beta * clip(network_index / n_ref, 0, 1)
            n_ref = float(cfg["network_factor"]["n_ref"])
            beta = float(cfg["network_factor"]["beta"])
            clipped_ratio = float(np.clip(network_index / n_ref, 0.0, 1.0)) if n_ref > 0 else 0.0
            network_factor = 1.0 + beta * clipped_ratio

    # -------------------------------------------------------------------------
    # 2. POTHOLE DISTRESS ANALYSIS
    # -------------------------------------------------------------------------
    num_p_lbl, p_lbls, p_stats, _ = cv2.connectedComponentsWithStats(
        pothole_rescaled, connectivity=8
    )

    q_pot_low = 0.0
    q_pot_med = 0.0
    q_pot_high = 0.0

    t_p_low = float(cfg["pothole_severity"]["low_threshold_norm"])
    t_p_high = float(cfg["pothole_severity"]["high_threshold_norm"])

    for lbl in range(1, num_p_lbl):
        p_area = float(p_stats[lbl, cv2.CC_STAT_AREA])
        if p_area >= 50.0:
            equiv_diam = 2.0 * np.sqrt(p_area / np.pi)
            norm_diam = equiv_diam / 512.0
            area_frac = p_area / total_img_area if total_img_area > 0 else 0.0

            if norm_diam < t_p_low:
                q_pot_low += area_frac
            elif norm_diam <= t_p_high:
                q_pot_med += area_frac
            else:
                q_pot_high += area_frac

    # -------------------------------------------------------------------------
    # 3. DEDUCT VALUES & TOTAL CORRECTED DEDUCT (CDV)
    # -------------------------------------------------------------------------
    curves = cfg["deduct_curves"]

    def calc_dv(q: float, params: dict, multiplier: float = 1.0) -> float:
        if q <= 0.0:
            return 0.0
        val = multiplier * float(params["a"]) * (q ** float(params["b"]))
        return float(np.clip(val, 0.0, 100.0))

    # Optional noise floor tau in short sides (length / 512)
    tau = float(cfg.get("noise_floor", {}).get("tau_short_sides", 0.0))
    if noise_floor_tau is not None:
        tau = float(noise_floor_tau)

    total_crack_length_short_sides = float(crack_length_px / 512.0)
    if total_crack_length_short_sides < tau:
        dv_crack_low = 0.0
        dv_crack_med = 0.0
        dv_crack_high = 0.0
    else:
        dv_crack_low = calc_dv(q_crack_low, curves["crack_low"], multiplier=network_factor)
        dv_crack_med = calc_dv(q_crack_med, curves["crack_medium"], multiplier=network_factor)
        dv_crack_high = calc_dv(q_crack_high, curves["crack_high"], multiplier=network_factor)



    dv_pot_low = calc_dv(q_pot_low, curves["pothole_low"], multiplier=1.0)
    dv_pot_med = calc_dv(q_pot_med, curves["pothole_medium"], multiplier=1.0)
    dv_pot_high = calc_dv(q_pot_high, curves["pothole_high"], multiplier=1.0)

    dvs = [
        dv_crack_low,
        dv_crack_med,
        dv_crack_high,
        dv_pot_low,
        dv_pot_med,
        dv_pot_high,
    ]

    # CDV = 100 * (1 - prod(1 - DV_i / 100))
    # sPCI = 100 - CDV = 100 * prod(1 - DV_i / 100)
    prod_factor = 1.0
    for dv in dvs:
        prod_factor *= (1.0 - (dv / 100.0))

    cdv = float(np.clip(100.0 * (1.0 - prod_factor), 0.0, 100.0))
    spci = float(np.clip(100.0 * prod_factor, 0.0, 100.0))

    condition_class = get_condition_class(spci, cfg.get("condition_classes"))

    return {
        "spci": float(round(spci, 2)),
        "condition_class": condition_class,
        "cdv": float(round(cdv, 2)),
        "deduct_values": {
            "crack_low": float(round(dv_crack_low, 2)),
            "crack_medium": float(round(dv_crack_med, 2)),
            "crack_high": float(round(dv_crack_high, 2)),
            "pothole_low": float(round(dv_pot_low, 2)),
            "pothole_medium": float(round(dv_pot_med, 2)),
            "pothole_high": float(round(dv_pot_high, 2)),
        },
        "quantities": {
            "crack_low_short_sides": float(round(q_crack_low, 4)),
            "crack_medium_short_sides": float(round(q_crack_med, 4)),
            "crack_high_short_sides": float(round(q_crack_high, 4)),
            "pothole_low_area_frac": float(round(q_pot_low, 6)),
            "pothole_medium_area_frac": float(round(q_pot_med, 6)),
            "pothole_high_area_frac": float(round(q_pot_high, 6)),
        },
        "network_factor": float(round(network_factor, 3)),
    }
