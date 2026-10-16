"""
Unified Perception & Assessment Pipeline for Road-PCI-Intelligence.

Integrates:
1. Deep Multi-Task U-Net (ResNet-34) Perception Model (Crack & Pothole segmentation)
2. ASTM D6433 Morphological Feature Extraction (Skeletons, widths, junction topology)
3. Simplified Pavement Condition Index (sPCI v1) Deduct-Curve Engine
4. Visual Qualitative Overlay Generator (High-contrast color blending)
5. Robust fallback handling with Demo Mode & synthetic benchmark generator
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import segmentation_models_pytorch as smp
    HAS_TORCH_SMP = True
except ImportError:
    HAS_TORCH_SMP = False

from src.pavement_features import compute_features, rescale_mask
from src.spci import compute_spci, get_condition_class, load_spci_config


# Standard Checkpoint Candidate Locations
DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("scratch/downloaded_final_weights/best.pt"),
    Path("scratch/road_pci_seg_final_weights/best.pt"),
    Path("scratch/run_b_weights/run_b_naive/best.pt"),
    Path("outputs/models/best.pt"),
    Path("outputs/models/model_b.pt"),
]


class RoadPCIPipeline:
    """
    Unified end-to-end inference and pavement assessment pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        spci_config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        force_demo_mode: bool = False,
    ):
        self.spci_config_path = spci_config_path
        self.spci_config = load_spci_config(spci_config_path)
        self.is_demo_mode = force_demo_mode
        self.model = None
        self.device = None
        self.checkpoint_path = None

        if not self.is_demo_mode and HAS_TORCH_SMP:
            self._init_model(checkpoint_path, device)
        else:
            self.is_demo_mode = True

    def _init_model(self, checkpoint_path: Optional[Union[str, Path]], device: Optional[str]):
        """Attempts to discover and load the frozen U-Net perception backbone."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        candidate_paths = [Path(checkpoint_path)] if checkpoint_path else DEFAULT_CHECKPOINT_CANDIDATES
        resolved_path = None
        for p in candidate_paths:
            if p.exists() and p.stat().st_size > 1000:
                resolved_path = p
                break

        if resolved_path is None:
            print("[Pipeline Info] No local model weights found in candidate paths. Initializing in Demo Mode.")
            self.is_demo_mode = True
            return

        try:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=2,
            )
            state_dict = torch.load(resolved_path, map_location=self.device)
            # Handle possible nested state dicts
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.checkpoint_path = resolved_path
            self.is_demo_mode = False
            print(f"[Pipeline OK] Loaded perception backbone from: {resolved_path} on {self.device}")
        except Exception as e:
            print(f"[Pipeline Warning] Failed to load checkpoint ({e}). Falling back to Demo Mode.")
            self.is_demo_mode = True

    def _load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Standardizes input to an RGB uint8 numpy array."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {p}")
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise ValueError(f"Failed to read image at {p}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                elif arr.shape[2] == 3:
                    return arr.astype(np.uint8)
            raise ValueError(f"Unsupported numpy image shape: {arr.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _demo_segmentation(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligent CV-based fallback segmentation for Demo Mode.
        Extracts plausible crack lines and pothole depressions using multi-scale morphological filtering.
        """
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Crack Proxy: High-frequency dark linear structures
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blackhat = cv2.morphologyEx(
            blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        )
        _, crack_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Filter small noise
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        crack_clean = cv2.morphologyEx(crack_thresh, cv2.MORPH_OPEN, k_open)
        crack_mask = (crack_clean > 0).astype(np.uint8)

        # 2. Pothole Proxy: Dark circular/blob depressions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        dark_thresh = cv2.adaptiveThreshold(
            eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7
        )
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        pothole_closed = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, k_close)
        
        # Only keep components of substantial area
        num_lbls, lbls, stats, _ = cv2.connectedComponentsWithStats(pothole_closed, connectivity=8)
        pothole_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_lbls):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > (h * w * 0.005) and area < (h * w * 0.35):
                pothole_mask[lbls == i] = 1

        return crack_mask, pothole_mask

    def _model_inference(
        self,
        img_rgb: np.ndarray,
        threshold_crack: float = 0.5,
        threshold_pothole: float = 0.5,
        use_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs PyTorch U-Net inference with Flip-TTA."""
        orig_h, orig_w = img_rgb.shape[:2]
        
        # Resize to standard 512x512 for network forward pass
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        tensor_x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_tta:
                # Forward + Horizontal + Vertical Flip-TTA
                p_orig = torch.sigmoid(self.model(tensor_x))
                p_h = torch.flip(torch.sigmoid(self.model(torch.flip(tensor_x, dims=[-1]))), dims=[-1])
                p_v = torch.flip(torch.sigmoid(self.model(torch.flip(tensor_x, dims=[-2]))), dims=[-2])
                probs = ((p_orig + p_h + p_v) / 3.0).cpu().numpy()[0]
            else:
                probs = torch.sigmoid(self.model(tensor_x)).cpu().numpy()[0]

        # Channel 0: Crack, Channel 1: Pothole
        prob_crack = cv2.resize(probs[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        prob_pothole = cv2.resize(probs[1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        crack_mask = (prob_crack >= threshold_crack).astype(np.uint8)
        pothole_mask = (prob_pothole >= threshold_pothole).astype(np.uint8)

        return crack_mask, pothole_mask

    def generate_overlay(
        self,
        img_rgb: np.ndarray,
        crack_mask: np.ndarray,
        pothole_mask: np.ndarray,
        alpha: float = 0.50,
    ) -> np.ndarray:
        """
        Creates an electric qualitative visual overlay on the pavement image:
        - Cracks highlighted in vivid Scarlet/Coral: RGB(255, 55, 95)
        - Potholes highlighted in vibrant Cyan/Turquoise: RGB(0, 225, 255)
        - Overlapping areas highlighted in Violet: RGB(220, 80, 255)
        """
        h, w = img_rgb.shape[:2]
        c_mask = rescale_mask(crack_mask, target_shorter_side=min(h, w))
        p_mask = rescale_mask(pothole_mask, target_shorter_side=min(h, w))

        # Resize masks back to exact image dimensions if necessary
        if c_mask.shape[:2] != (h, w):
            c_mask = cv2.resize(c_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if p_mask.shape[:2] != (h, w):
            p_mask = cv2.resize(p_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = img_rgb.copy().astype(np.float32)
        color_layer = np.zeros_like(overlay)

        # Colors
        crack_color = np.array([255, 55, 95], dtype=np.float32)
        pothole_color = np.array([0, 225, 255], dtype=np.float32)
        overlap_color = np.array([220, 80, 255], dtype=np.float32)

        has_crack = (c_mask > 0) & (p_mask == 0)
        has_pothole = (p_mask > 0) & (c_mask == 0)
        has_overlap = (c_mask > 0) & (p_mask > 0)
        has_distress = (c_mask > 0) | (p_mask > 0)

        color_layer[has_crack] = crack_color
        color_layer[has_pothole] = pothole_color
        color_layer[has_overlap] = overlap_color

        # Alpha blend where distress is present
        overlay[has_distress] = (
            (1.0 - alpha) * overlay[has_distress] + alpha * color_layer[has_distress]
        )

        # Draw crisp contours for professional definition
        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        if np.any(c_mask):
            contours_c, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_u8, contours_c, -1, (255, 20, 60), 1)
        if np.any(p_mask):
            contours_p, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_u8, contours_p, -1, (0, 255, 255), 2)

        return overlay_u8

    def infer(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        threshold_crack: float = 0.50,
        threshold_pothole: float = 0.50,
        use_tta: bool = True,
        overlay_alpha: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end assessment on a pavement image.

        Returns
        -------
        dict with keys:
            - 'spci': float in [0.0, 100.0]
            - 'condition_class': str ('Good', 'Satisfactory', 'Fair', 'Poor', 'Very Poor', 'Serious', 'Failed')
            - 'cdv': float
            - 'deduct_values': dict
            - 'features': dict (all 14 ASTM D6433 physical features)
            - 'crack_mask': np.ndarray (binary uint8)
            - 'pothole_mask': np.ndarray (binary uint8)
            - 'overlay_image': np.ndarray (RGB uint8)
            - 'original_image': np.ndarray (RGB uint8)
            - 'is_demo_mode': bool
            - 'checkpoint_path': str or None
        """
        img_rgb = self._load_image(image_input)

        if not self.is_demo_mode and self.model is not None:
            crack_mask, pothole_mask = self._model_inference(
                img_rgb,
                threshold_crack=threshold_crack,
                threshold_pothole=threshold_pothole,
                use_tta=use_tta,
            )
        else:
            crack_mask, pothole_mask = self._demo_segmentation(img_rgb)

        # 1. ASTM D6433 Feature Extraction
        features = compute_features(crack_mask, pothole_mask)

        # 2. sPCI v1 Score Computation
        spci_result = compute_spci(
            crack_mask=crack_mask,
            pothole_mask=pothole_mask,
            config=self.spci_config,
        )

        # 3. Qualitative Visual Overlay
        overlay_image = self.generate_overlay(
            img_rgb=img_rgb,
            crack_mask=crack_mask,
            pothole_mask=pothole_mask,
            alpha=overlay_alpha,
        )

        return {
            "spci": float(spci_result["spci"]),
            "condition_class": str(spci_result["condition_class"]),
            "cdv": float(spci_result.get("cdv", 0.0)),
            "deduct_values": spci_result.get("deduct_values", {}),
            "quantities": spci_result.get("quantities", {}),
            "network_factor": float(spci_result.get("network_factor", 1.0)),
            "features": features,
            "crack_mask": crack_mask,
            "pothole_mask": pothole_mask,
            "overlay_image": overlay_image,
            "original_image": img_rgb,
            "is_demo_mode": self.is_demo_mode,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


def create_synthetic_road_tile(
    condition: str = "crack",
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates realistic synthetic asphalt pavement tile for instantaneous zero-setup testing.
    """
    rng = np.random.RandomState(seed)
    
    # Asphalt base color with grain noise
    base_color = rng.randint(90, 115)
    tile = np.ones((height, width, 3), dtype=np.uint8) * base_color
    noise = rng.normal(0, 12, (height, width, 3)).astype(np.int16)
    tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Simulated aggregate specks
    specks = rng.rand(height, width) < 0.04
    tile[specks] = rng.randint(50, 80)
    specks_light = rng.rand(height, width) < 0.03
    tile[specks_light] = rng.randint(140, 175)

    if condition == "crack":
        # Draw meandering crack lines
        pts = []
        x = rng.randint(100, 200)
        for y in range(40, height - 40, 25):
            x += rng.randint(-18, 22)
            x = np.clip(x, 30, width - 30)
            pts.append([x, y])
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(tile, [pts_arr], False, (35, 35, 35), thickness=rng.randint(3, 5))
        # Branch
        branch_pts = np.array([[pts[5][0], pts[5][1]], [pts[5][0] + 70, pts[5][1] + 40], [pts[5][0] + 130, pts[5][1] + 90]], dtype=np.int32)
        cv2.polylines(tile, [branch_pts], False, (40, 40, 40), thickness=3)

    elif condition == "pothole":
        # Draw dark elliptical depression
        cx, cy = width // 2 + rng.randint(-30, 30), height // 2 + rng.randint(-30, 30)
        axes = (rng.randint(60, 95), rng.randint(45, 75))
        angle = rng.randint(0, 180)
        cv2.ellipse(tile, (cx, cy), axes, angle, 0, 360, (25, 25, 28), -1)
        cv2.ellipse(tile, (cx, cy), (axes[0] + 8, axes[1] + 8), angle, 0, 360, (55, 55, 60), 4)

    elif condition == "alligator":
        # Draw web of interconnected micro-cracks
        for _ in range(8):
            x1, y1 = rng.randint(50, width - 50), rng.randint(50, height - 50)
            x2, y2 = x1 + rng.randint(-80, 80), y1 + rng.randint(-80, 80)
            cv2.line(tile, (x1, y1), (x2, y2), (38, 38, 38), 2)

    return tile
