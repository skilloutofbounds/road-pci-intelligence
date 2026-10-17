"""
Unified Perception & Assessment Pipeline for Road-PCI-Intelligence.

Integrates:
1. Deep Multi-Task U-Net (ResNet-34) Perception Model (Crack & Pothole segmentation)
2. Exact Flip-TTA (mean of original, horizontal flip, vertical flip sigmoid probabilities)
3. ASTM D6433 Morphological Feature Extraction (Skeletons, widths, junction topology at 512 px scale)
4. Simplified Pavement Condition Index (sPCI v1) Deduct-Curve Engine (Frozen tag: spci-v1-frozen)
5. Visual Qualitative Overlay Generator (High-contrast color blending)
6. Strict weight verification via SHA-256 and Hugging Face Hub integration (No Demo Mode fallback)
"""

import hashlib
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image, ImageOps

import torch
import segmentation_models_pytorch as smp

from src.pavement_features import compute_features, rescale_mask
from src.spci import compute_spci, get_condition_class, load_spci_config

# Constants
EXPECTED_SHA256 = "9269c5edb6e5d37b359bd3337fbd83c60901cd2b241f9ef789e9116fee400ef7"
HF_MODEL_REPO_ID = "vedshah04/road-pci-intelligence"
DEFAULT_CHECKPOINT_NAME = "best.pt"

# Candidate Local Paths (checked before Hugging Face download)
DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("scratch/downloaded_final_weights/best.pt"),
    Path("outputs/models/best.pt"),
    Path("scratch/road_pci_seg_final_weights/best.pt"),
    Path("scratch/run_b_weights/run_b_naive/best.pt"),
]

# ImageNet Normalization Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def calculate_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_or_download_weights(
    checkpoint_path: Optional[Union[str, Path]] = None,
    hf_repo_id: str = HF_MODEL_REPO_ID,
) -> Path:
    """
    Resolves model weights from a local path or downloads from Hugging Face Hub.
    Strictly verifies SHA-256 checksum against EXPECTED_SHA256.
    Raises FileNotFoundError or ValueError if weights cannot be verified.
    """
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(
                f"Perception model checkpoint '{checkpoint_path}' does not exist.\n"
                f"Expected SHA-256: {EXPECTED_SHA256}.\n"
                f"To fix: place 'best.pt' in 'outputs/models/best.pt' or download from Hugging Face repo '{hf_repo_id}'."
            )
        sha = calculate_sha256(p)
        if sha != EXPECTED_SHA256:
            raise ValueError(
                f"Checksum mismatch for explicit checkpoint at {p}. "
                f"Expected {EXPECTED_SHA256}, got {sha}."
            )
        return p

    candidates = list(DEFAULT_CHECKPOINT_CANDIDATES)

    # 1. Search local paths
    for p in candidates:
        if p.exists() and p.is_file():
            sha = calculate_sha256(p)
            if sha == EXPECTED_SHA256:
                return p
            else:
                if checkpoint_path and Path(checkpoint_path) == p:
                    raise ValueError(
                        f"Checksum mismatch for explicit checkpoint at {p}. "
                        f"Expected {EXPECTED_SHA256}, got {sha}."
                    )

    # 2. Attempt Hugging Face Hub download
    try:
        from huggingface_hub import hf_hub_download
        print(f"[Pipeline] Downloading {DEFAULT_CHECKPOINT_NAME} from Hugging Face Hub '{hf_repo_id}'...")
        downloaded = hf_hub_download(repo_id=hf_repo_id, filename=DEFAULT_CHECKPOINT_NAME)
        downloaded_path = Path(downloaded)
        sha = calculate_sha256(downloaded_path)
        if sha == EXPECTED_SHA256:
            return downloaded_path
        else:
            raise ValueError(
                f"Hugging Face weights checksum mismatch! Expected {EXPECTED_SHA256}, got {sha}."
            )
    except Exception as e:
        raise FileNotFoundError(
            f"Perception model checkpoint '{DEFAULT_CHECKPOINT_NAME}' (SHA-256: {EXPECTED_SHA256}) "
            f"could not be loaded from local candidate paths or Hugging Face repository '{hf_repo_id}'.\n"
            f"Details: {e}\n"
            f"To fix: place 'best.pt' in 'outputs/models/best.pt' or create public repo '{hf_repo_id}'."
        )


class RoadPCIPipeline:
    """
    Unified end-to-end inference and pavement assessment pipeline.
    Runs deep segmentation, morphological ASTM feature extraction, and sPCI calculation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        spci_config_path: Optional[Union[str, Path]] = "configs/spci_params.yaml",
        device: Optional[str] = None,
        hf_repo_id: str = HF_MODEL_REPO_ID,
    ):
        self.spci_config_path = spci_config_path
        self.spci_config = load_spci_config(spci_config_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve weights (raises FileNotFoundError if missing or invalid)
        self.checkpoint_path = resolve_or_download_weights(checkpoint_path, hf_repo_id=hf_repo_id)

        # Initialize U-Net perception backbone
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[Pipeline OK] Loaded perception backbone from: {self.checkpoint_path} on {self.device}")

    def preprocess_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
        is_upload: bool = True,
    ) -> np.ndarray:
        """
        Processes image in-memory:
        - Never writes uploaded images to disk.
        - Applies EXIF orientation.
        - Enforces minimum resolution (256 px) and file size cap (25 MB).
        - For uploads: resizes so short side is 720 px (long side capped at 1280 px, keeping aspect ratio).
        - Returns RGB uint8 numpy array.
        """
        # Load into PIL Image
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {p}")
            if p.stat().st_size > 25 * 1024 * 1024:
                raise ValueError("Image file size exceeds maximum allowed limit of 25 MB.")
            pil_img = Image.open(p)
        elif isinstance(image_input, bytes):
            if len(image_input) > 25 * 1024 * 1024:
                raise ValueError("Image byte size exceeds maximum allowed limit of 25 MB.")
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image_input.seek(0, io.SEEK_END)
            size = image_input.tell()
            image_input.seek(0)
            if size > 25 * 1024 * 1024:
                raise ValueError("Image buffer exceeds maximum allowed limit of 25 MB.")
            pil_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.ndim == 2:
                pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB))
                elif arr.shape[2] == 3:
                    pil_img = Image.fromarray(arr.astype(np.uint8))
                else:
                    raise ValueError(f"Unsupported numpy image channels: {arr.shape[2]}")
            else:
                raise ValueError(f"Unsupported numpy image shape: {arr.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 1. Apply EXIF orientation
        pil_img = ImageOps.exif_transpose(pil_img)
        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        # 2. Validate minimum resolution
        h, w = img_rgb.shape[:2]
        if min(h, w) < 256:
            raise ValueError(
                f"Image resolution ({w}x{h}) is too small. Both dimensions must be at least 256 px."
            )

        # 3. For user uploads: resize short side to 720 px (long side capped at 1280 px, keep aspect)
        if is_upload:
            short_side = min(h, w)
            long_side = max(h, w)
            scale = 720.0 / short_side
            new_short = 720
            new_long = int(round(long_side * scale))

            # Cap long side at 1280 px
            if new_long > 1280:
                scale_cap = 1280.0 / new_long
                new_long = 1280
                new_short = int(round(720.0 * scale_cap))

            if h <= w:
                target_h, target_w = new_short, new_long
            else:
                target_h, target_w = new_long, new_short

            if (h, w) != (target_h, target_w):
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return img_rgb

    def _model_inference(
        self,
        img_rgb: np.ndarray,
        threshold_crack: float = 0.50,
        threshold_pothole: float = 0.50,
        use_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs model inference with exact pipeline fidelity:
        - ImageNet normalization
        - Pad to multiple of 32 (bottom and right)
        - Flip-TTA: mean of sigmoid(original, horizontal flip, vertical flip)
        - Threshold at 0.5
        - Crop back to original unpadded resolution
        """
        h, w = img_rgb.shape[:2]

        # 1. Pad to multiple of 32
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(
                img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )
        else:
            img_padded = img_rgb

        # 2. ImageNet normalization
        img_float = img_padded.astype(np.float32) / 255.0
        img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 3. Model forward pass with Flip-TTA
        with torch.no_grad():
            if use_tta:
                p_orig = torch.sigmoid(self.model(tensor))
                p_h = torch.flip(torch.sigmoid(self.model(torch.flip(tensor, dims=[-1]))), dims=[-1])
                p_v = torch.flip(torch.sigmoid(self.model(torch.flip(tensor, dims=[-2]))), dims=[-2])
                probs = ((p_orig + p_h + p_v) / 3.0).cpu().numpy()[0]
            else:
                probs = torch.sigmoid(self.model(tensor)).cpu().numpy()[0]

        # 4. Crop back to unpadded dimensions and threshold at 0.5
        crack_mask = (probs[0, :h, :w] >= threshold_crack).astype(np.uint8)
        pothole_mask = (probs[1, :h, :w] >= threshold_pothole).astype(np.uint8)

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
        c_mask = crack_mask
        p_mask = pothole_mask

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
        image_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image],
        is_upload: bool = True,
        threshold_crack: float = 0.50,
        threshold_pothole: float = 0.50,
        use_tta: bool = True,
        overlay_alpha: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end assessment on a pavement image.
        Processes in-memory without saving to disk.
        """
        img_rgb = self.preprocess_image(image_input, is_upload=is_upload)

        crack_mask, pothole_mask = self._model_inference(
            img_rgb,
            threshold_crack=threshold_crack,
            threshold_pothole=threshold_pothole,
            use_tta=use_tta,
        )

        # 1. ASTM D6433 Feature Extraction (normalized at 512 px scale inside compute_features)
        features = compute_features(crack_mask, pothole_mask)

        # 2. sPCI v1 Score Computation (normalized at 512 px scale inside compute_spci)
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
            "checkpoint_path": str(self.checkpoint_path),
        }


def create_synthetic_road_tile(
    condition: str = "crack",
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates synthetic asphalt pattern for advanced/developer demonstration.
    NOTE: Synthetic pattern, not real pavement, not evidence of accuracy.
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
        branch_pts = np.array(
            [[pts[5][0], pts[5][1]], [pts[5][0] + 70, pts[5][1] + 40], [pts[5][0] + 130, pts[5][1] + 90]],
            dtype=np.int32
        )
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
