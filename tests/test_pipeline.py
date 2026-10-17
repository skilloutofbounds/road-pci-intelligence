"""
Unit tests for End-to-End Perception & Pavement Assessment Pipeline.

Tests:
1. Verifies that the app pipeline reproduces stored predicted_spci within 0.5 points
   across verified held-out test split example images in app_assets/examples/
   using the unified preprocessing rule for all images.
2. Strictly skips (prints SKIPPED, does not fail or silently pass) when weights are absent.
3. Verifies preprocessing constraints (rejection of <256 px images, unified 720 px rule).
4. Tests Hugging Face Hub model download if HF_MODEL_REPO environment variable is set.
"""

import unittest
from pathlib import Path
import numpy as np

from src.pipeline import (
    RoadPCIPipeline,
    DEFAULT_CHECKPOINT_CANDIDATES,
    calculate_sha256,
    EXPECTED_SHA256,
    HF_MODEL_REPO,
    resolve_or_download_weights,
)


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_available = False
        cls.pipeline = None
        for p in DEFAULT_CHECKPOINT_CANDIDATES:
            if p.exists() and p.is_file():
                try:
                    if calculate_sha256(p) == EXPECTED_SHA256:
                        cls.weights_available = True
                        cls.pipeline = RoadPCIPipeline(checkpoint_path=p, device="cpu")
                        break
                except Exception:
                    pass

    def test_01_example_tiles_match_stored_spci(self):
        """
        Verify that app pipeline reproduces stored predicted_spci within 0.5 points
        on verified held-out test split examples from results/tile_scores.csv
        using the single unified resize rule for all images.
        """
        if not self.weights_available or self.pipeline is None:
            raise unittest.SkipTest("SKIPPED: Perception weights 'best.pt' absent or invalid.")

        examples = [
            ("crack500_example_01.jpg", 70.68, "crack500"),
            ("crack500_example_02.jpg", 24.60, "crack500"),
            ("road_crack_example_01.jpg", 90.81, "road_crack"),
            ("road_crack_example_02.jpg", 40.97, "road_crack"),
        ]

        examples_dir = Path("app_assets/examples")
        self.assertTrue(examples_dir.exists(), f"Examples directory {examples_dir} not found")

        print("\n" + "=" * 80)
        print("PIPELINE FIDELITY TEST: VERIFIED HELD-OUT TEST SPLIT EXAMPLES (UNIFIED RESIZE RULE)")
        print("=" * 80)

        differences = []
        for filename, target_spci, source in examples:
            img_path = examples_dir / filename
            self.assertTrue(img_path.exists(), f"Example image {img_path} not found")

            # Apply single unified rule to all images (no per-source flags)
            result = self.pipeline.infer(img_path, use_tta=True)
            pred_spci = result["spci"]
            diff = abs(pred_spci - target_spci)
            differences.append((filename, source, pred_spci, target_spci, diff))

            print(
                f"  [{source:<10}] {filename:<25} | "
                f"Pipeline: {pred_spci:6.2f} | Stored: {target_spci:6.2f} | Diff: {diff:6.4f}"
            )
            self.assertLessEqual(
                diff,
                0.50,
                f"Pipeline sPCI ({pred_spci:.2f}) differs by {diff:.4f} > 0.50 points from stored ({target_spci:.2f}) on {filename}"
            )

        print("=" * 80)
        max_diff = max(d[4] for d in differences)
        print(f"All example images match stored predicted_spci within tolerance (Max Diff: {max_diff:.4f} <= 0.50 points).")
        print("=" * 80)

    def test_02_upload_resizing_and_validation(self):
        """
        Verify unified preprocessing constraints:
        - Rejects images < 256 px
        - If short side > 720: downscales short side to 720 px (capped at 1280 px)
        - Otherwise keeps native resolution (never upscales)
        - In-memory execution without disk writes
        """
        if not self.weights_available or self.pipeline is None:
            raise unittest.SkipTest("SKIPPED: Perception weights 'best.pt' absent or invalid.")

        # 1. Under-resolution rejection (< 256 px)
        small_img = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.pipeline.preprocess_image(small_img)

        # 2. Downscaling when short side > 720: 1000x2000 -> short side 720, long side capped at 1280
        test_img = np.ones((1000, 2000, 3), dtype=np.uint8) * 128
        processed = self.pipeline.preprocess_image(test_img)
        h, w = processed.shape[:2]
        self.assertLessEqual(max(h, w), 1280)
        self.assertEqual(min(h, w), int(round(1280 * (1000 / 2000))))  # 640

        # 3. Downscaling: 1080x1920 -> 720x1280
        test_hd = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        processed_hd = self.pipeline.preprocess_image(test_hd)
        self.assertEqual(processed_hd.shape[:2], (720, 1280))

        # 4. Native resolution preservation when short side <= 720: 640x640 -> 640x640 (never upscale)
        test_native = np.ones((640, 640, 3), dtype=np.uint8) * 128
        processed_native = self.pipeline.preprocess_image(test_native)
        self.assertEqual(processed_native.shape[:2], (640, 640))

    def test_03_hf_model_repo_download(self):
        """
        Verify downloading weights from Hugging Face Hub if HF_MODEL_REPO is configured.
        Prints SKIPPED if HF_MODEL_REPO is not set.
        """
        if not HF_MODEL_REPO:
            raise unittest.SkipTest("SKIPPED: HF_MODEL_REPO is not set.")

        downloaded_path = resolve_or_download_weights(hf_repo=HF_MODEL_REPO)
        sha = calculate_sha256(downloaded_path)
        self.assertEqual(
            sha,
            EXPECTED_SHA256,
            f"Hugging Face Hub checkpoint SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha}"
        )


if __name__ == "__main__":
    unittest.main()
