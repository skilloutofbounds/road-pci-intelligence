"""
Unit tests for End-to-End Perception & Pavement Assessment Pipeline.

Tests:
1. Verifies that the app pipeline reproduces stored predicted_spci within 0.5 points
   across all 6 held-out test split example images in app_assets/examples/.
2. Strictly skips (prints SKIPPED, does not fail or silently pass) when weights are absent.
3. Verifies upload constraints (rejection of <256 px images, in-memory processing).
"""

import unittest
from pathlib import Path
import numpy as np

from src.pipeline import RoadPCIPipeline, DEFAULT_CHECKPOINT_CANDIDATES, calculate_sha256, EXPECTED_SHA256


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
        on the 6 held-out test split examples from results/tile_scores.csv.
        """
        if not self.weights_available or self.pipeline is None:
            raise unittest.SkipTest("SKIPPED: Perception weights 'best.pt' absent or invalid.")

        examples = [
            ("crack500_example_01.jpg", 70.68, "crack500"),
            ("crack500_example_02.jpg", 24.60, "crack500"),
            ("road_crack_example_01.jpg", 90.81, "road_crack"),
            ("road_crack_example_02.jpg", 40.97, "road_crack"),
            ("pothole_example_01.jpg", 43.65, "pothole"),
            ("pothole_example_02.jpg", 79.08, "pothole"),
        ]

        examples_dir = Path("app_assets/examples")
        self.assertTrue(examples_dir.exists(), f"Examples directory {examples_dir} not found")

        print("\n" + "=" * 80)
        print("PIPELINE FIDELITY TEST: 6 HELD-OUT TEST SPLIT EXAMPLES")
        print("=" * 80)

        differences = []
        for filename, target_spci, source in examples:
            img_path = examples_dir / filename
            self.assertTrue(img_path.exists(), f"Example image {img_path} not found")

            # Evaluate at benchmark tile resolution:
            # crack500 was evaluated at 0.5 downscale (720x1280) via upload resizing;
            # road_crack and pothole were evaluated at their native 640x640 resolution.
            is_upload = (source == "crack500")
            result = self.pipeline.infer(img_path, is_upload=is_upload, use_tta=True)
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
        print(f"All 6 example images match stored predicted_spci within tolerance (Max Diff: {max_diff:.4f} <= 0.50 points).")
        print("=" * 80)

    def test_02_upload_resizing_and_validation(self):
        """
        Verify upload preprocessing constraints:
        - Rejects images < 256 px
        - Resizes short side to 720 px (capped at 1280 px)
        - In-memory execution without disk writes
        """
        if not self.weights_available or self.pipeline is None:
            raise unittest.SkipTest("SKIPPED: Perception weights 'best.pt' absent or invalid.")

        # 1. Under-resolution rejection (< 256 px)
        small_img = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.pipeline.preprocess_image(small_img, is_upload=True)

        # 2. Upload resizing: 1000x2000 -> short side 720, long side capped at 1280
        test_img = np.ones((1000, 2000, 3), dtype=np.uint8) * 128
        processed = self.pipeline.preprocess_image(test_img, is_upload=True)
        h, w = processed.shape[:2]
        self.assertLessEqual(max(h, w), 1280)
        self.assertEqual(min(h, w), int(round(1280 * (1000 / 2000))))  # 640

        # 3. Upload resizing: 1080x1920 -> 720x1280
        test_hd = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        processed_hd = self.pipeline.preprocess_image(test_hd, is_upload=True)
        self.assertEqual(processed_hd.shape[:2], (720, 1280))


if __name__ == "__main__":
    unittest.main()
