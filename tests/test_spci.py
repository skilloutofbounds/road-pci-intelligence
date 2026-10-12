"""
Unit tests for simplified Pavement Condition Index (sPCI) module (src/spci.py).

Verifies core mathematical guarantees:
1. No distress gives sPCI = 100.0 and CDV = 0.0.
2. sPCI decreases monotonically as crack length grows.
3. sPCI decreases monotonically as crack width grows.
4. sPCI decreases monotonically as pothole area grows.
5. sPCI and CDV stay strictly in [0.0, 100.0] across 1,000 random inputs.
6. Combined deduct CDV never exceeds 100.0.
7. Output is strictly deterministic for identical inputs.
"""

import unittest
import numpy as np
import cv2
from src.spci import compute_spci, load_spci_config


class TestSPCI(unittest.TestCase):

    def setUp(self):
        self.config = load_spci_config()

    def test_01_no_distress_gives_100(self):
        """No distress gives sPCI = 100.0, CDV = 0.0, Good rating, and zero deducts."""
        empty = np.zeros((512, 512), dtype=np.uint8)
        res1 = compute_spci(empty, empty, config=self.config)
        res2 = compute_spci(None, None, config=self.config)

        print(f"[TEST 01] No distress: sPCI={res1['spci']}, CDV={res1['cdv']}, class={res1['condition_class']}")
        self.assertEqual(res1['spci'], 100.0)
        self.assertEqual(res1['cdv'], 0.0)
        self.assertEqual(res1['condition_class'], "Good")
        self.assertEqual(res2['spci'], 100.0)
        self.assertEqual(res2['cdv'], 0.0)
        self.assertEqual(res2['condition_class'], "Good")

        for k, v in res1['deduct_values'].items():
            self.assertEqual(v, 0.0)

    def test_02_monotonic_crack_length_growth(self):
        """sPCI decreases monotonically as crack length grows."""
        lengths = [50, 100, 200, 400]
        spci_vals = []

        for L in lengths:
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[255:258, 50:50 + L] = 1 # width 3, length L
            res = compute_spci(crack_mask=mask, pothole_mask=None, config=self.config)
            spci_vals.append(res['spci'])

        print(f"[TEST 02] Monotonic length: L={lengths} -> sPCI={spci_vals}")
        for i in range(len(spci_vals) - 1):
            self.assertGreater(
                spci_vals[i],
                spci_vals[i + 1],
                f"sPCI did not strictly decrease from L={lengths[i]} ({spci_vals[i]}) to L={lengths[i+1]} ({spci_vals[i+1]})"
            )

    def test_03_monotonic_crack_width_growth(self):
        """sPCI decreases monotonically as crack width grows (transitioning Low -> Med -> High)."""
        widths = [2, 7, 14] # 2 (Low <= 5), 7 (Med <= 9), 14 (High > 9)
        spci_vals = []

        for w in widths:
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[250:250 + w, 100:300] = 1 # length 200, width w
            res = compute_spci(crack_mask=mask, pothole_mask=None, config=self.config)
            spci_vals.append(res['spci'])

        print(f"[TEST 03] Monotonic width: w={widths} -> sPCI={spci_vals}")
        for i in range(len(spci_vals) - 1):
            self.assertGreater(
                spci_vals[i],
                spci_vals[i + 1],
                f"sPCI did not decrease from w={widths[i]} ({spci_vals[i]}) to w={widths[i+1]} ({spci_vals[i+1]})"
            )

    def test_04_monotonic_pothole_area_growth(self):
        """sPCI decreases monotonically as pothole area grows."""
        radii = [10, 25, 50, 100]
        spci_vals = []

        for r in radii:
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(mask, (256, 256), r, 1, thickness=-1)
            res = compute_spci(crack_mask=None, pothole_mask=mask, config=self.config)
            spci_vals.append(res['spci'])

        print(f"[TEST 04] Monotonic pothole: r={radii} -> sPCI={spci_vals}")
        for i in range(len(spci_vals) - 1):
            self.assertGreater(
                spci_vals[i],
                spci_vals[i + 1],
                f"sPCI did not decrease from r={radii[i]} ({spci_vals[i]}) to r={radii[i+1]} ({spci_vals[i+1]})"
            )

    def test_05_bounded_on_1000_random_inputs(self):
        """sPCI stays strictly in [0.0, 100.0] and CDV <= 100.0 across 1,000 random inputs."""
        rng = np.random.RandomState(42)
        valid_classes = {"Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed"}

        for i in range(1000):
            # Generate random synthetic shapes
            cm = np.zeros((128, 128), dtype=np.uint8)
            pm = np.zeros((128, 128), dtype=np.uint8)

            # Random cracks (lines)
            n_lines = rng.randint(0, 5)
            for _ in range(n_lines):
                pt1 = (rng.randint(0, 128), rng.randint(0, 128))
                pt2 = (rng.randint(0, 128), rng.randint(0, 128))
                thickness = rng.randint(1, 8)
                cv2.line(cm, pt1, pt2, 1, thickness)

            # Random potholes (circles)
            n_pots = rng.randint(0, 4)
            for _ in range(n_pots):
                center = (rng.randint(0, 128), rng.randint(0, 128))
                radius = rng.randint(2, 35)
                cv2.circle(pm, center, radius, 1, thickness=-1)

            res = compute_spci(crack_mask=cm, pothole_mask=pm, config=self.config)
            spci = res['spci']
            cdv = res['cdv']

            self.assertGreaterEqual(spci, 0.0, f"sPCI negative on sample {i}: {spci}")
            self.assertLessEqual(spci, 100.0, f"sPCI > 100 on sample {i}: {spci}")
            self.assertGreaterEqual(cdv, 0.0, f"CDV negative on sample {i}: {cdv}")
            self.assertLessEqual(cdv, 100.0, f"CDV > 100 on sample {i}: {cdv}")
            self.assertAlmostEqual(spci + cdv, 100.0, places=1)
            self.assertIn(res['condition_class'], valid_classes)

        print("[TEST 05] 1,000 random inputs: all sPCI and CDV strictly bounded in [0.0, 100.0].")

    def test_06_deterministic_for_identical_inputs(self):
        """sPCI computation is bitwise deterministic for identical inputs."""
        mask_c = np.zeros((512, 512), dtype=np.uint8)
        mask_c[255:258, 100:300] = 1
        mask_p = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(mask_p, (256, 256), 30, 1, thickness=-1)

        res_first = compute_spci(crack_mask=mask_c, pothole_mask=mask_p, config=self.config)
        for _ in range(10):
            res_repeat = compute_spci(crack_mask=mask_c, pothole_mask=mask_p, config=self.config)
            self.assertEqual(res_first, res_repeat)

        print(f"[TEST 06] Deterministic across 10 repetitions: sPCI={res_first['spci']}, CDV={res_first['cdv']}")

    def test_07_astm_d6433_rating_bands(self):
        """ASTM D6433 Figure 1 integer rating bands are non-overlapping and correctly classified."""
        from src.spci import get_condition_class
        cases = [
            (100.0, "Good"),
            (86.0, "Good"),
            (85.6, "Good"),
            (85.4, "Satisfactory"),
            (85.0, "Satisfactory"),
            (71.0, "Satisfactory"),
            (70.4, "Fair"),
            (70.0, "Fair"),
            (56.0, "Fair"),
            (55.4, "Poor"),
            (41.0, "Poor"),
            (40.4, "Very Poor"),
            (26.0, "Very Poor"),
            (25.4, "Serious"),
            (11.0, "Serious"),
            (10.4, "Failed"),
            (0.0, "Failed"),
        ]
        for score, expected in cases:
            actual = get_condition_class(score)
            self.assertEqual(actual, expected, f"Score {score} gave {actual}, expected {expected}")
        print("[TEST 07] ASTM D6433 integer rating bands verified across all boundaries.")

    def test_08_noise_floor_tau(self):
        """Noise floor tau suppresses crack deducts when skeleton length < tau."""
        mask_c = np.zeros((512, 512), dtype=np.uint8)
        mask_c[255:258, 100:150] = 1 # length ~50 px -> ~0.098 short sides

        # With tau = 0.0 (no noise floor), deduct is positive
        res_tau0 = compute_spci(crack_mask=mask_c, pothole_mask=None, config=self.config, noise_floor_tau=0.0)
        self.assertLess(res_tau0['spci'], 100.0)
        self.assertGreater(res_tau0['cdv'], 0.0)

        # With tau = 0.2 (above ~0.098 short sides), deduct is zero
        res_tau02 = compute_spci(crack_mask=mask_c, pothole_mask=None, config=self.config, noise_floor_tau=0.2)
        self.assertEqual(res_tau02['spci'], 100.0)
        self.assertEqual(res_tau02['cdv'], 0.0)
        self.assertEqual(res_tau02['condition_class'], "Good")
        print(f"[TEST 08] Noise floor tau verified: tau=0 -> sPCI={res_tau0['spci']}, tau=0.2 -> sPCI={res_tau02['spci']}")


if __name__ == '__main__':
    unittest.main()

