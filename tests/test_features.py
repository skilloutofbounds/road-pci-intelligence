"""
Unit tests for pavement feature extraction module (src/pavement_features.py).

Tests synthetic masks with known analytical ground-truth values:
1. Straight line length (~200 px within 3%) and width (~3 px).
2. Plus sign junction count (exactly 1 merged junction).
3. 3x3 crack grid enclosed cells (exactly 4 cells, explained).
4. Filled disk pothole area (matches continuous pi * R^2).
5. Disjoint lines component count (2 components).
6. Speckle noise removal (< 30 px components discarded).
7. Empty masks (zeros for all metrics, no NaN/exceptions).
8. Scaling invariance (relative_width within 5% across 2x resolution).
"""

import unittest
import numpy as np
import cv2
from src.pavement_features import compute_features, rescale_mask


class TestPavementFeatures(unittest.TestCase):

    def test_01_straight_line_length_and_width(self):
        """
        A straight line of width 3 and length 200 gives length about 200
        (within 3%) and mean width about 3 (within 5%).
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Horizontal line: rows 250:253 (width 3), cols 100:300 (length 200)
        mask[250:253, 100:300] = 1
        feats = compute_features(crack_mask=mask, pothole_mask=None)

        length = feats['crack_length_px']
        width = feats['mean_width_px']
        rel_len_err = abs(length - 200.0) / 200.0
        rel_w_err = abs(width - 3.0) / 3.0

        print(f"[TEST 01] Straight line (w=3, L=200): length={length:.2f} px (err={rel_len_err*100:.2f}%), width={width:.2f} px (err={rel_w_err*100:.2f}%)")
        self.assertLessEqual(rel_len_err, 0.03, f"Length {length} not within 3% of 200")
        self.assertLessEqual(rel_w_err, 0.05, f"Width {width} not within 5% of 3")

    def test_02_plus_sign_junction(self):
        """
        A plus sign gives exactly 1 junction (adjacent junction pixels merged).
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Horizontal arm: width 3, length 200
        mask[254:257, 156:356] = 1
        # Vertical arm: width 3, length 200
        mask[156:356, 254:257] = 1

        feats = compute_features(crack_mask=mask, pothole_mask=None)
        n_junc = feats['n_junctions']
        print(f"[TEST 02] Plus sign: n_junctions={n_junc}")
        self.assertEqual(n_junc, 1, f"Expected 1 junction, got {n_junc}")

    def test_03_grid_cells(self):
        """
        A 3x3 grid of cracks gives 4 enclosed cells.

        Explanation:
        A 3x3 grid consists of 3 horizontal lines and 3 vertical lines intersecting.
        The lines partition the interior space into (3 - 1) * (3 - 1) = 2 * 2 = 4
        interior rectangular enclosed cells.
        Exterior background regions touch the image boundary and are excluded,
        while the 4 interior cells are completely bounded by crack pixels and have
        area >> 100 px each.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # 3 horizontal lines at rows 100, 250, 400 (width 3, span 100 to 400)
        for r in [100, 250, 400]:
            mask[r-1:r+2, 100:401] = 1
        # 3 vertical lines at cols 100, 250, 400 (width 3, span 100 to 400)
        for c in [100, 250, 400]:
            mask[100:401, c-1:c+2] = 1

        feats = compute_features(crack_mask=mask, pothole_mask=None)
        cells = feats['cells']
        print(f"[TEST 03] 3x3 Crack grid: cells={cells} (Expected: (3-1)*(3-1) = 4 enclosed cells)")
        self.assertEqual(cells, 4, f"Expected 4 cells for 3x3 grid, got {cells}")

    def test_04_filled_disk_pothole_area(self):
        """
        A filled disk gives the expected pothole area matching pi * R^2.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        radius = 30
        cv2.circle(mask, (256, 256), radius, 1, thickness=-1)

        feats = compute_features(crack_mask=None, pothole_mask=mask)
        pothole_count = feats['pothole_count']
        area_frac = feats['pothole_area_frac']
        largest_frac = feats['largest_pothole_frac']

        measured_area = area_frac * (512 * 512)
        expected_area = np.pi * (radius ** 2)
        rel_area_err = abs(measured_area - expected_area) / expected_area

        print(f"[TEST 04] Filled disk (R={radius}): count={pothole_count}, measured_area={measured_area:.1f} px, expected_pi_R2={expected_area:.1f} px (err={rel_area_err*100:.2f}%)")
        self.assertEqual(pothole_count, 1)
        self.assertEqual(area_frac, largest_frac)
        self.assertLessEqual(rel_area_err, 0.02, f"Disk area error {rel_area_err} > 2%")

    def test_05_two_disjoint_lines_components(self):
        """
        Two disjoint lines give exactly 2 connected components.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Line 1: length 80, width 3 (area 240 >= 30)
        mask[100:103, 50:130] = 1
        # Line 2: length 80, width 3 (area 240 >= 30)
        mask[300:303, 300:380] = 1

        feats = compute_features(crack_mask=mask, pothole_mask=None)
        comps = feats['n_crack_components']
        print(f"[TEST 05] Two disjoint lines: n_crack_components={comps}")
        self.assertEqual(comps, 2, f"Expected 2 components, got {comps}")

    def test_06_speckle_removal(self):
        """
        A speckle of tiny blobs (< 30 px area each) is completely removed.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Place several tiny blobs of 3x3 = 9 px or 4x4 = 16 px
        mask[50:54, 50:54] = 1   # 16 px
        mask[120:123, 120:123] = 1 # 9 px
        mask[200:205, 200:205] = 1 # 25 px
        mask[300:304, 300:305] = 1 # 20 px

        feats = compute_features(crack_mask=mask, pothole_mask=None)
        print(f"[TEST 06] Tiny speckles: n_components={feats['n_crack_components']}, crack_area_frac={feats['crack_area_frac']}, crack_length_px={feats['crack_length_px']}")
        self.assertEqual(feats['n_crack_components'], 0)
        self.assertEqual(feats['crack_area_frac'], 0.0)
        self.assertEqual(feats['crack_length_px'], 0.0)
        self.assertEqual(feats['mean_width_px'], 0.0)

    def test_07_empty_masks_give_zeros(self):
        """
        Empty masks return zeros for all features without NaN or exceptions.
        """
        empty_mask = np.zeros((512, 512), dtype=np.uint8)
        feats1 = compute_features(crack_mask=empty_mask, pothole_mask=empty_mask)
        feats2 = compute_features(crack_mask=None, pothole_mask=None)

        print(f"[TEST 07] Empty mask features: {feats1}")
        for k, v in feats1.items():
            self.assertEqual(v, 0, f"Feature {k} is not 0 for empty mask: {v}")
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")

        for k, v in feats2.items():
            self.assertEqual(v, 0, f"Feature {k} is not 0 for None input: {v}")
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")

    def test_08_scaling_invariance_relative_width(self):
        """
        The same shape rendered at 2x resolution gives relative_width within 5%.
        """
        # Shape at 256x256: width 4, length 100
        m256 = np.zeros((256, 256), dtype=np.uint8)
        m256[100:104, 50:150] = 1

        # Identical shape at 512x512: width 8, length 200
        m512 = np.zeros((512, 512), dtype=np.uint8)
        m512[200:208, 100:300] = 1

        feats_256 = compute_features(crack_mask=m256, pothole_mask=None)
        feats_512 = compute_features(crack_mask=m512, pothole_mask=None)

        rw_256 = feats_256['relative_width']
        rw_512 = feats_512['relative_width']
        rel_diff = abs(rw_256 - rw_512) / rw_512

        print(f"[TEST 08] Scaling invariance: 256x256 rw={rw_256:.6f}, 512x512 rw={rw_512:.6f}, diff={rel_diff*100:.2f}%")
        self.assertLessEqual(rel_diff, 0.05, f"Relative width difference {rel_diff} > 5%")

    def test_09_spur_pruning_zero_junctions(self):
        """
        A line with a 5 px spur gives 0 junctions after skeleton pruning.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Main horizontal crack (area >> 30)
        mask[255:258, 100:300] = 1 # length 200, width 3
        # Vertical spur of 5 px attached at (255, 200)
        mask[250:255, 200:201] = 1 # 5 px spur

        feats = compute_features(crack_mask=mask, pothole_mask=None, prune_branches=True, min_branch_len=12)
        n_junc = feats['n_junctions']
        print(f"[TEST 09] Line with 5 px spur: n_junctions={n_junc}")
        self.assertEqual(n_junc, 0, f"Expected 0 junctions after pruning 5 px spur, got {n_junc}")

    def test_10_t_junction_retained_one_junction(self):
        """
        A T-junction with a 40 px branch gives exactly 1 junction after skeleton pruning.
        """
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Main horizontal crack
        mask[255:258, 100:300] = 1 # length 200, width 3
        # Vertical branch of 40 px attached at (255, 200)
        mask[215:255, 199:202] = 1 # 40 px branch, width 3

        feats = compute_features(crack_mask=mask, pothole_mask=None, prune_branches=True, min_branch_len=12)
        n_junc = feats['n_junctions']
        print(f"[TEST 10] T-junction with 40 px branch: n_junctions={n_junc}")
        self.assertEqual(n_junc, 1, f"Expected 1 junction for 40 px branch, got {n_junc}")


if __name__ == '__main__':
    unittest.main()

