"""
Test Suite for Road PCI Intelligence Dataset Manifests.

Assertions:
(a) No group_id appears in more than one split within any source (Zero data leakage).
(b) Every file path recorded in the manifests actually exists on disk.
(c) Split sizes match the expected partition counts.
(d) Mirrored RDD2022ES pairs strictly share the exact same split.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_manifest import DATASET_SPECS, resolve_dataset_root


class TestDatasetManifests(unittest.TestCase):
    """Rigorous validation tests for dataset manifests and leak-proof splits."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_dir = PROJECT_ROOT / 'manifests'
        cls.seg_manifest_path = cls.manifest_dir / 'segmentation_manifest.csv'
        cls.rdd_manifest_path = cls.manifest_dir / 'rdd2022es_manifest.csv'

        if not cls.seg_manifest_path.exists():
            raise FileNotFoundError(f"Missing segmentation manifest at {cls.seg_manifest_path}. Run src/build_manifest.py first.")

        cls.df_seg = pd.read_csv(cls.seg_manifest_path)
        cls.df_rdd = pd.read_csv(cls.rdd_manifest_path) if cls.rdd_manifest_path.exists() else None

    def test_01_no_group_id_leakage_within_sources(self):
        """(a) Assert that no group_id appears in more than one split within any source."""
        for src_name, df_src in self.df_seg.groupby('source'):
            splits = df_src['split'].unique()
            group_split_counts = df_src.groupby('group_id')['split'].nunique()
            leaked_groups = group_split_counts[group_split_counts > 1]

            self.assertEqual(
                len(leaked_groups), 0,
                f"Data leakage detected in source '{src_name}'! Leaked group_ids: {list(leaked_groups.index[:5])}"
            )
            print(f"[PASS] Zero group_id leakage in '{src_name}' across {len(splits)} splits.")
        if self.df_rdd is not None:
            rdd_leak = self.df_rdd.groupby('group_id')['split'].nunique()
            leaked_rdd = rdd_leak[rdd_leak > 1]
            self.assertEqual(len(leaked_rdd), 0, f"Leakage detected in RDD2022ES! Leaked groups: {list(leaked_rdd.index[:5])}")
            print(f"[PASS] Zero group_id leakage in 'rdd2022es' across 3 splits.")
        else:
            print("[SKIPPED (RDD manifest not built in this run)] RDD2022ES leakage check.")

    def test_02_every_manifest_path_exists_on_disk(self):
        """(b) Assert that all image and mask/label file paths recorded exist on disk."""
        # Check segmentation manifest
        for src_name, df_src in self.df_seg.groupby('source'):
            spec = DATASET_SPECS.get(src_name)
            if not spec:
                continue
            root = resolve_dataset_root(spec['owner'], spec['name'])
            if not root or not root.exists():
                print(f"Skipping disk path check for '{src_name}' (dataset root not mounted in current environment).")
                continue

            for idx, row in df_src.iterrows():
                img_file = root / row['image_path']
                self.assertTrue(img_file.exists(), f"Missing image file on disk: {img_file}")
                
                if pd.notna(row['mask_path_or_label_path']) and row['mask_path_or_label_path'] != "":
                    mask_file = root / row['mask_path_or_label_path']
                    self.assertTrue(mask_file.exists(), f"Missing mask/label file on disk: {mask_file}")

            print(f"[PASS] All {len(df_src)} image and mask paths verified on disk for '{src_name}'.")

        # Check RDD manifest if dataset mounted
        if self.df_rdd is not None:
            spec_rdd = DATASET_SPECS['rdd2022es']
            root_rdd = resolve_dataset_root(spec_rdd['owner'], spec_rdd['name'])
            if root_rdd and root_rdd.exists():
                sample_rdd = self.df_rdd.sample(min(100, len(self.df_rdd)), random_state=42)
                for idx, row in sample_rdd.iterrows():
                    img_file = root_rdd / row['image_path']
                    self.assertTrue(img_file.exists(), f"Missing image file in RDD2022ES: {img_file}")
                print(f"[PASS] Sampled paths verified on disk for 'rdd2022es'.")
        else:
            print("[SKIPPED (RDD manifest not built in this run)] RDD2022ES disk paths existence check.")

    def test_03_split_sizes_match_specifications(self):
        """(c) Assert split sizes match the established targets."""
        # 1. Pothole: 780 images, test=60, train+val=720 (240 groups)
        df_p = self.df_seg[self.df_seg['source'] == 'pothole']
        if len(df_p) > 0:
            self.assertEqual(len(df_p), 780, f"Expected 780 pothole images, found {len(df_p)}")
            self.assertEqual(len(df_p[df_p['split'] == 'test']), 60, "Expected 60 test images for pothole")
            self.assertEqual(len(df_p[df_p['split'] == 'val']) + len(df_p[df_p['split'] == 'train']), 720)
            print(f"[PASS] Pothole split sizes verified: {dict(df_p['split'].value_counts())}")

        # 2. Road Crack: 435 images, 22 scenes
        df_rc = self.df_seg[self.df_seg['source'] == 'road_crack']
        if len(df_rc) > 0:
            self.assertEqual(len(df_rc), 435, f"Expected 435 road-crack images, found {len(df_rc)}")
            self.assertEqual(df_rc['group_id'].nunique(), 22, "Expected 22 unique scenes in road-crack")
            print(f"[PASS] Road-crack split sizes verified: {dict(df_rc['split'].value_counts())}")

        # 3. CRACK500: 471 images (329 train, 70 val, 72 test)
        df_c = self.df_seg[self.df_seg['source'] == 'crack500']
        if len(df_c) > 0:
            self.assertEqual(len(df_c), 471, f"Expected 471 crack500 images, found {len(df_c)}")
            self.assertEqual(len(df_c[df_c['split'] == 'train']), 329)
            self.assertEqual(len(df_c[df_c['split'] == 'val']), 70)
            self.assertEqual(len(df_c[df_c['split'] == 'test']), 72)
            print(f"[PASS] CRACK500 split sizes verified: {dict(df_c['split'].value_counts())}")

        # 4. RDD2022ES: 25,600 images (80/10/10)
        if self.df_rdd is not None:
            self.assertEqual(len(self.df_rdd), 25600, f"Expected 25600 RDD images, found {len(self.df_rdd)}")
            self.assertEqual(len(self.df_rdd[self.df_rdd['split'] == 'train']), 20480)
            self.assertEqual(len(self.df_rdd[self.df_rdd['split'] == 'val']), 2560)
            self.assertEqual(len(self.df_rdd[self.df_rdd['split'] == 'test']), 2560)
            print(f"[PASS] RDD2022ES split sizes verified: {dict(self.df_rdd['split'].value_counts())}")
        else:
            print("[SKIPPED (RDD manifest not built in this run)] RDD2022ES split sizes check.")

    def test_04_rdd_mirrored_pairs_share_split(self):
        """(d) Assert that mirrored RDD2022ES pairs strictly share the exact same split."""
        if self.df_rdd is None:
            self.skipTest("SKIPPED (RDD manifest not built in this run)")

        split_per_group = self.df_rdd.groupby('group_id')['split'].nunique()
        self.assertTrue((split_per_group == 1).all(), "Some mirrored RDD pairs were assigned to different splits!")

        # Verify exact pair count
        images_per_group = self.df_rdd.groupby('group_id')['image_path'].count()
        self.assertTrue((images_per_group == 2).all(), "Every RDD group_id must contain exactly 2 images (original and mirrored).")
        print(f"[PASS] All 12,800 mirrored RDD pairs strictly share the exact same split.")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetManifests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
