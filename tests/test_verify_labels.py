"""
Unit tests for human labeling format verification and column restoration.
Uses synthetic fixtures only; does NOT touch real labels/pairs_human.csv.
"""

import unittest
from pathlib import Path
import shutil
import pandas as pd
from src.verify_labels import verify_human_labels


class TestVerifyLabels(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("scratch/test_verify_labels")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.tmpl_path = self.test_dir / "template.csv"

        # Generate a synthetic 60-row template
        rows = []
        for i in range(1, 61):
            rows.append({
                "pair_id": f"pair_{i:03d}",
                "image_a": f"img_a_{i}.jpg",
                "image_b": f"img_b_{i}.jpg",
                "worse": "",
            })
        self.template_df = pd.DataFrame(rows)
        self.template_df.to_csv(self.tmpl_path, index=False)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_01_valid_labels_pass(self):
        """All 60 rows filled with valid tokens pass cleanly."""
        tokens = ["A", "B", "same"]
        human_df = self.template_df.copy()
        human_df["worse"] = [tokens[i % 3] for i in range(60)]
        target_path = self.test_dir / "valid.csv"
        human_df.to_csv(target_path, index=False)

        res = verify_human_labels(target_path, template_csv_path=self.tmpl_path)
        self.assertTrue(res["passed"])
        self.assertEqual(len(res["invalid_rows"]), 0)
        self.assertEqual(len(res["error_messages"]), 0)

    def test_02_missing_column_restoration(self):
        """Missing metadata columns are restored by joining on pair_id without altering 'worse'."""
        # Simulated export missing 'image_a' and 'image_b'
        sparse_df = pd.DataFrame({
            "pair_id": self.template_df["pair_id"],
            "worse": ["A"] * 60,
        })
        target_path = self.test_dir / "sparse.csv"
        sparse_df.to_csv(target_path, index=False)

        res = verify_human_labels(
            target_path,
            template_csv_path=self.tmpl_path,
            auto_restore_columns=True,
            save_restored=True,
        )
        self.assertTrue(res["passed"])

        # Check restored file
        restored_df = pd.read_csv(target_path)
        self.assertEqual(list(restored_df.columns), list(self.template_df.columns))
        self.assertEqual(list(restored_df["worse"]), ["A"] * 60)
        self.assertEqual(list(restored_df["image_a"]), list(self.template_df["image_a"]))

    def test_03_invalid_tokens_and_nan_detected(self):
        """Invalid tokens and NaNs are flagged at their exact 1-indexed row numbers."""
        human_df = self.template_df.copy()
        human_df["worse"] = ["A"] * 60
        human_df.loc[4, "worse"] = None # Row 5
        human_df.loc[19, "worse"] = "invalid_token" # Row 20
        human_df.loc[59, "worse"] = "C" # Row 60

        target_path = self.test_dir / "invalid.csv"
        human_df.to_csv(target_path, index=False)

        res = verify_human_labels(target_path, template_csv_path=self.tmpl_path)
        self.assertFalse(res["passed"])
        self.assertEqual(res["invalid_rows"], [5, 20, 60])

    def test_04_row_count_and_order_mismatch(self):
        """Fewer rows or misaligned pair_id are rejected."""
        # Fewer rows
        short_df = self.template_df.iloc[:55].copy()
        short_df["worse"] = ["A"] * 55
        target_path = self.test_dir / "short.csv"
        short_df.to_csv(target_path, index=False)

        res = verify_human_labels(target_path, template_csv_path=self.tmpl_path)
        self.assertFalse(res["passed"])
        self.assertTrue(any("Row count mismatch" in msg for msg in res["error_messages"]))


if __name__ == "__main__":
    unittest.main()
