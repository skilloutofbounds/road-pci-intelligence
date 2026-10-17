"""
Unit test suite verifying project facts programmatically (Step 6C).

Verifies:
1. Perception model weights SHA-256 checksum equals EXPECTED_SHA256.
2. Commit hash behind git tag spci-v1-frozen starts with 68a30aa.
3. Human agreement equals 36.0 / 47 = 76.60% computed via app.compute_runtime_human_alignment().
4. Dataset split sizes from manifests/segmentation_manifest.csv match exact documented counts.
5. sPCI parameters in configs/spci_params.yaml match frozen parameters.
"""

import unittest
import subprocess
from pathlib import Path
import pandas as pd
import yaml

from src.pipeline import (
    calculate_sha256,
    EXPECTED_SHA256,
    DEFAULT_CHECKPOINT_CANDIDATES,
)
from app import compute_runtime_human_alignment
from src.spci import load_spci_config


class TestProjectFacts(unittest.TestCase):

    def test_01_weights_sha256(self):
        """Verify weights SHA-256 checksum against EXPECTED_SHA256."""
        found = False
        for p in DEFAULT_CHECKPOINT_CANDIDATES:
            if p.exists() and p.is_file():
                sha = calculate_sha256(p)
                self.assertEqual(
                    sha,
                    EXPECTED_SHA256,
                    f"Checksum mismatch for {p}: expected {EXPECTED_SHA256}, got {sha}"
                )
                found = True
                break
        if not found:
            self.skipTest("Weights not present in clean CI runner")

    def test_02_spci_frozen_tag_commit(self):
        """Verify the commit hash behind git tag spci-v1-frozen starts with 68a30aa."""
        res = subprocess.run(
            ["git", "cat-file", "-p", "spci-v1-frozen"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = res.stdout.strip().splitlines()
        object_line = [l for l in lines if l.startswith("object ")][0]
        commit_hash = object_line.split()[1]
        self.assertTrue(
            commit_hash.startswith("68a30aa"),
            f"Tag spci-v1-frozen points to commit {commit_hash}, expected prefix 68a30aa"
        )

    def test_03_human_agreement_runtime(self):
        """Verify double-blind human agreement computed by app function equals 36.0 / 47."""
        ha = compute_runtime_human_alignment()
        self.assertIsNotNone(ha, "compute_runtime_human_alignment returned None")
        self.assertEqual(ha["n_total"], 60)
        self.assertEqual(ha["n_same"], 13)
        self.assertEqual(ha["n_valid"], 47)
        self.assertEqual(ha["ties"], 2)
        self.assertAlmostEqual(ha["total_points"], 36.0, places=5)
        self.assertAlmostEqual(ha["overall_mean"], 36.0 / 47.0, places=5)
        self.assertAlmostEqual(ha["sources"]["crack_combined"]["points"], 30.0, places=5)
        self.assertEqual(ha["sources"]["crack_combined"]["n"], 33)
        self.assertAlmostEqual(ha["sources"]["pothole"]["points"], 6.0, places=5)
        self.assertEqual(ha["sources"]["pothole"]["n"], 14)

    def test_04_split_sizes_from_manifest(self):
        """Verify split sizes from manifests/segmentation_manifest.csv match exact counts."""
        manifest_path = Path("manifests/segmentation_manifest.csv")
        self.assertTrue(manifest_path.exists(), f"Manifest {manifest_path} not found")
        df = pd.read_csv(manifest_path)
        self.assertEqual(len(df), 1686, f"Expected 1,686 rows, got {len(df)}")

        split_counts = df["split"].value_counts().to_dict()
        self.assertEqual(split_counts.get("train", 0), 1231, "Expected 1,231 train rows")
        self.assertEqual(split_counts.get("val", 0), 255, "Expected 255 val rows")
        self.assertEqual(split_counts.get("test", 0), 200, "Expected 200 test rows")

        # Per source totals
        source_counts = df["source"].value_counts().to_dict()
        self.assertEqual(source_counts.get("crack500", 0), 471)
        self.assertEqual(source_counts.get("pothole", 0), 780)
        self.assertEqual(source_counts.get("road_crack", 0), 435)

        # Per source split breakdown
        sub = df.groupby(["source", "split"]).size().to_dict()
        self.assertEqual(sub[("crack500", "train")], 329)
        self.assertEqual(sub[("crack500", "val")], 70)
        self.assertEqual(sub[("crack500", "test")], 72)

        self.assertEqual(sub[("pothole", "train")], 612)
        self.assertEqual(sub[("pothole", "val")], 108)
        self.assertEqual(sub[("pothole", "test")], 60)

        self.assertEqual(sub[("road_crack", "train")], 290)
        self.assertEqual(sub[("road_crack", "val")], 77)
        self.assertEqual(sub[("road_crack", "test")], 68)

    def test_05_spci_params_match_config(self):
        """Verify sPCI parameters match configs/spci_params.yaml."""
        cfg_path = Path("configs/spci_params.yaml")
        self.assertTrue(cfg_path.exists(), f"Config {cfg_path} not found")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertAlmostEqual(cfg["crack_severity"]["low_threshold_px"], 4.394, places=3)
        self.assertAlmostEqual(cfg["crack_severity"]["high_threshold_px"], 11.588, places=3)
        self.assertAlmostEqual(cfg["network_factor"]["n_ref"], 1.275, places=3)
        self.assertAlmostEqual(cfg["noise_floor"]["tau_short_sides"], 0.00, places=2)

    def test_06_no_contradictory_legacy_claims_in_docs(self):
        """
        Verify that forbidden unverified strings ('81.7%', '+6.7%', '6.30%', 'Spain subset', '$200,000')
        do not appear anywhere in repository documentation (README.md and docs/*.md).
        """
        forbidden_strings = ["81.7%", "+6.7%", "6.30%", "Spain subset", "$200,000"]
        files_to_check = [Path("README.md")] + list(Path("docs").glob("*.md"))

        violations = []
        for file_path in files_to_check:
            if not file_path.exists():
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for s in forbidden_strings:
                if s in text:
                    violations.append(f"{file_path}: contains forbidden string '{s}'")

        self.assertEqual(violations, [], f"Forbidden legacy strings found in documentation:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
