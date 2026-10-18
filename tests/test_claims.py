"""
Unit test suite verifying every public claim in docs/claims.csv.

Asserts:
1. Every source_file exists on disk.
2. Every value string is contained in the source_file.
3. Every source_locator string is contained in the source_file.
4. Prints the exact number of rows checked.
"""

import csv
import unittest
from pathlib import Path

CLAIMS_CSV_PATH = Path("docs/claims.csv")


def verify_claims_file(claims_path: Path):
    """Verify all claims in a given CSV file against repository sources."""
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims CSV not found: {claims_path}")

    with open(claims_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    checked_count = 0
    errors = []
    for idx, row in enumerate(rows, start=1):
        claim_id = row.get("claim_id", f"row_{idx}")
        value = row.get("value", "")
        source_file = row.get("source_file", "")
        locator = row.get("source_locator", "")

        fpath = Path(source_file)
        if not fpath.exists():
            errors.append(f"[{claim_id}] Source file '{source_file}' does not exist on disk.")
            continue

        content = fpath.read_text(encoding="utf-8")

        if value not in content:
            errors.append(f"[{claim_id}] Value '{value}' not found in '{source_file}'.")

        if locator and locator not in content:
            errors.append(f"[{claim_id}] Locator '{locator}' not found in '{source_file}'.")

        checked_count += 1

    return checked_count, errors


class TestClaimsTraceability(unittest.TestCase):

    def test_all_claims_traceable_to_source(self):
        """Verify that every row in docs/claims.csv exists and contains value string."""
        checked_count, errors = verify_claims_file(CLAIMS_CSV_PATH)
        print(f"\n[Claims Check] Checked {checked_count} claim rows from {CLAIMS_CSV_PATH} successfully.")
        self.assertEqual(len(errors), 0, f"Claims verification failed with {len(errors)} error(s):\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
