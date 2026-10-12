"""
Verification and alignment utility for human pairwise labeling data.
Adheres strictly to blinding: never inspects, logs, prints, or summarizes
the semantic contents of the 'worse' column beyond validation against
allowed tokens {'A', 'B', 'same'}.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd


def verify_human_labels(
    human_csv_path: Union[str, Path],
    template_csv_path: Union[str, Path] = "labels/pairs_template.csv",
    auto_restore_columns: bool = True,
    save_restored: bool = False,
) -> Dict[str, Any]:
    """
    Verifies human labeling format against the template specification.

    Checks:
    1. Exactly 60 rows matching template length.
    2. pair_id matches template exactly in order.
    3. No NaN / empty values in 'worse'.
    4. Valid values strictly in {'A', 'B', 'same'}.

    If human_csv is missing columns present in template_csv (e.g., 'source', 'image_a', 'image_b'),
    it restores them by joining on 'pair_id' from the template without modifying 'worse'.

    Returns
    -------
    dict
        {"passed": bool, "invalid_rows": List[int], "error_messages": List[str]}
    """
    human_path = Path(human_csv_path)
    tmpl_path = Path(template_csv_path)

    if not human_path.exists():
        return {
            "passed": False,
            "invalid_rows": [],
            "error_messages": [f"File not found: {human_path}"],
        }

    if not tmpl_path.exists():
        return {
            "passed": False,
            "invalid_rows": [],
            "error_messages": [f"Template not found: {tmpl_path}"],
        }

    template_df = pd.read_csv(tmpl_path)
    human_df = pd.read_csv(human_path)

    # 1. Missing columns restoration
    template_cols = list(template_df.columns)
    missing_cols = [c for c in template_cols if c not in human_df.columns]
    if missing_cols and auto_restore_columns:
        if "pair_id" not in human_df.columns:
            return {
                "passed": False,
                "invalid_rows": [],
                "error_messages": ["CRITICAL: human_csv is missing 'pair_id'; cannot join on template."],
            }
        # Keep user's 'worse' column intact and join template metadata
        restore_cols = ["pair_id"] + [c for c in missing_cols if c != "worse"]
        merged = pd.merge(template_df[restore_cols], human_df, on="pair_id", how="right")
        # Reorder to match template column order + any extra columns
        ordered_cols = [c for c in template_cols if c in merged.columns] + [
            c for c in merged.columns if c not in template_cols
        ]
        human_df = merged[ordered_cols]
        if save_restored:
            human_df.to_csv(human_path, index=False)

    errors = []
    invalid_rows = []

    # 2. Check row count
    if len(human_df) != len(template_df):
        errors.append(f"Row count mismatch: expected {len(template_df)}, found {len(human_df)}.")

    # 3. Check pair_id matching
    if "pair_id" in human_df.columns and "pair_id" in template_df.columns:
        if list(human_df["pair_id"]) != list(template_df["pair_id"]):
            errors.append("pair_id values or order do not match template exactly.")
    else:
        errors.append("Missing 'pair_id' column.")

    # 4. Check 'worse' column values
    if "worse" not in human_df.columns:
        errors.append("Missing required 'worse' column.")
    else:
        valid_tokens = {"A", "B", "same"}
        for idx, val in enumerate(human_df["worse"]):
            row_num = idx + 1 # 1-indexed row number
            if pd.isna(val):
                invalid_rows.append(row_num)
            else:
                str_val = str(val).strip()
                if str_val not in valid_tokens:
                    invalid_rows.append(row_num)

        if invalid_rows:
            errors.append(f"Found invalid or empty 'worse' entries at rows: {invalid_rows}.")

    passed = (len(errors) == 0 and len(invalid_rows) == 0)
    return {
        "passed": passed,
        "invalid_rows": invalid_rows,
        "error_messages": errors,
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "labels/pairs_human.csv"
    res = verify_human_labels(target)
    if res["passed"]:
        print("PASS")
    else:
        print("FAIL")
        if res["invalid_rows"]:
            print(f"Invalid row numbers: {res['invalid_rows']}")
        for msg in res["error_messages"]:
            print(f"  • {msg}")
