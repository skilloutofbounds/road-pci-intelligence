"""
Human Agreement Evaluation Script for Frozen sPCI (Step 4C Phase 2).
Adheres strictly to the pre-registered analysis protocol:
- Pairwise comparison with tie threshold |sPCI_A - sPCI_B| < 1.0 point.
- 10,000 bootstrap resamples (seed 42) for 95% confidence intervals.
- Paired difference against Damage Area baseline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def load_and_prepare_data(
    test_pairs_csv: str = "results/spci_test_pairs.csv",
    human_labels_csv: str = "labels/pairs_human.csv",
):
    pairs_df = pd.read_csv(test_pairs_csv)
    human_df = pd.read_csv(human_labels_csv)

    # Separate A and B
    df_a = pairs_df[pairs_df['which'] == 'A'].set_index('pair_id')
    df_b = pairs_df[pairs_df['which'] == 'B'].set_index('pair_id')

    records = []
    for _, row in human_df.iterrows():
        p_id = row['pair_id']
        human_label = str(row['worse']).strip()
        
        row_a = df_a.loc[p_id]
        row_b = df_b.loc[p_id]
        
        source = row_a['source']
        spci_a = float(row_a['sPCI'])
        spci_b = float(row_b['sPCI'])
        
        area_a = float(row_a['crack_area_frac'] + row_a['pothole_area_frac'])
        area_b = float(row_b['crack_area_frac'] + row_b['pothole_area_frac'])
        
        spci_gt_a = float(row_a['sPCI_gt'])
        spci_gt_b = float(row_b['sPCI_gt'])
        
        records.append({
            'pair_id': p_id,
            'source': source,
            'image_a': row['image_a'],
            'image_b': row['image_b'],
            'human_label': human_label,
            'spci_a': spci_a,
            'spci_b': spci_b,
            'area_a': area_a,
            'area_b': area_b,
            'spci_gt_a': spci_gt_a,
            'spci_gt_b': spci_gt_b,
        })

    return pd.DataFrame(records)


def compute_agreement(df: pd.DataFrame, tie_threshold: float = 1.0, area_tie_threshold: float = 0.0):
    # Filter to pairs labeled A or B
    valid = df[df['human_label'].isin(['A', 'B'])].copy()
    n_valid = len(valid)
    n_same = len(df) - n_valid

    # sPCI agreement
    # Model picks image with LOWER sPCI as worse
    spci_scores = []
    spci_ties = 0
    spci_agreements = 0
    spci_disagreements = 0

    for _, row in valid.iterrows():
        diff = abs(row['spci_a'] - row['spci_b'])
        lbl = row['human_label']
        
        if diff < tie_threshold:
            spci_scores.append(0.5)
            spci_ties += 1
        elif row['spci_a'] < row['spci_b']:
            # A is worse
            score = 1.0 if lbl == 'A' else 0.0
            spci_scores.append(score)
            if score == 1.0:
                spci_agreements += 1
            else:
                spci_disagreements += 1
        else:
            # B is worse
            score = 1.0 if lbl == 'B' else 0.0
            spci_scores.append(score)
            if score == 1.0:
                spci_agreements += 1
            else:
                spci_disagreements += 1

    valid['spci_score'] = spci_scores

    # Damage Area baseline
    # Larger area = worse
    area_scores = []
    area_ties = 0
    for _, row in valid.iterrows():
        diff = abs(row['area_a'] - row['area_b'])
        lbl = row['human_label']
        
        if diff <= area_tie_threshold:
            area_scores.append(0.5)
            area_ties += 1
        elif row['area_a'] > row['area_b']:
            # A is worse (larger damage area)
            area_scores.append(1.0 if lbl == 'A' else 0.0)
        else:
            # B is worse
            area_scores.append(1.0 if lbl == 'B' else 0.0)

    valid['area_score'] = area_scores

    # Ground-truth sPCI agreement
    gt_scores = []
    gt_ties = 0
    for _, row in valid.iterrows():
        diff = abs(row['spci_gt_a'] - row['spci_gt_b'])
        lbl = row['human_label']
        
        if diff < tie_threshold:
            gt_scores.append(0.5)
            gt_ties += 1
        elif row['spci_gt_a'] < row['spci_gt_b']:
            gt_scores.append(1.0 if lbl == 'A' else 0.0)
        else:
            gt_scores.append(1.0 if lbl == 'B' else 0.0)

    valid['gt_score'] = gt_scores

    return valid, n_valid, n_same, spci_ties, area_ties, gt_ties


def bootstrap_ci(scores: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(scores)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    low = np.percentile(boot_means, 2.5)
    high = np.percentile(boot_means, 97.5)
    return float(np.mean(scores)), float(low), float(high)


def bootstrap_paired_diff_ci(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.RandomState(seed)
    diffs = scores_a - scores_b
    n = len(diffs)
    boot_diffs = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_diffs.append(np.mean(sample))
    boot_diffs = np.array(boot_diffs)
    mean_diff = float(np.mean(diffs))
    low = float(np.percentile(boot_diffs, 2.5))
    high = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, low, high


def main():
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} total pairs from labels/pairs_human.csv")
    print(f"Human label counts: {df['human_label'].value_counts().to_dict()}")

    valid_df, n_valid, n_same, spci_ties, area_ties, gt_ties = compute_agreement(df)
    print("\n" + "=" * 80)
    print(f"EVALUATION DATASET: N = {n_valid} pairs (A or B labeled), {n_same} 'same' pairs excluded")
    print(f"sPCI Ties (|sPCI_A - sPCI_B| < 1.0): {spci_ties}")
    print("=" * 80)

    # 1. Overall Agreement
    spci_scores = np.array(valid_df['spci_score'])
    mean_spci, spci_ci_low, spci_ci_high = bootstrap_ci(spci_scores)
    print(f"\nOVERALL sPCI AGREEMENT: {mean_spci * 100:.2f}% (95% CI: [{spci_ci_low * 100:.2f}%, {spci_ci_high * 100:.2f}%])")

    # 2. Per-Source Agreement
    sources = ['crack500', 'road_crack', 'pothole']
    source_results = {}
    for src in sources:
        src_df = valid_df[valid_df['source'] == src]
        src_scores = np.array(src_df['spci_score'])
        n_src = len(src_df)
        src_ties = int(np.sum(src_scores == 0.5))
        mean_s, low_s, high_s = bootstrap_ci(src_scores)
        source_results[src] = {
            'n': n_src,
            'ties': src_ties,
            'agreement': mean_s,
            'ci_low': low_s,
            'ci_high': high_s
        }
        print(f"  • {src:<12}: N = {n_src:2d} | Ties = {src_ties:2d} | Agreement = {mean_s * 100:6.2f}% (95% CI: [{low_s * 100:6.2f}%, {high_s * 100:6.2f}%])")

    # 3. Baselines
    area_scores = np.array(valid_df['area_score'])
    mean_area, area_ci_low, area_ci_high = bootstrap_ci(area_scores)
    print(f"\nBASELINES:")
    print(f"  • Damage Area Baseline : {mean_area * 100:.2f}% (95% CI: [{area_ci_low * 100:.2f}%, {area_ci_high * 100:.2f}%])")
    print(f"  • Random Guessing      : 50.00%")

    # Reference GT sPCI for crack500 and road_crack
    cr_df = valid_df[valid_df['source'].isin(['crack500', 'road_crack'])]
    cr_gt_scores = np.array(cr_df['gt_score'])
    cr_spci_scores = np.array(cr_df['spci_score'])
    mean_cr_gt, cr_gt_low, cr_gt_high = bootstrap_ci(cr_gt_scores)
    mean_cr_pred, cr_pred_low, cr_pred_high = bootstrap_ci(cr_spci_scores)
    print(f"  • GT sPCI Reference (CRACK500 + Road Crack, N={len(cr_df)}): {mean_cr_gt * 100:.2f}% (95% CI: [{cr_gt_low * 100:.2f}%, {cr_gt_high * 100:.2f}%])")
    print(f"    (vs Pred sPCI on same subset: {mean_cr_pred * 100:.2f}%)")

    # 4. Paired Difference (sPCI - Area Baseline)
    mean_diff, diff_ci_low, diff_ci_high = bootstrap_paired_diff_ci(spci_scores, area_scores)
    print(f"\nPAIRED DIFFERENCE (sPCI Agreement - Area Baseline Agreement):")
    print(f"  • Mean Difference: {mean_diff * 100:+.2f} percentage points")
    print(f"  • 95% Bootstrap CI: [{diff_ci_low * 100:+.2f}%, {diff_ci_high * 100:+.2f}%]")

    if diff_ci_low > 0:
        verdict = "sPCI is statistically significantly BETTER than the area baseline (CI strictly excludes 0)."
    elif diff_ci_high < 0:
        verdict = "sPCI is statistically significantly WORSE than the area baseline (CI strictly excludes 0)."
    else:
        verdict = "No detectable difference between sPCI and area baseline (CI includes 0)."
    print(f"  • Pre-registered Reading: {verdict}")

    # 5. Disagreeing Pairs
    disagreeing = valid_df[valid_df['spci_score'] < 1.0].copy()
    print(f"\nDISAGREEING PAIRS (Score < 1.0): {len(disagreeing)} pairs ({len(disagreeing) - spci_ties} full disagreements, {spci_ties} ties)")
    print(f"{'Pair ID':<10} | {'Source':<12} | {'Human':<6} | {'sPCI A':<8} | {'sPCI B':<8} | {'Model Pick':<10} | {'Score':<6}")
    print("-" * 75)
    for _, r in disagreeing.iterrows():
        diff = abs(r['spci_a'] - r['spci_b'])
        if diff < 1.0:
            pick = "Tie"
        elif r['spci_a'] < r['spci_b']:
            pick = "A"
        else:
            pick = "B"
        print(f"{r['pair_id']:<10} | {r['source']:<12} | {r['human_label']:<6} | {r['spci_a']:8.2f} | {r['spci_b']:8.2f} | {pick:<10} | {r['spci_score']:6.1f}")
    print("=" * 75)

    return valid_df, disagreeing


if __name__ == "__main__":
    main()
