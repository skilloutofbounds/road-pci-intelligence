"""
Generates the error analysis figure displaying 6 disagreeing pairs between
human visual rating and frozen sPCI model ranking.
Crops high-resolution imagery directly from pre-rendered test pair sheets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def generate_figure():
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "human_agreement_disagreements.png"

    # Load evaluated pairs
    test_pairs = pd.read_csv("results/spci_test_pairs.csv")
    human_labels = pd.read_csv("labels/pairs_human.csv").set_index('pair_id')
    
    df_a = test_pairs[test_pairs['which'] == 'A'].set_index('pair_id')
    df_b = test_pairs[test_pairs['which'] == 'B'].set_index('pair_id')

    # Selected 6 diverse disagreeing pairs: 2 crack500, 1 road_crack, 3 pothole
    selected_pids = ['pair_004', 'pair_007', 'pair_037', 'pair_042', 'pair_045', 'pair_049']

    # Page mapping (6 pairs per page, 1-indexed)
    # pair_001..pair_006 -> Page 1
    # pair_007..pair_012 -> Page 2
    # pair_037..pair_042 -> Page 7
    # pair_043..pair_048 -> Page 8
    # pair_049..pair_054 -> Page 9
    
    fig, axes = plt.subplots(6, 2, figsize=(14, 22))
    plt.subplots_adjust(top=0.95, bottom=0.03, hspace=0.45, wspace=0.15)
    fig.suptitle(
        "Disagreement Analysis: Human Visual Ranking vs. Frozen sPCI (6 Selected Pairs)",
        fontsize=15, fontweight='bold', y=0.98
    )

    for i, pid in enumerate(selected_pids):
        p_num = int(pid.split('_')[1])
        page_num = (p_num - 1) // 6 + 1
        row_in_page = (p_num - 1) % 6

        # Open page sheet
        page_path = Path(f"labels/pair_sheet_page_{page_num:02d}.png")
        page_img = Image.open(page_path)
        W, H = page_img.size

        # Crop corresponding row (approximate vertical slicing accounting for header/footer)
        header_h = int(H * 0.04)
        usable_h = H - header_h - int(H * 0.02)
        row_h = usable_h / 6.0
        y0 = int(header_h + row_in_page * row_h)
        y1 = int(y0 + row_h)

        # Slice A (left half) and B (right half)
        # Account for border margins and center divider
        x_mid = W // 2
        img_a_crop = page_img.crop((int(W * 0.02), y0, x_mid - int(W * 0.01), y1))
        img_b_crop = page_img.crop((x_mid + int(W * 0.01), y0, int(W * 0.98), y1))

        row_a = df_a.loc[pid]
        row_b = df_b.loc[pid]
        h_label = human_labels.loc[pid, 'worse']
        src = row_a['source']
        spci_a = row_a['sPCI']
        spci_b = row_b['sPCI']
        model_pick = "A" if spci_a < spci_b else ("B" if spci_b < spci_a else "Tie")

        # Plot A
        ax_a = axes[i, 0]
        ax_a.imshow(img_a_crop)
        ax_a.axis('off')
        ax_a.set_title(
            f"[{pid} Image A] {src.upper()}\nsPCI: {spci_a:.1f} ({row_a['condition_band']}) | Crack: {row_a['crack_area_frac']*100:.2f}%, Pot: {row_a['pothole_area_frac']*100:.2f}%",
            fontsize=10, fontweight='bold' if model_pick == 'A' else 'normal'
        )

        # Plot B
        ax_b = axes[i, 1]
        ax_b.imshow(img_b_crop)
        ax_b.axis('off')
        ax_b.set_title(
            f"[{pid} Image B] {src.upper()}\nsPCI: {spci_b:.1f} ({row_b['condition_band']}) | Crack: {row_b['crack_area_frac']*100:.2f}%, Pot: {row_b['pothole_area_frac']*100:.2f}%",
            fontsize=10, fontweight='bold' if model_pick == 'B' else 'normal'
        )

        # Add caption below the pair
        diff_str = f"|sPCI_A - sPCI_B| = {abs(spci_a - spci_b):.1f}"
        fig.text(
            0.5, 0.95 - (i * 0.155) - 0.13,
            f"Pair: {pid} | Human Label: '{h_label}' Worse  vs.  Model Decision: '{model_pick}' Worse ({diff_str} pts)",
            ha='center', fontsize=9.5, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec="#cbd5e1", lw=1)
        )

    plt.savefig(out_fig, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {out_fig} ({out_fig.stat().st_size:,} bytes)")

if __name__ == "__main__":
    generate_figure()
