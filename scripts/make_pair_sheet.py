"""
Script to generate human labeling sheet and template CSV from test split images.

Selects 20 random pairs from each of crack500, road_crack, and pothole test splits
(60 pairs total, seed 42, no image reused).
Renders 10 PNG pages of 6 pairs each (Image A left, Image B right), labeled only
with 'pair_XXX A' and 'pair_XXX B' (blinded: no filenames, no model outputs, no masks).
Exports labels/pairs_template.csv with pair metadata and an empty 'worse' column.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def resolve_roots():
    # Common locations on Kaggle or local
    candidates = {
        'crack500': [
            Path('/kaggle/input/datasets/sj26717/crack500'),
            Path('/kaggle/input/crack500'),
            Path('data/crack500')
        ],
        'road_crack': [
            Path('/kaggle/input/datasets/rukiyeaydn/road-crack-dataset'),
            Path('/kaggle/input/road-crack-dataset'),
            Path('data/road_crack')
        ],
        'pothole': [
            Path('/kaggle/input/datasets/farzadnekouei/pothole-image-segmentation-dataset'),
            Path('/kaggle/input/pothole-image-segmentation-dataset'),
            Path('data/pothole')
        ]
    }
    roots = {}
    for src, cands in candidates.items():
        for c in cands:
            if c.exists() and c.is_dir():
                roots[src] = c.resolve()
                break
        if src not in roots:
            # Fallback search under /kaggle/input
            for d in Path('/kaggle/input').rglob(src):
                if d.is_dir():
                    roots[src] = d.resolve()
                    break
    return roots


def make_pairs(manifest_path: Path, seed: int = 42):
    df = pd.read_csv(manifest_path)
    test_df = df[df['split'] == 'test'].copy()

    rng = np.random.RandomState(seed)
    pairs = []
    pair_counter = 1

    for src in ['crack500', 'road_crack', 'pothole']:
        src_df = test_df[test_df['source'] == src].copy()
        img_paths = sorted(src_df['image_path'].unique().tolist())
        rng.shuffle(img_paths)

        # Pick 20 pairs (40 images)
        selected = img_paths[:40]
        if len(selected) < 40:
            raise ValueError(f"Not enough test images in {src}: {len(selected)} < 40")

        for i in range(0, 40, 2):
            pairs.append({
                'pair_id': f"pair_{pair_counter:03d}",
                'source': src,
                'image_a': selected[i],
                'image_b': selected[i + 1],
                'worse': ''
            })
            pair_counter += 1

    return pd.DataFrame(pairs)


def render_pages(pairs_df: pd.DataFrame, roots: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pairs = len(pairs_df)
    pairs_per_page = 6
    n_pages = (n_pairs + pairs_per_page - 1) // pairs_per_page

    for page_idx in range(n_pages):
        start_idx = page_idx * pairs_per_page
        end_idx = min(start_idx + pairs_per_page, n_pairs)
        page_pairs = pairs_df.iloc[start_idx:end_idx]

        fig, axes = plt.subplots(len(page_pairs), 2, figsize=(12, 3.6 * len(page_pairs)))
        if len(page_pairs) == 1:
            axes = np.array([axes])

        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.35, wspace=0.15)
        fig.suptitle(f"Pavement Distress Comparison Sheet — Page {page_idx + 1:02d} of {n_pages:02d}", fontsize=14, fontweight='bold', y=0.99)

        for row_i, (_, row) in enumerate(page_pairs.iterrows()):
            pid = row['pair_id']
            src = row['source']
            root = roots.get(src)

            # Load Image A
            ax_a = axes[row_i, 0]
            if root and (root / row['image_a']).exists():
                with Image.open(root / row['image_a']) as img:
                    ax_a.imshow(img.convert('RGB'))
            else:
                ax_a.text(0.5, 0.5, f"[Image not found]\n{row['image_a']}", ha='center', va='center')
            ax_a.set_title(f"{pid} A", fontsize=11, fontweight='bold')
            ax_a.axis('off')

            # Load Image B
            ax_b = axes[row_i, 1]
            if root and (root / row['image_b']).exists():
                with Image.open(root / row['image_b']) as img:
                    ax_b.imshow(img.convert('RGB'))
            else:
                ax_b.text(0.5, 0.5, f"[Image not found]\n{row['image_b']}", ha='center', va='center')
            ax_b.set_title(f"{pid} B", fontsize=11, fontweight='bold')
            ax_b.axis('off')

        page_out = out_dir / f"pair_sheet_page_{page_idx + 1:02d}.png"
        plt.savefig(page_out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved {page_out}")


def main():
    parser = argparse.ArgumentParser(description="Generate human labeling sheet.")
    parser.add_argument("--manifest", type=str, default="manifests/segmentation_manifest.csv")
    parser.add_argument("--out-csv", type=str, default="labels/pairs_template.csv")
    parser.add_argument("--out-dir", type=str, default="labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Generating 60 test pairs with seed {args.seed}...")
    pairs_df = make_pairs(manifest_path, seed=args.seed)

    # Save CSV template with required columns
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # The prompt requests columns: pair_id, image_a, image_b, worse
    template_df = pairs_df[['pair_id', 'image_a', 'image_b', 'worse']]
    template_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved template to {out_csv} ({len(template_df)} pairs)")

    # Render PNG pages if dataset roots are available
    roots = resolve_roots()
    print(f"Resolved roots: {roots}")
    if any(r is not None for r in roots.values()):
        print("Rendering PNG pages...")
        render_pages(pairs_df, roots, Path(args.out_dir))
    else:
        print("[NOTICE] Dataset roots not located locally. Rendering placeholder sheets or run on Kaggle.")


if __name__ == '__main__':
    main()
