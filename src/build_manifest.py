"""
Manifest Builder for Road PCI Intelligence.

Generates leak-proof, group-aware dataset manifests for road distress segmentation
and object detection tasks. All paths in manifests are recorded relative to the
respective dataset root to maintain portability across Kaggle and local environments.
"""

import os
import sys
import re
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

try:
    from src.data_rules import SOURCE_CAPABILITIES
except ImportError:
    from data_rules import SOURCE_CAPABILITIES

# Global random seed for reproducible splits
RANDOM_SEED = 42

# Dataset registry definitions
DATASET_SPECS = {
    'pothole': {
        'owner': 'farzadnekouei',
        'name': 'pothole-image-segmentation-dataset',
        'display_name': 'Pothole Image Segmentation'
    },
    'road_crack': {
        'owner': 'rukiyeaydn',
        'name': 'road-crack-dataset',
        'display_name': 'Road Crack Dataset'
    },
    'crack500': {
        'owner': 'sj26717',
        'name': 'crack500',
        'display_name': 'CRACK500'
    },
    'rdd2022es': {
        'owner': 'juusos',
        'name': 'rdd2022es',
        'display_name': 'RDD2022ES'
    }
}


def resolve_dataset_root(owner: str, name: str) -> Optional[Path]:
    """
    Locates the dataset root directory on Kaggle or locally.
    Checks /kaggle/input/datasets/<owner>/<name>/ first, then fallback paths.
    """
    candidates = [
        Path('/kaggle/input/datasets') / owner / name,
        Path('/kaggle/input') / owner / name,
        Path('/kaggle/input') / name,
        Path('data') / name,
        Path('../data') / name
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()

    # Recursive fallback
    k_input = Path('/kaggle/input')
    if k_input.exists():
        for d in k_input.rglob(name):
            if d.is_dir():
                return d.resolve()

    return None


def get_image_dimensions(img_path: Path) -> Tuple[int, int]:
    """Reads image (width, height) without loading the full raster pixel buffer."""
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (640, 640)


# ==============================================================================
# 1. POTHOLE DATASET MANIFEST BUILDER
# ==============================================================================
def build_pothole_manifest(dataset_root: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Builds manifest for farzadnekouei/pothole-image-segmentation-dataset.

    Split rules:
    - group_id = 'pic-<N>' from filename (e.g. 'pic-282').
    - test = official valid/ folder (un-augmented validation scenes).
    - val = 15% of train group_ids (all augmented variants stay together).
    - train = remaining 85% of train group_ids.
    - labels_crack = False, labels_pothole = True.
    """
    records = []
    train_dir = dataset_root / 'Pothole_Segmentation_YOLOv8' / 'train'
    valid_dir = dataset_root / 'Pothole_Segmentation_YOLOv8' / 'valid'

    # Fallback to search if directory structure varies slightly
    if not train_dir.exists():
        matches = list(dataset_root.rglob('train'))
        if matches:
            train_dir = matches[0]
            valid_dir = train_dir.parent / 'valid'

    train_imgs = sorted(list((train_dir / 'images').glob('*.jpg')))
    valid_imgs = sorted(list((valid_dir / 'images').glob('*.jpg')))

    def get_pothole_group_id(filename: str) -> str:
        m = re.match(r'^(pic-\d+)', filename, re.IGNORECASE)
        return m.group(1).lower() if m else Path(filename).stem

    # 1. Assign test split from official valid/ directory
    for img_p in valid_imgs:
        label_p = (valid_dir / 'labels' / f"{img_p.stem}.txt")
        rel_img = img_p.relative_to(dataset_root).as_posix()
        rel_lbl = label_p.relative_to(dataset_root).as_posix() if label_p.exists() else ""
        w, h = get_image_dimensions(img_p)
        gid = get_pothole_group_id(img_p.name)

        records.append({
            'source': 'pothole',
            'image_path': rel_img,
            'mask_path_or_label_path': rel_lbl,
            'group_id': gid,
            'split': 'test',
            'width': w,
            'height': h,
            'labels_crack': False,
            'labels_pothole': True
        })

    # 2. Partition train/ directory into train and val by group_id
    train_group_to_imgs = defaultdict(list)
    for img_p in train_imgs:
        gid = get_pothole_group_id(img_p.name)
        train_group_to_imgs[gid].append(img_p)

    unique_train_groups = sorted(list(train_group_to_imgs.keys()))
    rng = random.Random(seed)
    shuffled_groups = unique_train_groups.copy()
    rng.shuffle(shuffled_groups)

    n_val = max(1, int(round(len(shuffled_groups) * 0.15)))
    val_groups = set(shuffled_groups[:n_val])
    train_groups = set(shuffled_groups[n_val:])

    for gid, img_list in train_group_to_imgs.items():
        split = 'val' if gid in val_groups else 'train'
        for img_p in img_list:
            label_p = (train_dir / 'labels' / f"{img_p.stem}.txt")
            rel_img = img_p.relative_to(dataset_root).as_posix()
            rel_lbl = label_p.relative_to(dataset_root).as_posix() if label_p.exists() else ""
            w, h = get_image_dimensions(img_p)

            records.append({
                'source': 'pothole',
                'image_path': rel_img,
                'mask_path_or_label_path': rel_lbl,
                'group_id': gid,
                'split': split,
                'width': w,
                'height': h,
                'labels_crack': False,
                'labels_pothole': True
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# 2. ROAD-CRACK DATASET MANIFEST BUILDER
# ==============================================================================
def build_road_crack_manifest(dataset_root: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Builds manifest for rukiyeaydn/road-crack-dataset.

    Split rules:
    - group_id = scene (e.g. 'road_<N>' or 'Production_1_ortho_merge').
    - Partition scenes targeting approximately 70% train / 15% val / 15% test by image count.
    - All crops/tiles of a scene go to the exact same split (zero scene leakage).
    - labels_crack = True, labels_pothole = False.
    """
    records = []
    # Collect all image files (excluding _mask.png)
    img_files = []
    for root, _, files in os.walk(dataset_root):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in {'.jpg', '.png'} and not f.endswith('_mask.png') and '_mask' not in p.stem:
                img_files.append(p)

    def extract_scene_group_id(filename: str) -> str:
        stem = Path(filename).stem
        if '.rf.' in filename:
            key1 = filename.split('.rf.')[0]
        else:
            key1 = stem
        key1 = re.sub(r'(_jpg|_png)$', '', key1, flags=re.IGNORECASE)

        m_road = re.match(r'^(road_\d+)', key1)
        if m_road:
            return m_road.group(1)
        elif key1.startswith('Production_1_ortho_merge'):
            return 'Production_1_ortho_merge'
        else:
            return re.sub(r'(_\d+)+$', '', key1)

    scene_to_images = defaultdict(list)
    for img_p in img_files:
        scene = extract_scene_group_id(img_p.name)
        scene_to_images[scene].append(img_p)

    total_images = len(img_files)
    target_train = total_images * 0.70
    target_val = total_images * 0.15
    target_test = total_images * 0.15

    # Deterministic scene allocation
    rng = random.Random(seed)
    scene_items = sorted(list(scene_to_images.items()), key=lambda x: x[0])
    rng.shuffle(scene_items)

    scene_split_map = {}
    train_c, val_c, test_c = 0, 0, 0

    for sc, imgs in scene_items:
        count = len(imgs)
        # Allocate to the split furthest below its target percentage
        ratios = [
            (train_c / target_train, 0),
            (val_c / target_val, 1),
            (test_c / target_test, 2)
        ]
        ratios.sort(key=lambda x: x[0])
        chosen_split = ratios[0][1]

        if chosen_split == 0:
            scene_split_map[sc] = 'train'
            train_c += count
        elif chosen_split == 1:
            scene_split_map[sc] = 'val'
            val_c += count
        else:
            scene_split_map[sc] = 'test'
            test_c += count

    print(f"\n[road-crack] Scene Allocation (Total: {total_images} images across {len(scene_to_images)} scenes):")
    print(f"  • Train: {train_c} images ({train_c/total_images*100:.1f}%) -> {sorted([sc for sc, sp in scene_split_map.items() if sp=='train'])}")
    print(f"  • Val:   {val_c} images ({val_c/total_images*100:.1f}%) -> {sorted([sc for sc, sp in scene_split_map.items() if sp=='val'])}")
    print(f"  • Test:  {test_c} images ({test_c/total_images*100:.1f}%) -> {sorted([sc for sc, sp in scene_split_map.items() if sp=='test'])}")

    for sc, imgs in scene_to_images.items():
        split = scene_split_map[sc]
        for img_p in imgs:
            # Mask matching: image.jpg -> image_mask.png
            mask_name = f"{img_p.stem}_mask.png"
            mask_p = img_p.parent / mask_name
            if not mask_p.exists():
                # Alternative search
                base_name = img_p.name.replace('.jpg', '').replace('.png', '')
                cands = list(img_p.parent.glob(f"{base_name}*mask.png"))
                mask_p = cands[0] if cands else None

            rel_img = img_p.relative_to(dataset_root).as_posix()
            rel_mask = mask_p.relative_to(dataset_root).as_posix() if mask_p and mask_p.exists() else ""
            w, h = get_image_dimensions(img_p)

            records.append({
                'source': 'road_crack',
                'image_path': rel_img,
                'mask_path_or_label_path': rel_mask,
                'group_id': sc,
                'split': split,
                'width': w,
                'height': h,
                'labels_crack': True,
                'labels_pothole': False
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# 3. CRACK500 DATASET MANIFEST BUILDER
# ==============================================================================
def build_crack500_manifest(dataset_root: Path) -> pd.DataFrame:
    """
    Builds manifest for sj26717/crack500.

    Split rules:
    - Uses official CRACK500/splits/train.lst, val.lst, test.lst.
    - group_id = base image stem (e.g. '20160222_165402').
    - labels_crack = True, labels_pothole = False.
    """
    records = []
    splits_dir = dataset_root / 'CRACK500' / 'splits'
    if not splits_dir.exists():
        splits_dirs = list(dataset_root.rglob('splits'))
        splits_dir = splits_dirs[0] if splits_dirs else dataset_root

    split_files = {
        'train': splits_dir / 'train.lst',
        'val': splits_dir / 'val.lst',
        'test': splits_dir / 'test.lst'
    }

    for split_name, lst_path in split_files.items():
        if not lst_path.exists():
            print(f"Warning: {lst_path} does not exist!")
            continue

        with open(lst_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                rel_img_raw = parts[0]
                rel_mask_raw = parts[1]
            else:
                rel_img_raw = parts[0]
                rel_mask_raw = parts[0].replace('JPEGImages/images/', 'Annotations/masks/').replace('.jpg', '_mask.png')

            # Build full paths under dataset_root
            img_p = dataset_root / 'CRACK500' / rel_img_raw
            mask_p = dataset_root / 'CRACK500' / rel_mask_raw
            if not img_p.exists():
                img_p = dataset_root / rel_img_raw
                mask_p = dataset_root / rel_mask_raw

            w, h = get_image_dimensions(img_p) if img_p.exists() else (2000, 1500)
            gid = Path(rel_img_raw).stem

            # Record relative to dataset_root
            rel_img = img_p.relative_to(dataset_root).as_posix() if img_p.exists() else rel_img_raw
            rel_mask = mask_p.relative_to(dataset_root).as_posix() if mask_p.exists() else rel_mask_raw

            records.append({
                'source': 'crack500',
                'image_path': rel_img,
                'mask_path_or_label_path': rel_mask,
                'group_id': gid,
                'split': split_name,
                'width': w,
                'height': h,
                'labels_crack': True,
                'labels_pothole': False
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# 4. RDD2022ES DATASET MANIFEST BUILDER (BOUNDING BOXES ONLY)
# ==============================================================================
def build_rdd2022es_manifest(dataset_root: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Builds manifest for juusos/rdd2022es (Bounding box dataset).

    Split rules:
    - group_id = filename without 'xmirror_' prefix (e.g. 'Czech_000000').
    - 80% train / 10% val / 10% test partitioned deterministically by group_id.
    - Mirrored pairs ('xmirror_...') strictly share the exact same split as their original.
    - labels_crack = False, labels_pothole = False (not used for segmentation loss).
    """
    records = []
    annotated_dir = dataset_root / 'combined_annotatedv2'
    if not annotated_dir.exists():
        dirs = list(dataset_root.rglob('combined_annotatedv2'))
        annotated_dir = dirs[0] if dirs else dataset_root

    img_files = sorted(list(annotated_dir.glob('*.jpg')))

    def get_rdd_group_id(filename: str) -> str:
        stem = Path(filename).stem
        if stem.startswith('xmirror_'):
            return stem[len('xmirror_'):]
        return stem

    # Group images by original scene
    group_to_imgs = defaultdict(list)
    for img_p in img_files:
        gid = get_rdd_group_id(img_p.name)
        group_to_imgs[gid].append(img_p)

    unique_groups = sorted(list(group_to_imgs.keys()))
    rng = random.Random(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    n_total = len(shuffled_groups)
    n_train = int(round(n_total * 0.80))
    n_val = int(round(n_total * 0.10))

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])

    for gid, imgs in group_to_imgs.items():
        if gid in train_groups:
            split = 'train'
        elif gid in val_groups:
            split = 'val'
        else:
            split = 'test'

        for img_p in imgs:
            txt_p = img_p.with_suffix('.txt')
            rel_img = img_p.relative_to(dataset_root).as_posix()
            rel_txt = txt_p.relative_to(dataset_root).as_posix() if txt_p.exists() else ""
            w, h = get_image_dimensions(img_p)

            records.append({
                'source': 'rdd2022es',
                'image_path': rel_img,
                'mask_path_or_label_path': rel_txt,
                'group_id': gid,
                'split': split,
                'width': w,
                'height': h,
                'labels_crack': False,
                'labels_pothole': False
            })

    df = pd.DataFrame(records)
    return df


# ==============================================================================
# MAIN DRIVER: GENERATE AND EXPORT ALL MANIFESTS
# ==============================================================================
def generate_all_manifests(output_dir: Path = Path('manifests')) -> Dict[str, pd.DataFrame]:
    """
    Discovers all 4 datasets, generates their manifests, and exports CSV files
    to output_dir (root/manifests/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}

    print("=" * 80)
    print("ROAD PCI INTELLIGENCE: BUILDING DATASET MANIFESTS")
    print(f"Output Directory: {output_dir.resolve()}")
    print("=" * 80)

    # 1. Pothole
    spec_p = DATASET_SPECS['pothole']
    root_p = resolve_dataset_root(spec_p['owner'], spec_p['name'])
    if root_p:
        print(f"\nBuilding manifest for Pothole dataset ({root_p})...")
        df_p = build_pothole_manifest(root_p)
        df_p.to_csv(output_dir / 'pothole_manifest.csv', index=False)
        manifests['pothole'] = df_p
        print(f"Saved {output_dir / 'pothole_manifest.csv'} ({len(df_p)} records)")
    else:
        print(f"Warning: Root for {spec_p['name']} not found!")

    # 2. Road Crack
    spec_rc = DATASET_SPECS['road_crack']
    root_rc = resolve_dataset_root(spec_rc['owner'], spec_rc['name'])
    if root_rc:
        print(f"\nBuilding manifest for Road-Crack dataset ({root_rc})...")
        df_rc = build_road_crack_manifest(root_rc)
        df_rc.to_csv(output_dir / 'road_crack_manifest.csv', index=False)
        manifests['road_crack'] = df_rc
        print(f"Saved {output_dir / 'road_crack_manifest.csv'} ({len(df_rc)} records)")
    else:
        print(f"Warning: Root for {spec_rc['name']} not found!")

    # 3. CRACK500
    spec_c = DATASET_SPECS['crack500']
    root_c = resolve_dataset_root(spec_c['owner'], spec_c['name'])
    if root_c:
        print(f"\nBuilding manifest for CRACK500 dataset ({root_c})...")
        df_c = build_crack500_manifest(root_c)
        df_c.to_csv(output_dir / 'crack500_manifest.csv', index=False)
        manifests['crack500'] = df_c
        print(f"Saved {output_dir / 'crack500_manifest.csv'} ({len(df_c)} records)")
    else:
        print(f"Warning: Root for {spec_c['name']} not found!")

    # 4. Unified Segmentation Manifest
    seg_dfs = [df for k, df in manifests.items() if k in {'pothole', 'road_crack', 'crack500'}]
    if seg_dfs:
        df_seg = pd.concat(seg_dfs, ignore_index=True)
        df_seg.to_csv(output_dir / 'segmentation_manifest.csv', index=False)
        manifests['segmentation'] = df_seg
        print(f"\nSaved Unified {output_dir / 'segmentation_manifest.csv'} ({len(df_seg)} records)")

    # 5. RDD2022ES (Bounding Box manifest)
    spec_r = DATASET_SPECS['rdd2022es']
    root_r = resolve_dataset_root(spec_r['owner'], spec_r['name'])
    if root_r:
        print(f"\nBuilding manifest for RDD2022ES dataset ({root_r})...")
        df_r = build_rdd2022es_manifest(root_r)
        df_r.to_csv(output_dir / 'rdd2022es_manifest.csv', index=False)
        manifests['rdd2022es'] = df_r
        print(f"Saved {output_dir / 'rdd2022es_manifest.csv'} ({len(df_r)} records)")
    else:
        print(f"Warning: Root for {spec_r['name']} not found!")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("MANIFEST SUMMARY TABLE")
    print("=" * 80)
    summary_rows = []
    for src_key, df in manifests.items():
        if src_key == 'segmentation':
            continue
        for sp in ['train', 'val', 'test']:
            sub = df[df['split'] == sp]
            summary_rows.append({
                'Source': src_key,
                'Split': sp,
                'Images': len(sub),
                'Unique Groups': sub['group_id'].nunique(),
                'Crack Channel': sub['labels_crack'].iloc[0] if len(sub) > 0 else False,
                'Pothole Channel': sub['labels_pothole'].iloc[0] if len(sub) > 0 else False
            })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    return manifests


if __name__ == '__main__':
    generate_all_manifests()
