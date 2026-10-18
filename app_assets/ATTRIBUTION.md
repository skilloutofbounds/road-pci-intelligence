# Dataset Attribution & Licensing for Example Images

This directory contains representative test-split pavement inspection images utilized in the `road-pci-intelligence` interactive dashboard and verification test suite (`tests/test_pipeline.py`).

All images originate strictly from the held-out test splits evaluated in `results/tile_scores.csv`.

---

## 1. CRACK500 Examples (Pavement Cracks on Asphalt)

- **Files**:
  - `crack500_example_01.jpg` (Original: `CRACK500/JPEGImages/images/20160307_163552.jpg`)
    - Split: Test
    - Benchmark sPCI: 70.68
  - `crack500_example_02.jpg` (Original: `CRACK500/JPEGImages/images/20160328_154202.jpg`)
    - Split: Test
    - Benchmark sPCI: 24.60
- **Source**: Yang et al. (2019) and Kaggle uploader `sj26717` (`sj26717/crack500`)
- **Citation**:
  > Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., & Ling, H. (2019). Feature pyramid and hierarchical boosting network for pavement crack detection. *IEEE Transactions on Intelligent Transportation Systems*, 21(4), 1525-1535.
- **License**: Kaggle metadata states `CC0-1.0`.

---

## 2. Road Crack Dataset Examples (Concrete Pavement Cracks)

- **Files**:
  - `road_crack_example_01.jpg` (Original: `road crack dataset semantic/valid/road_1047_6_jpg.rf.aac632486bb3f74cd1d87c56bedc2f68.jpg`)
    - Split: Test
    - Benchmark sPCI: 90.81
  - `road_crack_example_02.jpg` (Original: `road crack dataset semantic/valid/road_1047_1_jpg.rf.08eeaefeb2d3e5e164c42931ae238c7f.jpg`)
    - Split: Test
    - Benchmark sPCI: 40.97
- **Source**: Roboflow Universe dataset `crack-gfyha/road_crack_semantic`
- **Dataset URL**: https://universe.roboflow.com/crack-gfyha/road_crack_semantic
- **License**: `CC BY 4.0`.

---

## 3. Pothole Dataset Audit & Removal Action

- **Previous Files**:
  - `pothole_example_01.jpg`
  - `pothole_example_02.jpg`
- **Source**: `farzadnekouei/pothole-image-segmentation-dataset`
- **License Status**: **unverified**
- **Action Taken**: Permanently removed from `app_assets/examples/` because public redistribution licensing could not be established.
