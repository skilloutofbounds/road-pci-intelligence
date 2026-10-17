# Dataset Attribution & Licensing for Example Images

This directory contains 6 representative test-split pavement inspection images utilized in the `road-pci-intelligence` interactive dashboard and verification test suite (`tests/test_pipeline.py`).

All images originate strictly from the held-out test splits evaluated in `results/tile_scores.csv`.

---

## 1. CRACK500 Examples

- **Files**:
  - `crack500_example_01.jpg` (Original: `CRACK500/JPEGImages/images/20160307_163552.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 70.68 (Satisfactory)
  - `crack500_example_02.jpg` (Original: `CRACK500/JPEGImages/images/20160328_154202.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 24.60 (Serious)
- **Source**: Yang et al., 2016 / Kaggle distribution by `sj26717/crack500`
- **Citation**:
  > Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., & Ling, H. (2019). Feature pyramid and hierarchical boosting network for pavement crack detection. *IEEE Transactions on Intelligent Transportation Systems*, 21(4), 1525-1535.
- **License**: Academic / Research Use (CC0-1.0 noted in Kaggle distribution metadata).

---

## 2. Road Crack Dataset Examples

- **Files**:
  - `road_crack_example_01.jpg` (Original: `road crack dataset semantic/valid/road_1047_6_jpg.rf.aac632486bb3f74cd1d87c56bedc2f68.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 90.81 (Good)
  - `road_crack_example_02.jpg` (Original: `road crack dataset semantic/valid/road_1047_1_jpg.rf.08eeaefeb2d3e5e164c42931ae238c7f.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 40.97 (Poor)
- **Source**: `rukiyeaydn/road-crack-dataset` (Roboflow Universe export)
- **License**: CC BY 4.0 / CC0 Public Domain.

---

## 3. Pothole Segmentation Dataset Examples

- **Files**:
  - `pothole_example_01.jpg` (Original: `Pothole_Segmentation_YOLOv8/valid/images/pic-116-_jpg.rf.ddcb9d0e0bcf2a1a5096c4f04e6b7f9e.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 43.65 (Poor)
  - `pothole_example_02.jpg` (Original: `Pothole_Segmentation_YOLOv8/valid/images/pic-140-_jpg.rf.430bfe74decf2b904377d127914f2092.jpg`)
    - Split: Test
    - Target Benchmark sPCI: 79.08 (Satisfactory)
- **Source**: `farzadnekouei/pothole-image-segmentation-dataset` (Roboflow Universe export)
- **License**: CC BY 4.0 / CC0 Public Domain.
