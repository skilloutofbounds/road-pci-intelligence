# Road PCI Intelligence — Experiment Tracking & Model Ablations

This document tracks all training runs, ablation experiments, hyperparameter sweeps, and validation performance benchmarks for the **Road PCI Intelligence** semantic segmentation model.

---

## 1. Experiment Registry & Benchmark Summary

*Note: Per strict protocol, only values printed by a real Kaggle run are filled in. Unrun or in-progress experiments are marked accordingly.*

| Run ID | Notebook / Script | Loss Mode | Loss Terms | Seed | Epochs | Best Epoch | Mean Val Dice (C500 + Pot) | CRACK500 Val Dice | Road Crack Val Dice | Pothole Val Dice | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 04 (Smoke)** | `04_train_baseline` | Masked | BCE + Dice | 42 | 3 | Ep 3 | 0.5885 | 0.5478 | 0.1274 | 0.6292 | Baseline 3-epoch smoke test completed |
| **Run A** | `05_train_masked` | Masked | BCE + Dice | 42 | 30 | Ep 25 | **0.7003** | **0.6639** | **0.3824** | **0.7368** | Completed 30 epochs (851.05s / 14.18 min); best epoch 25 |
| **Run B** | `06_train_naive` | Naive | BCE + Dice | 42 | 30 | Ep 30 | **0.7140** | **0.6734** | **0.4379** | **0.7546** | Completed 30 epochs (929.89s / 15.50 min); best epoch 30 |

---

## 2. Configuration Details

### Model Specification
- **Architecture**: U-Net (`segmentation-models-pytorch`)
- **Encoder**: ResNet-34 pretrained on ImageNet
- **Output Heads**: 2 independent sigmoid logits (`crack`, `pothole`)
- **Resolution**: 512×512 random crops for training; native resolution (padded to multiple of 32) for validation
- **Batch Size**: 8 (training), 1 (validation)
- **Optimizer**: AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- **Scheduler**: Cosine Annealing (`eta_min=1e-6`)
- **Precision**: Automatic Mixed Precision (AMP)
- **Sampler**: `SourceBalancedSampler` drawing 600 samples/epoch (200 crack500, 200 road_crack, 200 pothole)

### Evaluation Policy
- Selection Metric for `best.pt`: $\frac{\text{Dice}_{\text{crack500}} + \text{Dice}_{\text{pothole}}}{2.0}$
- Validation split only; test split strictly untouched.
- Decision threshold sweep across $\{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ evaluated on `best.pt`.

---

## 3. Threshold Sweep Results (Run A: Masked Baseline, `best.pt` on Val Split)

Evaluated via `notebooks/07_eval_threshold_sweep` loading `best.pt` (Epoch 25).

| Source | Defect Channel | Threshold | Dataset IoU | Dataset Dice | Mean Img Dice | Non-Empty Img Dice (N) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`crack500`** | Crack | 0.3 | 0.4949 | 0.6621 | 0.6245 | 0.6245 (70/70) |
| **`road_crack`** | Crack | 0.3 | 0.2269 | 0.3699 | 0.2818 | 0.5184 (38/77) |
| **`pothole`** | Pothole | 0.3 | 0.5521 | 0.7114 | 0.6593 | 0.6593 (108/108) |
| **`crack500`** | Crack | 0.4 | 0.4965 | 0.6635 | 0.6267 | 0.6267 (70/70) |
| **`road_crack`** | Crack | 0.4 | 0.2322 | 0.3769 | 0.2973 | 0.5234 (38/77) |
| **`pothole`** | Pothole | 0.4 | 0.5703 | 0.7263 | 0.6719 | 0.6719 (108/108) |
| **`crack500`** | Crack | **0.5** | **0.4968** | **0.6639** | **0.6279** | 0.6279 (70/70) |
| **`road_crack`** | Crack | 0.5 | 0.2365 | 0.3825 | 0.2985 | 0.5259 (38/77) |
| **`pothole`** | Pothole | 0.5 | 0.5833 | 0.7368 | 0.6805 | 0.6805 (108/108) |
| **`crack500`** | Crack | 0.6 | 0.4961 | 0.6632 | 0.6281 | 0.6281 (70/70) |
| **`road_crack`** | Crack | 0.6 | 0.2402 | 0.3873 | 0.2988 | 0.5265 (38/77) |
| **`pothole`** | Pothole | 0.6 | 0.5952 | 0.7463 | 0.6871 | 0.6871 (108/108) |
| **`crack500`** | Crack | 0.7 | 0.4938 | 0.6612 | 0.6270 | 0.6270 (70/70) |
| **`road_crack`** | Crack | 0.7 | 0.2445 | 0.3929 | 0.2987 | 0.5263 (38/77) |
| **`pothole`** | Pothole | 0.7 | 0.6041 | 0.7532 | 0.6910 | 0.6910 (108/108) |
| **`crack500`** | Crack | 0.8 | 0.4881 | 0.6560 | 0.6233 | 0.6233 (70/70) |
| **`road_crack`** | Crack | 0.8 | 0.2490 | 0.3987 | 0.2969 | 0.5227 (38/77) |
| **`pothole`** | Pothole | **0.8** | **0.6088** | **0.7568** | **0.6909** | 0.6909 (108/108) |
| **`crack500`** | Crack | 0.9 | 0.4690 | 0.6386 | 0.6079 | 0.6079 (70/70) |
| **`road_crack`** | Crack | **0.9** | **0.2512** | **0.4015** | **0.3131** | **0.5029** (38/77) |
| **`pothole`** | Pothole | 0.9 | 0.5910 | 0.7429 | 0.6637 | 0.6637 (108/108) |

---

## 4. Threshold Sweep Results (Run B: Naive Baseline, `best.pt` on Val Split)

Evaluated via `notebooks/06_train_naive` loading `best.pt` (Epoch 30).

| Source | Defect Channel | Threshold | Dataset IoU | Dataset Dice | Mean Img Dice | Non-Empty Img Dice (N) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`crack500`** | Crack | 0.3 | 0.5085 | 0.6742 | 0.6383 | 0.6383 (70/70) |
| **`road_crack`** | Crack | 0.3 | 0.2705 | 0.4259 | 0.4000 | 0.5210 (38/77) |
| **`pothole`** | Pothole | 0.3 | 0.5912 | 0.7431 | 0.7161 | 0.7161 (108/108) |
| **`crack500`** | Crack | **0.4** | **0.5085** | **0.6742** | **0.6390** | 0.6390 (70/70) |
| **`road_crack`** | Crack | 0.4 | 0.2758 | 0.4324 | 0.4017 | 0.5245 (38/77) |
| **`pothole`** | Pothole | 0.4 | 0.6002 | 0.7502 | 0.7226 | 0.7226 (108/108) |
| **`crack500`** | Crack | 0.5 | 0.5076 | 0.6734 | 0.6387 | 0.6387 (70/70) |
| **`road_crack`** | Crack | 0.5 | 0.2803 | 0.4379 | 0.4031 | 0.5274 (38/77) |
| **`pothole`** | Pothole | 0.5 | 0.6059 | 0.7546 | 0.7268 | 0.7268 (108/108) |
| **`crack500`** | Crack | 0.6 | 0.5058 | 0.6718 | 0.6377 | 0.6377 (70/70) |
| **`road_crack`** | Crack | 0.6 | 0.2841 | 0.4425 | 0.4038 | 0.5287 (38/77) |
| **`pothole`** | Pothole | 0.6 | 0.6086 | 0.7567 | 0.7283 | 0.7283 (108/108) |
| **`crack500`** | Crack | 0.7 | 0.5023 | 0.6687 | 0.6352 | 0.6352 (70/70) |
| **`road_crack`** | Crack | 0.7 | 0.2886 | 0.4480 | 0.4043 | 0.5298 (38/77) |
| **`pothole`** | Pothole | **0.7** | **0.6098** | **0.7576** | **0.7284** | 0.7284 (108/108) |
| **`crack500`** | Crack | 0.8 | 0.4957 | 0.6629 | 0.6302 | 0.6302 (70/70) |
| **`road_crack`** | Crack | 0.8 | 0.2936 | 0.4539 | 0.4164 | 0.5279 (38/77) |
| **`pothole`** | Pothole | 0.8 | 0.6065 | 0.7550 | 0.7242 | 0.7242 (108/108) |
| **`crack500`** | Crack | 0.9 | 0.4763 | 0.6453 | 0.6148 | 0.6148 (70/70) |
| **`road_crack`** | Crack | **0.9** | **0.3034** | **0.4656** | **0.4116** | 0.5183 (38/77) |
| **`pothole`** | Pothole | 0.9 | 0.5805 | 0.7345 | 0.6990 | 0.6990 (108/108) |

---

## 5. Road Crack Validation Non-Empty Mask Audit (Threshold 0.5)

| Metric | Run A (`best.pt`, Ep 25) | Run B (`best.pt`, Ep 30) |
| :--- | :--- | :--- |
| **Total `road_crack` Val Images** | 77 | 77 |
| **Non-Empty Images ($\ge 1$ GT Crack Pixel)** | 38 | 38 |
| **Empty Images (0 GT Crack Pixels)** | 39 | 39 |
| **Mean Img Dice (All 77 Images)** | 0.2985 | 0.4031 |
| **Mean Img Dice (Only 38 Non-Empty Images)** | 0.5259 | 0.5274 |
| **Dataset Dice (Pixel-accumulated)** | 0.3825 | 0.4379 |
| **Dataset IoU (Pixel-accumulated)** | 0.2365 | 0.2803 |

---

## 6. Cross-Domain RDD2022ES Validation Check (Model A vs Model B)

*Note: RDD bounding boxes are loose weak labels, so these are not Dice scores and not comparable to earlier segmentation numbers.*  
Evaluated via `notebooks/08_rdd_cross_domain` across all 2,560 images of `rdd2022es` val split at threshold 0.5.

### A. Images with ZERO Bounding Boxes (N = 788 images)

| Model | Channel | Flag > 0.1% | Flag > 0.5% | Flag > 1.0% | Mean FG Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | Crack | 34.39% (271) | 6.35% (50) | 1.27% (10) | 0.1288% |
| **Model A (Masked)** | Pothole | 69.92% (551) | 56.85% (448) | 51.02% (402) | 5.2213% |
| **Model B (Naive)** | Crack | 5.08% (40) | 0.76% (6) | 0.25% (2) | 0.0205% |
| **Model B (Naive)** | Pothole | 43.53% (343) | 30.58% (241) | 22.21% (175) | 1.4874% |

### B. Hit Rate & Box Pixel Coverage on Images with Ground-Truth Boxes

#### Pothole Channel vs Pothole Boxes (N = 302 images, 498 boxes)
| Model | Category | Images | Boxes | Hit Rate (≥ 5% Box Pixels) | Mean Box Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | **All Potholes (6-7)** | 302 | 498 | 23.84% (72/302) | 7.66% |
| Model A (Masked) | Class 6 (pothole) | 282 | 434 | 19.50% (55/282) | 6.78% |
| Model A (Masked) | Class 7 (pothole deep) | 48 | 64 | 43.75% (21/48) | 13.57% |
| **Model B (Naive)** | **All Potholes (6-7)** | 302 | 498 | 25.50% (77/302) | 10.10% |
| Model B (Naive) | Class 6 (pothole) | 282 | 434 | 21.28% (60/282) | 7.81% |
| Model B (Naive) | Class 7 (pothole deep) | 48 | 64 | 52.08% (25/48) | 25.65% |

#### Crack Channel vs Crack Boxes (N = 1,650 images, 3,278 boxes)
| Model | Category | Images | Boxes | Hit Rate (≥ 5% Box Pixels) | Mean Box Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | **All Cracks (0-5)** | 1650 | 3278 | 6.24% (103/1650) | 0.56% |
| Model A (Masked) | Class 0 (longitudinal) | 932 | 1342 | 5.15% (48/932) | 0.50% |
| Model A (Masked) | Class 1 (longitudinal wide) | 206 | 242 | 12.14% (25/206) | 1.45% |
| Model A (Masked) | Class 2 (transverse) | 540 | 788 | 4.26% (23/540) | 0.38% |
| Model A (Masked) | Class 3 (transverse wide) | 80 | 86 | 12.50% (10/80) | 1.46% |
| Model A (Masked) | Class 4 (alligator) | 584 | 728 | 2.23% (13/584) | 0.47% |
| Model A (Masked) | Class 5 (alligator sunken) | 88 | 92 | 2.27% (2/88) | 0.31% |
| **Model B (Naive)** | **All Cracks (0-5)** | 1650 | 3278 | 8.73% (144/1650) | 0.72% |
| Model B (Naive) | Class 0 (longitudinal) | 932 | 1342 | 8.80% (82/932) | 0.81% |
| Model B (Naive) | Class 1 (longitudinal wide) | 206 | 242 | 16.02% (33/206) | 1.62% |
| Model B (Naive) | Class 2 (transverse) | 540 | 788 | 5.74% (31/540) | 0.65% |
| Model B (Naive) | Class 3 (transverse wide) | 80 | 86 | 21.25% (17/80) | 2.15% |
| Model B (Naive) | Class 4 (alligator) | 584 | 728 | 0.34% (2/584) | 0.23% |
| Model B (Naive) | Class 5 (alligator sunken) | 88 | 92 | 1.14% (1/88) | 0.15% |

### C. Cross-Channel Audit

| Model | Image Subset | Evaluated Channel | Flag > 0.1% | Flag > 0.5% | Mean FG Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | Only Crack Boxes (N = 1,470) | Pothole | 63.33% (931) | 39.12% (575) | 2.6762% |
| **Model B (Naive)** | Only Crack Boxes (N = 1,470) | Pothole | 35.31% (519) | 22.38% (329) | 0.7491% |
| **Model A (Masked)** | Only Pothole Boxes (N = 122) | Crack | 41.80% (51) | 9.02% (11) | 0.1418% |
| **Model B (Naive)** | Only Pothole Boxes (N = 122) | Crack | 3.28% (4) | 0.00% (0) | 0.0135% |

---

## 7. Multi-Seed Training & Validation Ablation (Seeds 42, 43, 44)

Evaluated via `notebooks/09_seed_runs` on validation splits at threshold 0.5 over 30 epochs each.

### A. Run-by-Run Best Epoch Selection Table

| Mode | Seed | Best Epoch | Selection Score | CRACK500 Dice | Pothole Dice | Road Crack Dice | Road Crack Non-Empty Dice | Road Crack Empty 0-Pred |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked** | 42 | 25 | 0.7003 | 0.6639 | 0.7368 | 0.3825 | 0.5259 | 3 / 39 (7.7%) |
| **Masked** | 43 | 28 | 0.7106 | 0.6569 | 0.7643 | 0.3937 | 0.5013 | 3 / 39 (7.7%) |
| **Masked** | 44 | 28 | 0.7140 | 0.6610 | 0.7670 | 0.4094 | 0.5391 | 3 / 39 (7.7%) |
| **Naive** | 42 | 30 | 0.7140 | 0.6734 | 0.7546 | 0.4379 | 0.5274 | 11 / 39 (28.2%) |
| **Naive** | 43 | 18 | 0.7156 | 0.6619 | 0.7693 | 0.3989 | 0.5334 | 11 / 39 (28.2%) |
| **Naive** | 44 | 30 | 0.7169 | 0.6731 | 0.7607 | 0.4217 | 0.5281 | 10 / 39 (25.6%) |

### B. Run-by-Run Final Epoch 30 Table

| Mode | Seed | Epoch | Selection Score | CRACK500 Dice | Pothole Dice | Road Crack Dice | Road Crack Non-Empty Dice | Road Crack Empty 0-Pred |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked** | 42 | 30 | 0.6994 | 0.6641 | 0.7348 | 0.3813 | 0.5230 | 3 / 39 (7.7%) |
| **Masked** | 43 | 30 | 0.7055 | 0.6570 | 0.7540 | 0.3997 | 0.5076 | 5 / 39 (12.8%) |
| **Masked** | 44 | 30 | 0.7018 | 0.6605 | 0.7432 | 0.3933 | 0.5366 | 2 / 39 (5.1%) |
| **Naive** | 42 | 30 | 0.7140 | 0.6734 | 0.7546 | 0.4379 | 0.5274 | 11 / 39 (28.2%) |
| **Naive** | 43 | 30 | 0.7111 | 0.6685 | 0.7537 | 0.4475 | 0.5648 | 12 / 39 (30.8%) |
| **Naive** | 44 | 30 | 0.7169 | 0.6731 | 0.7607 | 0.4217 | 0.5281 | 10 / 39 (25.6%) |

### C. Summary Statistics Across 3 Seeds (Mean ± Std)

| Evaluation Setting | Metric | Masked (Mean ± Std) | Naive (Mean ± Std) |
| :--- | :--- | :--- | :--- |
| **Best Epoch** | Selection Score | 0.7083 ± 0.0058 | 0.7155 ± 0.0012 |
| | CRACK500 Dice | 0.6606 ± 0.0029 | 0.6695 ± 0.0053 |
| | Pothole Dice | 0.7560 ± 0.0136 | 0.7615 ± 0.0060 |
| | Road Crack Dice | 0.3952 ± 0.0110 | 0.4195 ± 0.0160 |
| | Road Crack Non-Empty Dice | 0.5221 ± 0.0156 | 0.5296 ± 0.0027 |
| | Road Crack Empty 0-Pred Count | 3.00 ± 0.00 | 10.67 ± 0.47 |
| **Final Epoch 30** | Selection Score | 0.7022 ± 0.0025 | 0.7140 ± 0.0024 |
| | CRACK500 Dice | 0.6605 ± 0.0029 | 0.6717 ± 0.0022 |
| | Pothole Dice | 0.7440 ± 0.0079 | 0.7563 ± 0.0031 |
| | Road Crack Dice | 0.3915 ± 0.0076 | 0.4357 ± 0.0106 |
| | Road Crack Non-Empty Dice | 0.5224 ± 0.0119 | 0.5401 ± 0.0174 |
| | Road Crack Empty 0-Pred Count | 3.33 ± 1.25 | 11.00 ± 0.82 |

---

## 8. Findings

*The following observations summarize exact printed figures from cross-domain RDD validation and multi-seed training runs without speculative attribution:*

### Cross-Domain RDD2022ES Validation Check (Threshold 0.5)
- **Zero-Box Images ($N = 788$)**:
  - Crack channel flag rate (>0.1% foreground): Model A was 34.39% (271 images, mean FG fraction 0.1288%) vs Model B at 5.08% (40 images, mean FG fraction 0.0205%).
  - Pothole channel flag rate (>0.1% foreground): Model A was 69.92% (551 images, mean FG fraction 5.2213%) vs Model B at 43.53% (343 images, mean FG fraction 1.4874%).
- **Distress Box Hit Rate ($\ge 5\%$ box pixels predicted) & Box Pixel Coverage**:
  - Pothole channel on images with pothole boxes ($N = 302$ images, 498 boxes): Model A achieved a 23.84% hit rate (mean box pixel fraction 7.66%) vs Model B at 25.50% (mean box pixel fraction 10.10%).
  - Crack channel on images with crack boxes ($N = 1,650$ images, 3,278 boxes): Model A achieved a 6.24% hit rate (mean box pixel fraction 0.56%) vs Model B at 8.73% (mean box pixel fraction 0.72%).
- **Cross-Channel Audit**:
  - On images containing only crack boxes ($N = 1,470$): Model A flagged pothole foreground (>0.1%) on 63.33% of images (mean FG fraction 2.6762%) vs Model B at 35.31% (mean FG fraction 0.7491%).
  - On images containing only pothole boxes ($N = 122$): Model A flagged crack foreground (>0.1%) on 41.80% of images (mean FG fraction 0.1418%) vs Model B at 3.28% (mean FG fraction 0.0135%).

### Multi-Seed Training (Seeds 42, 43, 44)
- **Selection Score**: Across 3 seeds, the best-epoch selection score was 0.7083 ± 0.0058 for Masked vs 0.7155 ± 0.0012 for Naive. At final epoch 30, it was 0.7022 ± 0.0025 for Masked vs 0.7140 ± 0.0024 for Naive.
- **In-Domain Dice Scores**:
  - CRACK500 (best epoch): Masked 0.6606 ± 0.0029 vs Naive 0.6695 ± 0.0053; (final epoch 30): Masked 0.6605 ± 0.0029 vs Naive 0.6717 ± 0.0022.
  - Pothole (best epoch): Masked 0.7560 ± 0.0136 vs Naive 0.7615 ± 0.0060; (final epoch 30): Masked 0.7440 ± 0.0079 vs Naive 0.7563 ± 0.0031.
  - Road Crack overall (best epoch): Masked 0.3952 ± 0.0110 vs Naive 0.4195 ± 0.0160; (final epoch 30): Masked 0.3915 ± 0.0076 vs Naive 0.4357 ± 0.0106.
- **Road Crack Non-Empty vs Clean Counts**:
  - On the 38 non-empty road_crack val images, best-epoch Dice was 0.5221 ± 0.0156 for Masked vs 0.5296 ± 0.0027 for Naive; final epoch 30 Dice was 0.5224 ± 0.0119 for Masked vs 0.5401 ± 0.0174 for Naive.
  - Out of the 39 empty road_crack val images, the number of images receiving exactly 0 predicted pixels at threshold 0.5 was:
    - Best epoch: Masked 3.00 ± 0.00 (7.7%) vs Naive 10.67 ± 0.47 (27.4%).
    - Final epoch 30: Masked 3.33 ± 1.25 (8.5%) vs Naive 11.00 ± 0.82 (28.2%).

