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
| **Run C (Seed 42)** | `11_train_asymmetric` | Asymmetric | BCE + Dice | 42 | 30 | Ep 29 | **0.7219** | **0.6746** | **0.4174** | **0.7691** | Completed 30 epochs (842.72s / 14.05 min); best epoch 29 (final ep 30 score 0.7195) |

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

## 5. Exhaustive RDD2022ES Ground Truth Statistics (All 25,599 Real Label Files)

Recomputed exhaustively from all 25,599 `.txt` label files in the official RDD2022ES distribution across 16 parallel threads.  
*Audit Note: The earlier Step 2 exploratory numbers could not be reproduced; the verified figures below reflect the exhaustive ground-truth file scan.*

### A. Split-Level Summary & Histogram

| Split | Total Images | Zero-Box Images (%) | Total Boxes | Mean Boxes/Img | Median | Max Boxes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 20,480 | 6,730 (32.86%) | 28,506 | 1.39 | 1.0 | 11 |
| **Val** | 2,560 | 788 (30.78%) | 3,776 | 1.48 | 1.0 | 9 |
| **Test** | 2,560 | 800 (31.25%) | 3,628 | 1.42 | 1.0 | 11 |
| **Overall** | 25,600 | 8,318 (32.49%) | 35,910 | 1.40 | 1.0 | 11 |

*Boxes Per Image Histogram (Overall, N = 25,600):*
- **0 boxes**: 8,318 (32.49%)
- **1 box**: 7,696 (30.06%)
- **2 boxes**: 4,714 (18.41%)
- **3 boxes**: 2,556 (9.98%)
- **4 boxes**: 1,242 (4.85%)
- **5 boxes**: 596 (2.33%)
- **6 boxes**: 294 (1.15%)
- **7 boxes**: 108 (0.42%)
- **8 boxes**: 46 (0.18%)
- **9 boxes**: 22 (0.09%)
- **10 boxes**: 4 (0.02%)
- **11 boxes**: 4 (0.02%)

### B. Class-Level Box Counts & Normalized Area Fractions (Median & 90th Percentile)

Using exact classes from `classes.txt`:

| Class ID | Class Name | Train Boxes (Med / P90 Area) | Val Boxes (Med / P90 Area) | Test Boxes (Med / P90 Area) | Overall Boxes (Med / P90 Area) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `longitudinal` | 10,132 (1.444% / 5.476%) | 1,342 (1.356% / 5.235%) | 1,336 (1.500% / 5.437%) | 12,810 (1.436% / 5.446%) |
| **1** | `longitudinal wide` | 1,852 (2.731% / 9.093%) | 242 (3.265% / 8.667%) | 246 (2.802% / 8.900%) | 2,340 (2.791% / 9.037%) |
| **2** | `transverse` | 5,948 (1.003% / 3.304%) | 788 (1.028% / 3.396%) | 720 (1.012% / 3.132%) | 7,456 (1.005% / 3.308%) |
| **3** | `transverse wide` | 538 (2.409% / 8.495%) | 86 (2.663% / 8.892%) | 84 (2.417% / 5.890%) | 708 (2.448% / 8.490%) |
| **4** | `alligator` | 5,482 (7.785% / 24.527%) | 728 (7.479% / 24.263%) | 708 (7.669% / 27.098%) | 6,918 (7.746% / 24.671%) |
| **5** | `alligator sunken` | 712 (9.807% / 25.953%) | 92 (10.155% / 27.108%) | 102 (12.139% / 30.796%) | 906 (10.046% / 26.766%) |
| **6** | `pothole` | 3,406 (0.373% / 2.023%) | 434 (0.408% / 1.542%) | 384 (0.336% / 1.554%) | 4,224 (0.371% / 1.946%) |
| **7** | `pothole deep` | 436 (1.111% / 6.458%) | 64 (1.317% / 3.609%) | 48 (0.981% / 10.111%) | 548 (1.152% / 6.198%) |

---

## 6. Cross-Domain RDD2022ES Validation Check (Combined Model A, Model B, Model C)

*Evaluation Protocol*: Evaluated across all 2,560 images of `rdd2022es` val split at decision threshold 0.5 with full-image padding to multiples of 32:
- **Model A**: Masked baseline (`best.pt`, Epoch 25)
- **Model B**: Naive baseline (`best.pt`, Epoch 30)
- **Model C**: Asymmetric baseline (Seed 42, Epoch 30 checkpoint `last.pt`)

### A. Images with ZERO Bounding Boxes (N = 788 images)

| Model | Channel | Flag > 0.1% | Flag > 0.5% | Flag > 1.0% | Mean FG Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | Crack | 34.39% (271) | 6.35% (50) | 1.27% (10) | 0.1288% |
| **Model A (Masked)** | Pothole | 69.92% (551) | 56.85% (448) | 51.02% (402) | 5.2213% |
| **Model B (Naive)** | Crack | 5.08% (40) | 0.76% (6) | 0.25% (2) | 0.0205% |
| **Model B (Naive)** | Pothole | 43.53% (343) | 30.58% (241) | 22.21% (175) | 1.4874% |
| **Model C (Asymmetric, Ep30)** | Crack | 81.35% (641) | 46.19% (364) | 27.28% (215) | 0.7414% |
| **Model C (Asymmetric, Ep30)** | Pothole | 47.21% (372) | 35.15% (277) | 27.66% (218) | 2.1666% |

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
| **Model C (Asymmetric, Ep30)** | **All Potholes (6-7)** | 302 | 498 | 21.85% (66/302) | 5.41% |
| Model C (Asymmetric, Ep30)** | Class 6 (pothole) | 282 | 434 | 19.15% (54/282) | 4.79% |
| Model C (Asymmetric, Ep30)** | Class 7 (pothole deep) | 48 | 64 | 33.33% (16/48) | 9.56% |

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
| **Model C (Asymmetric, Ep30)** | **All Cracks (0-5)** | 1650 | 3278 | 16.91% (279/1650) | 1.46% |
| Model C (Asymmetric, Ep30)** | Class 0 (longitudinal) | 932 | 1342 | 11.16% (104/932) | 1.12% |
| Model C (Asymmetric, Ep30)** | Class 1 (longitudinal wide) | 206 | 242 | 20.87% (43/206) | 2.45% |
| Model C (Asymmetric, Ep30)** | Class 2 (transverse) | 540 | 788 | 20.19% (109/540) | 2.02% |
| Model C (Asymmetric, Ep30)** | Class 3 (transverse wide) | 80 | 86 | 38.75% (31/80) | 4.30% |
| Model C (Asymmetric, Ep30)** | Class 4 (alligator) | 584 | 728 | 5.82% (34/584) | 0.96% |
| Model C (Asymmetric, Ep30)** | Class 5 (alligator sunken) | 88 | 92 | 2.27% (2/88) | 0.57% |

### C. Cross-Channel Audit

| Model | Image Subset | Evaluated Channel | Flag > 0.1% | Flag > 0.5% | Mean FG Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | Only Crack Boxes (N = 1,470) | Pothole | 63.33% (931) | 39.12% (575) | 2.6762% |
| **Model B (Naive)** | Only Crack Boxes (N = 1,470) | Pothole | 35.31% (519) | 22.38% (329) | 0.7491% |
| **Model C (Asymmetric, Ep30)** | Only Crack Boxes (N = 1,470) | Pothole | 32.38% (476) | 21.09% (310) | 0.9373% |
| **Model A (Masked)** | Only Pothole Boxes (N = 122) | Crack | 41.80% (51) | 9.02% (11) | 0.1418% |
| **Model B (Naive)** | Only Pothole Boxes (N = 122) | Crack | 3.28% (4) | 0.00% (0) | 0.0135% |
| **Model C (Asymmetric, Ep30)** | Only Pothole Boxes (N = 122) | Crack | 68.03% (83) | 37.70% (46) | 0.6267% |

---

## 7. Multi-Seed Training & Validation Ablation (Seeds 42, 43, 44)

Evaluated across seeds 42, 43, 44 on validation splits at threshold 0.5 over 30 epochs each.

### A. Run-by-Run Best Epoch Selection Table

| Mode | Seed | Best Epoch | Selection Score | CRACK500 Dice | Pothole Dice | Road Crack Dice | Road Crack Non-Empty Dice | Road Crack Empty 0-Pred |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked** | 42 | 25 | 0.7003 | 0.6639 | 0.7368 | 0.3825 | 0.5259 | 3 / 39 (7.7%) |
| **Masked** | 43 | 28 | 0.7106 | 0.6569 | 0.7643 | 0.3937 | 0.5013 | 3 / 39 (7.7%) |
| **Masked** | 44 | 28 | 0.7140 | 0.6610 | 0.7670 | 0.4094 | 0.5391 | 3 / 39 (7.7%) |
| **Naive** | 42 | 30 | 0.7140 | 0.6734 | 0.7546 | 0.4379 | 0.5274 | 11 / 39 (28.2%) |
| **Naive** | 43 | 18 | 0.7156 | 0.6619 | 0.7693 | 0.3989 | 0.5334 | 11 / 39 (28.2%) |
| **Naive** | 44 | 30 | 0.7169 | 0.6731 | 0.7607 | 0.4217 | 0.5281 | 10 / 39 (25.6%) |
| **Asymmetric** | 42 | 29 | 0.7219 | 0.6746 | 0.7691 | 0.4174 | 0.5450 | 6 / 39 (15.4%) |
| **Asymmetric** | 43 | 18 | 0.7053 | 0.6625 | 0.7480 | 0.4342 | 0.5106 | 7 / 39 (17.9%) |
| **Asymmetric** | 44 | 30 | 0.7193 | 0.6707 | 0.7680 | 0.4280 | 0.5371 | 10 / 39 (25.6%) |

### B. Run-by-Run Final Epoch 30 Table

| Mode | Seed | Epoch | Selection Score | CRACK500 Dice | Pothole Dice | Road Crack Dice | Road Crack Non-Empty Dice | Road Crack Empty 0-Pred |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked** | 42 | 30 | 0.6994 | 0.6641 | 0.7348 | 0.3813 | 0.5230 | 3 / 39 (7.7%) |
| **Masked** | 43 | 30 | 0.7055 | 0.6570 | 0.7540 | 0.3997 | 0.5076 | 5 / 39 (12.8%) |
| **Masked** | 44 | 30 | 0.7018 | 0.6605 | 0.7432 | 0.3933 | 0.5366 | 2 / 39 (5.1%) |
| **Naive** | 42 | 30 | 0.7140 | 0.6734 | 0.7546 | 0.4379 | 0.5274 | 11 / 39 (28.2%) |
| **Naive** | 43 | 30 | 0.7111 | 0.6685 | 0.7537 | 0.4475 | 0.5648 | 12 / 39 (30.8%) |
| **Naive** | 44 | 30 | 0.7169 | 0.6731 | 0.7607 | 0.4217 | 0.5281 | 10 / 39 (25.6%) |
| **Asymmetric** | 42 | 30 | 0.7195 | 0.6750 | 0.7639 | 0.4276 | 0.5443 | 8 / 39 (20.5%) |
| **Asymmetric** | 43 | 30 | 0.6933 | 0.6690 | 0.7175 | 0.4210 | 0.5639 | 12 / 39 (30.8%) |
| **Asymmetric** | 44 | 30 | 0.7193 | 0.6707 | 0.7680 | 0.4280 | 0.5371 | 10 / 39 (25.6%) |

### C. Summary Statistics Across 3 Seeds (Sample Std `ddof=1` and Population Std `ddof=0`)

*Note on standard deviations: Sample std is calculated with $N-1$ degrees of freedom (`ddof=1`); Population std is calculated with $N$ degrees of freedom (`ddof=0`).*

| Evaluation Setting | Metric | Masked (Sample Std, `ddof=1`) | Naive (Sample Std, `ddof=1`) | Asymmetric (Sample Std, `ddof=1`) | Masked (Pop Std, `ddof=0`) | Naive (Pop Std, `ddof=0`) | Asymmetric (Pop Std, `ddof=0`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Best Epoch** | Selection Score | 0.7083 ± 0.0058 | 0.7155 ± 0.0012 | 0.7155 ± 0.0089 | 0.7083 ± 0.0048 | 0.7155 ± 0.0010 | 0.7155 ± 0.0073 |
| | CRACK500 Dice | 0.6606 ± 0.0029 | 0.6695 ± 0.0053 | 0.6693 ± 0.0062 | 0.6606 ± 0.0024 | 0.6695 ± 0.0044 | 0.6693 ± 0.0050 |
| | Pothole Dice | 0.7560 ± 0.0136 | 0.7615 ± 0.0060 | 0.7617 ± 0.0119 | 0.7560 ± 0.0111 | 0.7615 ± 0.0049 | 0.7617 ± 0.0097 |
| | Road Crack Dice | 0.3952 ± 0.0110 | 0.4195 ± 0.0160 | 0.4265 ± 0.0085 | 0.3952 ± 0.0090 | 0.4195 ± 0.0131 | 0.4265 ± 0.0069 |
| | Road Crack Non-Empty Dice | 0.5221 ± 0.0156 | 0.5296 ± 0.0027 | 0.5309 ± 0.0180 | 0.5221 ± 0.0127 | 0.5296 ± 0.0022 | 0.5309 ± 0.0147 |
| | RC Empty Clean Count | 3.00 ± 0.00 | 10.67 ± 0.47 | 7.67 ± 2.08 | 3.00 ± 0.00 | 10.67 ± 0.38 | 7.67 ± 1.70 |
| **Final Epoch 30** | Selection Score | 0.7022 ± 0.0030 | 0.7140 ± 0.0029 | 0.7107 ± 0.0151 | 0.7022 ± 0.0025 | 0.7140 ± 0.0024 | 0.7107 ± 0.0123 |
| | CRACK500 Dice | 0.6605 ± 0.0036 | 0.6717 ± 0.0027 | 0.6716 ± 0.0031 | 0.6605 ± 0.0029 | 0.6717 ± 0.0022 | 0.6716 ± 0.0025 |
| | Pothole Dice | 0.7440 ± 0.0096 | 0.7563 ± 0.0038 | 0.7498 ± 0.0281 | 0.7440 ± 0.0079 | 0.7563 ± 0.0031 | 0.7498 ± 0.0229 |
| | Road Crack Dice | 0.3915 ± 0.0093 | 0.4357 ± 0.0130 | 0.4255 ± 0.0039 | 0.3915 ± 0.0076 | 0.4357 ± 0.0106 | 0.4255 ± 0.0032 |
| | Road Crack Non-Empty Dice | 0.5224 ± 0.0145 | 0.5401 ± 0.0214 | 0.5484 ± 0.0139 | 0.5224 ± 0.0119 | 0.5401 ± 0.0174 | 0.5484 ± 0.0113 |
| | RC Empty Clean Count | 3.33 ± 1.53 | 11.00 ± 1.00 | 10.00 ± 2.00 | 3.33 ± 1.25 | 11.00 ± 0.82 | 10.00 ± 1.63 |

---

## 8. Findings

*The following observations summarize exact printed figures from cross-domain RDD validation and multi-seed training runs without speculative attribution:*

### Cross-Domain RDD2022ES Validation Check (Threshold 0.5)
- **Zero-Box Images ($N = 788$)**:
  - Crack channel flag rate (>0.1% foreground): Model A was 34.39% (271 images, mean FG 0.1288%), Model B was 5.08% (40 images, mean FG 0.0205%), Model C was 81.35% (641 images, mean FG 0.7414%).
  - Pothole channel flag rate (>0.1% foreground): Model A was 69.92% (551 images, mean FG 5.2213%), Model B was 43.53% (343 images, mean FG 1.4874%), Model C was 47.21% (372 images, mean FG 2.1666%).
- **Distress Box Hit Rate ($\ge 5\%$ box pixels predicted) & Box Pixel Coverage**:
  - Pothole channel on images with pothole boxes ($N = 302$ images, 498 boxes):
    - All Potholes (6-7): Model A achieved a 23.84% hit rate (mean box pixel fraction 7.66%), Model B achieved 25.50% (mean box pixel fraction 10.10%), Model C achieved 21.85% (mean box pixel fraction 5.41%).
    - Class 6 (`pothole`): Model A 19.50% (mean fraction 6.78%), Model B 21.28% (mean fraction 7.81%), Model C 19.15% (mean fraction 4.79%).
    - Class 7 (`pothole deep`): Model A 43.75% (mean fraction 13.57%), Model B 52.08% (mean fraction 25.65%), Model C 33.33% (mean fraction 9.56%).
  - Crack channel on images with crack boxes ($N = 1,650$ images, 3,278 boxes):
    - All Cracks (0-5): Model A achieved a 6.24% hit rate (mean box pixel fraction 0.56%), Model B achieved 8.73% (mean box pixel fraction 0.72%), Model C achieved 16.91% (mean box pixel fraction 1.46%).
    - Class 0 (`longitudinal`): Model A 5.15% (0.50%), Model B 8.80% (0.81%), Model C 11.16% (1.12%).
    - Class 1 (`longitudinal wide`): Model A 12.14% (1.45%), Model B 16.02% (1.62%), Model C 20.87% (2.45%).
    - Class 2 (`transverse`): Model A 4.26% (0.38%), Model B 5.74% (0.65%), Model C 20.19% (2.02%).
    - Class 3 (`transverse wide`): Model A 12.50% (1.46%), Model B 21.25% (2.15%), Model C 38.75% (4.30%).
    - Class 4 (`alligator`): Model A 2.23% (0.47%), Model B 0.34% (0.23%), Model C 5.82% (0.96%).
    - Class 5 (`alligator sunken`): Model A 2.27% (0.31%), Model B 1.14% (0.15%), Model C 2.27% (0.57%).
- **Cross-Channel Audit**:
  - On images containing only crack boxes ($N = 1,470$):
    - Pothole channel flagged (>0.1%): Model A was 63.33% (931 images, mean FG fraction 2.6762%), Model B was 35.31% (519 images, mean FG fraction 0.7491%), Model C was 32.38% (476 images, mean FG fraction 0.9373%).
  - On images containing only pothole boxes ($N = 122$):
    - Crack channel flagged (>0.1%): Model A was 41.80% (51 images, mean FG fraction 0.1418%), Model B was 3.28% (4 images, mean FG fraction 0.0135%), Model C was 68.03% (83 images, mean FG fraction 0.6267%).

### Multi-Seed Training (Seeds 42, 43, 44)
- **Selection Score**: Across 3 seeds:
  - Best epoch: Masked 0.7083 ± 0.0058 (pop: ± 0.0048), Naive 0.7155 ± 0.0012 (pop: ± 0.0010), Asymmetric 0.7155 ± 0.0089 (pop: ± 0.0073).
  - Final epoch 30: Masked 0.7022 ± 0.0030 (pop: ± 0.0025), Naive 0.7140 ± 0.0029 (pop: ± 0.0024), Asymmetric 0.7107 ± 0.0151 (pop: ± 0.0123).
- **In-Domain Dice Scores**:
  - CRACK500 (best epoch): Masked 0.6606 ± 0.0029, Naive 0.6695 ± 0.0053, Asymmetric 0.6693 ± 0.0062; (final epoch 30): Masked 0.6605 ± 0.0036, Naive 0.6717 ± 0.0027, Asymmetric 0.6716 ± 0.0031.
  - Pothole (best epoch): Masked 0.7560 ± 0.0136, Naive 0.7615 ± 0.0060, Asymmetric 0.7617 ± 0.0119; (final epoch 30): Masked 0.7440 ± 0.0096, Naive 0.7563 ± 0.0038, Asymmetric 0.7498 ± 0.0281.
  - Road Crack overall (best epoch): Masked 0.3952 ± 0.0110, Naive 0.4195 ± 0.0160, Asymmetric 0.4265 ± 0.0085; (final epoch 30): Masked 0.3915 ± 0.0093, Naive 0.4357 ± 0.0130, Asymmetric 0.4255 ± 0.0039.
- **Road Crack Non-Empty vs Clean Counts**:
  - On the 38 non-empty road_crack val images:
    - Best epoch Dice: Masked 0.5221 ± 0.0156, Naive 0.5296 ± 0.0027, Asymmetric 0.5309 ± 0.0180.
    - Final epoch 30 Dice: Masked 0.5224 ± 0.0145, Naive 0.5401 ± 0.0214, Asymmetric 0.5484 ± 0.0139.
  - Out of the 39 clean empty road_crack val images, images receiving exactly 0 predicted pixels at threshold 0.5:
    - Best epoch: Masked 3.00 ± 0.00 (7.7%), Naive 10.67 ± 0.47 (27.4%), Asymmetric 7.67 ± 2.08 (19.7%).
    - Final epoch 30: Masked 3.33 ± 1.53 (8.5%), Naive 11.00 ± 1.00 (28.2%), Asymmetric 10.00 ± 2.00 (25.6%).

---

## 9. Decision: Selection of Frozen Model

**Pre-registered Selection Rule**:
> Model C (Asymmetric Loss) is chosen as the official model only if its mean validation Dice is within 0.01 of Model B (Naive Multi-Task Loss) AND its pothole flag rate on clean RDD roads at >1% foreground is lower than Model B's; otherwise, Model B (Naive Multi-Task Loss) is retained.

**Empirical Result**:
- Selection Metric (Mean Val Dice: CRACK500 + Pothole):
  - Model B (Seed 42 Best / Final Ep 30): 0.7140
  - Model C (Seed 42 Best Ep 29): 0.7219 / (Seed 42 Final Ep 30): 0.7195
  - Multi-seed final epoch means (ddof=1): Model B 0.7140 ± 0.0029 vs Model C 0.7107 ± 0.0151.
  - The mean validation Dice difference satisfies $|B - C| \le 0.01$.
- Clean RDD Roads ($N = 788$ zero-box images) Pothole Flag Rate (>1.0% foreground):
  - Model B (Naive): **22.21%** (175 / 788 images)
  - Model C (Asymmetric): **27.66%** (218 / 788 images)
- Clean RDD Roads ($N = 788$ zero-box images) Pothole Flag Rate (>0.1% foreground):
  - Model B (Naive): **43.53%** (343 / 788 images)
  - Model C (Asymmetric): **47.21%** (372 / 788 images)

**Decision**:
Model C did not achieve a lower pothole false flag rate on clean RDD roads compared to Model B (27.66% vs 22.21% at >1% foreground; 47.21% vs 43.53% at >0.1% foreground). Therefore, according to the pre-registered decision rule, **Model B (Naive Multi-Task Loss, Run B, Seed 42, Epoch 30)** is selected and frozen as the official segmentation model.

*Hypothesis on Negative Supervision*: The hypothesis that treating un-annotated channels as negative supervision suppresses false positive activations across single-source datasets is supported on clean RDD zero-box roads where Model B substantially reduced false positives relative to Model A (43.53% vs 69.92% at >0.1%), but this remains a working hypothesis subject to domain limitations, as non-pavement artifacts in forward-facing dashcam perspectives still induce significant false activations across all model configurations.

