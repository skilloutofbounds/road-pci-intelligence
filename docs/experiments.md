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
| Model A (Masked) | Class 6 | 282 | 434 | 19.50% (55/282) | 6.78% |
| Model A (Masked) | Class 7 | 48 | 64 | 43.75% (21/48) | 13.57% |
| **Model B (Naive)** | **All Potholes (6-7)** | 302 | 498 | 25.50% (77/302) | 10.10% |
| Model B (Naive) | Class 6 | 282 | 434 | 21.28% (60/282) | 7.81% |
| Model B (Naive) | Class 7 | 48 | 64 | 52.08% (25/48) | 25.65% |

#### Crack Channel vs Crack Boxes (N = 1,650 images, 3,278 boxes)
| Model | Category | Images | Boxes | Hit Rate (≥ 5% Box Pixels) | Mean Box Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | **All Cracks (0-5)** | 1650 | 3278 | 6.24% (103/1650) | 0.56% |
| Model A (Masked) | Class 0 (Longitudinal) | 932 | 1342 | 5.15% (48/932) | 0.50% |
| Model A (Masked) | Class 1 (Transverse) | 206 | 242 | 12.14% (25/206) | 1.45% |
| Model A (Masked) | Class 2 (Alligator) | 540 | 788 | 4.26% (23/540) | 0.38% |
| Model A (Masked) | Class 3 | 80 | 86 | 12.50% (10/80) | 1.46% |
| Model A (Masked) | Class 4 | 584 | 728 | 2.23% (13/584) | 0.47% |
| Model A (Masked) | Class 5 | 88 | 92 | 2.27% (2/88) | 0.31% |
| **Model B (Naive)** | **All Cracks (0-5)** | 1650 | 3278 | 8.73% (144/1650) | 0.72% |
| Model B (Naive) | Class 0 (Longitudinal) | 932 | 1342 | 8.80% (82/932) | 0.81% |
| Model B (Naive) | Class 1 (Transverse) | 206 | 242 | 16.02% (33/206) | 1.62% |
| Model B (Naive) | Class 2 (Alligator) | 540 | 788 | 5.74% (31/540) | 0.65% |
| Model B (Naive) | Class 3 | 80 | 86 | 21.25% (17/80) | 2.15% |
| Model B (Naive) | Class 4 | 584 | 728 | 0.34% (2/584) | 0.23% |
| Model B (Naive) | Class 5 | 88 | 92 | 1.14% (1/88) | 0.15% |

### C. Cross-Channel Audit

| Model | Image Subset | Evaluated Channel | Flag > 0.1% | Flag > 0.5% | Mean FG Fraction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Masked)** | Only Crack Boxes (N = 1,470) | Pothole | 63.33% (931) | 39.12% (575) | 2.6762% |
| **Model B (Naive)** | Only Crack Boxes (N = 1,470) | Pothole | 35.31% (519) | 22.38% (329) | 0.7491% |
| **Model A (Masked)** | Only Pothole Boxes (N = 122) | Crack | 41.80% (51) | 9.02% (11) | 0.1418% |
| **Model B (Naive)** | Only Pothole Boxes (N = 122) | Crack | 3.28% (4) | 0.00% (0) | 0.0135% |
