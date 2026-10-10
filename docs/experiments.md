# Road PCI Intelligence — Experiment Tracking & Model Ablations

This document tracks all training runs, ablation experiments, hyperparameter sweeps, and validation performance benchmarks for the **Road PCI Intelligence** semantic segmentation model.

---

## 1. Experiment Registry & Benchmark Summary

*Note: Per strict protocol, only values printed by a real Kaggle run are filled in. Unrun or in-progress experiments are marked accordingly.*

| Run ID | Notebook / Script | Loss Mode | Loss Terms | Seed | Epochs | Best Epoch | Mean Val Dice (C500 + Pot) | CRACK500 Val Dice | Road Crack Val Dice | Pothole Val Dice | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 04 (Smoke)** | `04_train_baseline` | Masked | BCE + Dice | 42 | 3 | Ep 3 | 0.5885 | 0.5478 | 0.1274 | 0.6292 | Baseline 3-epoch smoke test completed |
| **Run A** | `05_train_masked` | Masked | BCE + Dice | 42 | 30 | Ep 25 | **0.7003** | **0.6639** | **0.3824** | **0.7368** | Completed 30 epochs (851.05s / 14.18 min); best epoch 25 |
| **Run B** | `06_train_naive` | Naive | BCE + Dice | 42 | 30 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | 30-epoch naive ablation (awaiting user go-ahead) |

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
- Decision threshold sweep across $\{0.3, 0.4, 0.5, 0.6, 0.7\}$ evaluated on `best.pt`.

---

## 3. Threshold Sweep Results (Run A: Masked Baseline, `best.pt` on Val Split)

| Source | Defect Channel | Threshold | Dataset IoU | Dataset Dice | Mean Img Dice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`crack500`** | Crack | 0.3 | 0.4949 | 0.6621 | 0.6245 |
| **`road_crack`** | Crack | 0.3 | 0.2269 | 0.3699 | 0.2818 |
| **`pothole`** | Pothole | 0.3 | 0.5521 | 0.7114 | 0.6593 |
| **`crack500`** | Crack | 0.4 | 0.4964 | 0.6635 | 0.6267 |
| **`road_crack`** | Crack | 0.4 | 0.2322 | 0.3769 | 0.2973 |
| **`pothole`** | Pothole | 0.4 | 0.5702 | 0.7263 | 0.6719 |
| **`crack500`** | Crack | **0.5** | **0.4969** | **0.6639** | **0.6279** |
| **`road_crack`** | Crack | 0.5 | 0.2364 | 0.3824 | 0.2985 |
| **`pothole`** | Pothole | 0.5 | 0.5833 | 0.7368 | 0.6805 |
| **`crack500`** | Crack | 0.6 | 0.4961 | 0.6632 | 0.6281 |
| **`road_crack`** | Crack | 0.6 | 0.2402 | 0.3873 | 0.2988 |
| **`pothole`** | Pothole | 0.6 | 0.5953 | 0.7463 | 0.6871 |
| **`crack500`** | Crack | 0.7 | 0.4938 | 0.6612 | 0.6270 |
| **`road_crack`** | Crack | **0.7** | **0.2444** | **0.3929** | **0.2987** |
| **`pothole`** | Pothole | **0.7** | **0.6041** | **0.7532** | **0.6911** |
