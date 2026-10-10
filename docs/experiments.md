# Road PCI Intelligence — Experiment Tracking & Model Ablations

This document tracks all training runs, ablation experiments, hyperparameter sweeps, and validation performance benchmarks for the **Road PCI Intelligence** semantic segmentation model.

---

## 1. Experiment Registry & Benchmark Summary

*Note: Per strict protocol, only values printed by a real Kaggle run are filled in. Unrun or in-progress experiments are marked accordingly.*

| Run ID | Notebook / Script | Loss Mode | Loss Terms | Seed | Epochs | Best Epoch | Mean Val Dice (C500 + Pot) | CRACK500 Val Dice | Road Crack Val Dice | Pothole Val Dice | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 04 (Smoke)** | `04_train_baseline` | Masked | BCE + Dice | 42 | 3 | Ep 3 | 0.5885 | 0.5478 | 0.1274 | 0.6292 | Baseline 3-epoch smoke test completed |
| **Run A** | `05_train_masked` | Masked | BCE + Dice | 42 | 30 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | 30-epoch full masked baseline |
| **Run B** | `06_train_naive` | Naive | BCE + Dice | 42 | 30 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | 30-epoch naive ablation (unlabeled as all-0s) |

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
