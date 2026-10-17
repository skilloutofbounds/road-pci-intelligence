# ROAD-PCI-INTELLIGENCE: Traceable Project Facts Record

This document records the verified empirical facts, configurations, datasets, model architectures, human alignment statistics, and simulation findings of the `road-pci-intelligence` project. Every statement and metric is directly tied to a traceable source file or Git tag.

---

## 1. Datasets, Annotations, and Splits

| Dataset | Source / Reference | License | Raw Count | Used in Splits | Base-ID Deduplication Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pothole-600** | `farzadnekouei/pothole-image-segmentation-dataset` | CC BY 4.0 | 665 images | 516 images (train: 396, val: 60, test: 60) | Prefix `pic-<N>` before augmentation hash |
| **Road Crack** | `rukiyeaydn/road-crack-dataset` | CC BY 4.0 | 2,238 images | 586 images (train: 450, val: 68, test: 68) | Filename prior to `.rf.` hash |
| **CRACK500** | `sj26717/crack500` (Yang et al., 2016) | Academic / Research | 1,896 crops | 584 crops (train: 440, val: 72, test: 72) | Parent image ID prior to crop coordinates |
| **RDD2022ES** | `juusos/rdd2022es` (Spain subset) | CC BY-SA 4.0 | 25,599 label files | 2,560 validation images | Used strictly for out-of-domain perspective check |

- **Total Combined Dataset**: 1,686 images partitioned into 70% Train (1,286), 15% Val (200), and 15% Test (200).
- **Traceable Source**: [`manifests/segmentation_manifest.csv`](../manifests/segmentation_manifest.csv), [`docs/data_decisions.md`](data_decisions.md).

### Ground-Truth Foreground Pixel Reconciliation
Recomputed directly from all 1,686 masks on disk (`src/data_rules.py`):
- **CRACK500**: $6.30\%$ total foreground pixels (mean per-image: $6.34\%$)
- **Road Crack**: $4.12\%$ total foreground pixels (mean per-image: $4.05\%$)
- **Pothole**: $3.41\%$ total foreground pixels (mean per-image: $3.46\%$)
- **Traceable Source**: [`docs/data_decisions.md`](data_decisions.md) Section 4.

---

## 2. Perception Model & Benchmark Results

### 2.1 Model Specification
- **Architecture**: `segmentation_models_pytorch.Unet` with `resnet34` backbone.
- **Output Channels**: 2 channels (Channel 0: Crack, Channel 1: Pothole).
- **Parameter Count**: 24,436,514 trainable parameters.
- **Frozen Checkpoint**: Model B (Naive Multi-Task Loss, Run B, Seed 42, Epoch 30).
- **SHA-256 Checksum**: `9269c5edb6e5d37b359bd3337fbd83c60901cd2b241f9ef789e9116fee400ef7`.
- **Test-Time Augmentation**: Flip-TTA (mean of original, horizontal flip, vertical flip sigmoid probabilities).
- **Traceable Source**: [`docs/experiments.md`](experiments.md) Section 7, Git commit `4cf2926`.

### 2.2 Validation Performance (Threshold 0.5, Flip-TTA)
- **CRACK500** ($N=72$): Dataset Dice = 0.6632, Mean Image Dice = 0.6281
- **Road Crack** ($N=68$): Dataset Dice = 0.6233, Mean Image Dice = 0.5340
- **Pothole** ($N=60$): Dataset Dice = 0.8145, Mean Image Dice = 0.7742
- **Traceable Source**: [`docs/experiments.md`](experiments.md) Section 7.

### 2.3 Official Test Split Evaluation (Evaluated Once, $N=200$)
- **CRACK500** ($N=72$): Dataset IoU = 0.4923, Dataset Dice = 0.6597, Mean Image Dice = 0.6545
- **Pothole** ($N=60$): Dataset IoU = 0.6805, Dataset Dice = 0.8099, Mean Image Dice = 0.7689
- **Road Crack** ($N=68$): Dataset IoU = 0.4495, Dataset Dice = 0.6202, Mean Image Dice = 0.5319 (Non-Empty Dice: 0.6213 on 55 images; 2 / 13 clean images predicted empty).
- **Traceable Source**: [`docs/experiments.md`](experiments.md) Section 7.

### 2.4 Out-of-Domain RDD2022 Limitation
- **Benchmark**: Evaluated on $N = 2,560$ dashcam perspective images.
- **Limitation**: Clean road false-positive flag rate is 27.50% at >1% area for potholes (0.00% for cracks); deep pothole box hit rate is 35.42%.
- **Finding**: Model is strictly tailored to close-up orthogonal downward-facing pavement imagery (handheld/drone), not forward-facing vehicle dashcam views.
- **Traceable Source**: [`docs/experiments.md`](experiments.md) Section 6.

---

## 3. Simplified Pavement Condition Index (sPCI v1)

- **Git Tag**: `spci-v1-frozen` (Commit `68a30aa532f9c7ae9eb36890915d957656eaa8d7`).
- **Rating Bands**: ASTM D6433 7 standard bands:
  - Good ($86 - 100$)
  - Satisfactory ($71 - 85$)
  - Fair ($56 - 70$)
  - Poor ($41 - 55$)
  - Very Poor ($26 - 40$)
  - Serious ($11 - 25$)
  - Failed ($0 - 10$)
- **Aggregation Formula**: Multiplicative Corrected Deduct Value $\text{CDV} = 100 \cdot (1 - \prod (1 - \text{DV}_i / 100))$, $\text{sPCI} = 100 - \text{CDV}$.
- **Calibrated Parameters**:
  - Crack Low ($T \le 4.394\text{ px}$): $a = 5.0, b = 0.75$
  - Crack Medium ($4.394 < T \le 11.588\text{ px}$): $a = 12.0, b = 0.75$
  - Crack High ($T > 11.588\text{ px}$): $a = 25.0, b = 0.75$
  - Pothole Low ($q \le 0.08$): $a = 100.0, b = 0.60$
  - Pothole Medium ($0.08 < q \le 0.20$): $a = 160.0, b = 0.60$
  - Pothole High ($q > 0.20$): $a = 240.0, b = 0.60$
  - Network factor: $m = 1.0 + 0.5 \cdot \max(0, (n_{\text{idx}} - 1.275)/1.275)$.
  - Noise floor: $\tau = 0.00$.
- **Traceable Source**: [`configs/spci_params.yaml`](../configs/spci_params.yaml), [`docs/pci_design.md`](pci_design.md).

---

## 4. Double-Blind Human Validation Agreement

- **Evaluator**: One human rater (blinded).
- **Dataset**: 60 test image pairs in [`labels/pairs_human.csv`](../labels/pairs_human.csv).
- **Model Test Scores Commit**: Tag `test-scores-computed` (Commit `3a097be85123ca1a3b45ff4fdb47cd8a7bc98ed8`), committed prior to unblinding labels.
- **Pre-Registered Tie Rule**: 13 pairs labeled "same" by the evaluator are excluded from the denominator ($N = 47$). Differences $|\Delta\text{sPCI}| < 1.0$ point are counted as ties (awarded 0.5 points).

### Exact Empirical Results:
- **Overall Agreement**: **$36.0 / 47 = 76.60\%$** [95% Bootstrap CI: 64.89%, 87.23%].
- **Ties**: 2 ties (both in pothole subset, scored 0.5 each).
- **Crack Distress Combined** (`crack500` + `road_crack`): **$30 / 33 = 90.91\%$** [95% CI: 78.79%, 100.00%].
  - `crack500`: **$15 / 17 = 88.24\%$** [95% CI: 70.59%, 100.00%]
  - `road_crack`: **$15 / 16 = 93.75\%$** [95% CI: 81.25%, 100.00%]
- **Pothole Distress**: **$6.0 / 14 = 42.86\%$** [95% CI: 17.86%, 67.86%].
- **Damage Area Baseline Comparison**:
  - Damage Area Baseline Agreement: **$35.0 / 47 = 74.47\%$** [95% CI: 61.70%, 85.11%].
  - Paired Difference (sPCI minus Area Baseline): **$+2.13\%$** [95% CI: $-2.13\%, +7.45\%$].
  - Interpretation: Because the 95% bootstrap confidence interval strictly contains 0, there is **no detectable difference** between sPCI and the damage-area baseline on the overall set.
- **Traceable Source**: [`docs/human_validation.md`](human_validation.md).

---

## 5. Municipal Maintenance Simulation Setup & Findings

- **Network Topology**: 200 synthetic roadway segments ($668,193\text{ m}^2$ total area), lengths $100 - 600\text{ m}$, widths $6 - 14\text{ m}$. Traffic priority weights: Low ($w=1$, 50%), Medium ($w=2$, 35%), High ($w=3$, 15%).
- **Deterioration Model**: Accelerated lognormal deterioration curve ($\text{median} = 3.0\text{ PCI points/year}$).
- **Treatment Costs**: None ($0.0$), Preventive ($1.0$), Rehabilitation ($c_{\text{rehab}} \in \{3, 5, 8\}$), Reconstruction ($12.0$). All costs are illustrative relative benchmark units.
- **Policies Evaluated**:
  - `P0`: Do Nothing
  - `P1`: Reactive (worst-first on $\text{PCI} < 40$)
  - `P2`: Worst-First by ASTM Condition Bands
  - `P2t`: Traffic-Aware Worst-First ($(100 - \text{PCI}) \times w_i$)
  - `P3`: Greedy Benefit/Cost Heuristic
  - `P4`: Exact 0/1 Integer Linear Program (HiGHS branch-and-cut solver)

### 5.1 Base Scenario Performance (Pool 'all', Budget 10%, Ratio 5.0, 200 Runs)
- **P0**: Area-Weighted (AW) 361.64, Traffic-Weighted (TW) 357.01, Spend 0 (0.00%)
- **P1**: AW 642.80, TW 644.01, Spend 3,068,687 (91.85%), TW Efficiency 0.0935
- **P2**: AW **660.50**, TW 661.66, Spend 3,337,775 (99.90%), TW Efficiency 0.0913
- **P2t**: AW 626.94, TW 672.45, Spend 3,336,948 (99.88%), TW Efficiency 0.0945
- **P3**: AW 655.82, TW 730.69, Spend 3,337,095 (99.88%), TW Efficiency **0.112**
- **P4**: AW 656.77, TW **731.54**, Spend 3,339,967 (99.97%), TW Efficiency **0.112**

### 5.2 Grid 2 Sensitivity Summary (Fixed Absolute Budget, 18 Cells x 100 Runs)
- **P2**: 6 nominal wins in Area-Weighted condition, 0 nominal wins in Traffic-Weighted condition.
- **P2t**: 0 nominal wins in AW condition (1 co-win), 1 nominal win in TW condition (`cracks, 5% budget, ratio 3`).
- **P3**: 8 nominal wins in AW condition, 10 nominal wins in TW condition.
- **P4**: 4 nominal wins in AW condition, 7 nominal wins in TW condition.

### 5.3 Allowed Conclusions
1. Policy rankings depend heavily on the evaluation metric: P2 leads Area-Weighted condition under constrained budgets; P3/P4 dominate Traffic-Weighted condition.
2. P2t improves traffic-weighted condition over P2 at the cost of area-weighted condition.
3. P1 under-spending is an inherent definitional constraint: it ignores all roads with $\text{PCI} \ge 40$.
4. Zero-mean inspection noise uniformly degrades decision quality (strictly non-negative regret).
5. Systematic bias creates threshold artifacts (negative regret in P1 and P3 due to forced budget expenditure and preventive eligibility shifts).
6. Costs and rates are illustrative relative benchmarks. No claims are made regarding real municipal tax dollar savings or physical infrastructure life-cycle guarantees.
- **Traceable Source**: [`docs/simulation.md`](simulation.md).
