# ROAD-PCI-INTELLIGENCE: Master Lifecycle & System Technical Handover Record

> **Document Type**: Comprehensive Engineering Lifecycle, Architectural Specification, and Technical Handover  
> **Repository**: `road-pci-intelligence`  
> **Status**: Frozen & Synchronized with `docs/PROJECT_FACTS.md` (Commit `spci-v1-frozen` = `68a30aa`)  
> **Target Audience**: Any incoming AI Agent, Machine Learning Engineer, or Infrastructure Asset Specialist auditing or extending this codebase.

---

## 1. Executive Summary & Core Architectural Principles

**`road-pci-intelligence`** is an end-to-end cyber-physical pavement intelligence and decision-support framework. It automates roadway distress detection from visual inspection imagery, derives standardized ASTM D6433 morphological features, calculates an objective simplified Pavement Condition Index (**sPCI v1**), and executes budget-constrained 10-year Monte Carlo maintenance allocation simulations.

$$\text{Pavement Imagery} \xrightarrow{\text{Perception}} \text{Binary Distress Mask} \xrightarrow{\text{Morphology}} \text{ASTM D6433 Features} \xrightarrow{\text{sPCI Engine}} \text{sPCI Rating} \xrightarrow{\text{Optimization}} \text{10-Year Lifecycle Allocation}$$

### Foundational Principles
1. **Strict Separation of Perception & Index Logic**: The deep segmentation model does not output a subjective PCI number directly. It predicts a spatial distress mask; ASTM D6433 morphological logic extracts physical geometry (lengths, widths, counts, areas); and the deterministic, frozen `sPCI v1` engine computes the index.
2. **Quarantined Evaluation**: Test splits were quarantined and evaluated once. All hyperparameter tuning and calibration were conducted exclusively on `train` and `val` partitions.
3. **Double-Blind Human Validation**: Human pair judgments were collected without model knowledge, and model test scores were frozen under Git tag `test-scores-computed` before unblinding.
4. **Deterministic Reproducibility**: Random seeds are fixed (seed 42), and weights are validated with SHA-256 (`4f631d8c11ceabafabfe33bb46ebcda460163c467a99f1ab8da3725b80459da1`).
5. **Stylized Simulation Disclaimer**: Maintenance unit costs ($1, 3, 5, 8, 12$ relative units) and network budgets ($0.50/\text{m}^2/\text{year}$) are stylized algorithmic benchmarks for policy comparison, not municipal cost guarantees.

---

## 2. Ingestion & Dataset Manifest Facts

All training, validation, and test data are cataloged in `manifests/segmentation_manifest.csv` with leak-proof, group-aware partitioning:

| Dataset Source | License / Terms | Total Images | Train | Val | Test | Group Partitioning Rule |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`pothole`** (`farzadnekouei`) | CC0 Public Domain / CC BY 4.0 | 780 | 612 | 108 | 60 | Prefix `pic-<N>` stripped of augmentation hash (300 groups) |
| **`road_crack`** (`rukiyeaydn`) | CC0 Public Domain / CC BY 4.0 | 435 | 290 | 77 | 68 | Physical scene stem `road_<N>` or `Production_1` (22 scenes) |
| **`crack500`** (`sj26717` / Yang et al.) | Academic / Research Use | 471 | 329 | 70 | 72 | Base capture stem (e.g., `20160222_165402`) |
| **Combined Segmentation Total** | **Multi-Benchmark** | **1,686** | **1,231** | **255** | **200** | **Leak-proof group partitioning** |
| **`rdd2022es`** (`juusos`) | Metadata: `CC-BY-NC-SA-4.0`<br>Description: `CC BY-SA 4.0`<br>*(Treated strictly as non-commercial)* | 25,600 | 20,480 | 2,560 | 2,560 | Stem with `xmirror_` stripped across 4 countries (USA, India, Japan, Czech Republic; 8 damage classes) |

### Reconciled Foreground Fractions (Pixel-Weighted)
Recomputed directly from binary mask reconciliation (Notebook 02, recorded in `docs/data_decisions.md` Section 5.3):
- **`crack500`**: **2.65%** pixel-weighted (mean per image: **2.73%**).
- **`road_crack`**: **1.75%** pixel-weighted for mask value 1 only (3,126,349 crack px / 178,176,000 total px; mean per image: **1.75%**).
  - Raw mask value distribution: Value 0 (Background) = 96.19%, Value 1 (Crack) = **1.75%**, Value 2 (Construction Joints) = **2.03%**, Value 3 (Core Drilling) = **0.02%**.
- **`pothole`**: **15.98%** pixel-weighted (mean per image: **15.98%**).

---

## 3. Deep Perception Model Specification

- **Architecture**: `segmentation_models_pytorch.Unet` with ImageNet-pretrained `resnet34` backbone.
- **Model Checkpoint**: ResNet34-UNet generating distress segmentation mask.
- **Frozen Weights Checkpoint**: `outputs/models/best.pt` (and `scratch/downloaded_final_weights/best.pt`).
- **Cryptographic SHA-256**:
  `4f631d8c11ceabafabfe33bb46ebcda460163c467a99f1ab8da3725b80459da1`
- **Inference Resolution / Upload Rule**:
  - EXIF orientation auto-corrected; image converted to RGB.
  - If $\min(H, W) > 720$, downscale preserving aspect ratio such that $\min(H, W) = 720$, with the long side capped at $1280$.
  - Otherwise, maintain native resolution (never upscale). Minimum allowable dimension is $256\times 256$.
- **Validation Dice Performance (Val Split, Threshold 0.5, Flip-TTA)**:
  - `crack500`: Dataset Dice = 0.6632, Mean Image Dice = 0.6281
  - `road_crack`: Dataset Dice = 0.6233, Mean Image Dice = 0.5340
  - `pothole`: Dataset Dice = 0.8145, Mean Image Dice = 0.7742
- **Held-Out Test Split Performance ($N=200$)**:
  - `crack500` ($N=72$): Dataset IoU = 0.4923, Dataset Dice = 0.6597, Mean Image Dice = 0.6545
  - `pothole` ($N=60$): Dataset IoU = 0.6805, Dataset Dice = 0.8099, Mean Image Dice = 0.7689
  - `road_crack` ($N=68$): Dataset IoU = 0.4495, Dataset Dice = 0.6202, Mean Image Dice = 0.5319

---

## 4. ASTM D6433 Morphological Feature Extraction

Binary predicted distress masks are classified into 4 distress types via morphological analysis (`src/pavement_features.py`):
1. **Potholes**: Connected components with circularity/compactness criteria and area thresholds.
2. **Alligator Cracking**: Interconnected polygonal closed cells and high junction node density.
3. **Transverse Cracking**: Predominantly horizontal linear distress skeletons.
4. **Longitudinal Cracking**: Predominantly vertical linear distress skeletons.

Extracted features include: skeleton length ($L_{\text{px}}$), mean width ($w_{\text{px}}$), junction node counts ($J$), enclosed cell counts ($C$), pothole count ($N_{\text{pot}}$), and defect density ($d_k$).

---

## 5. Frozen sPCI v1 Rating Engine

Permanently frozen under Git tag **`spci-v1-frozen`** (pointing to commit **`68a30aa532f9c7ae9eb36890915d957656eaa8d7`**):
- **Canonical Configuration**: `configs/spci_params.yaml`
- **Distress Weights ($w_k$)**:
  - Alligator cracking: `0.35`
  - Pothole: `0.30`
  - Transverse cracking: `0.20`
  - Longitudinal cracking: `0.15`
  *(Sum of weights $\sum w_k = 1.00$)*
- **Log Density Scale ($\beta$)**: `5.0`
- **Severity Exponents**:
  - Low severity ($s_{\text{low}}$): `0.5`
  - Medium severity ($s_{\text{med}}$): `1.0`
  - High severity ($s_{\text{high}}$): `1.5`
- **Empirical Thresholds**:
  - Crack severity: Low $\le 4.394\text{ px}$, High $> 11.588\text{ px}$
  - Network factor: $n_{\text{ref}} = 1.275$, $\beta = 0.5$
  - Noise floor: $\tau = 0.00$
- **Aggregation Formula**: Multiplicative Corrected Deduct Value:
  $$\text{CDV} = 100 \cdot \left(1 - \prod_{k=1}^4 (1 - \text{DV}_k / 100)\right), \quad \text{sPCI} = 100 - \text{CDV}$$
- **ASTM D6433 Condition Bands**:
  Good ($86-100$), Satisfactory ($71-85$), Fair ($56-70$), Poor ($41-55$), Very Poor ($26-40$), Serious ($11-25$), Failed ($0-10$).

---

## 6. Double-Blind Human Validation Alignment

Validation results against blinded human pair evaluations on the held-out test split (`docs/human_validation.md`, `results/rdd_validation_features.csv`):
- **Total Pairs**: 60 pairs; 13 pairs labeled "same" excluded from denominator ($N = 47$).
- **Overall Agreement**: **$36.0 / 47 = 76.60\%$** [95% Bootstrap CI: 64.89%, 87.23%].
- **Mean Absolute Error (MAE)**: 7.91 points.
- **Ties ($|\Delta\text{sPCI}| < 1.0$)**: 2 (both in pothole subset, awarded 0.5 points each).
- **Crack Combined** (`crack500` + `road_crack`): **$30 / 33 = 90.91\%$** [95% CI: 78.79%, 100.00%].
  - `crack500`: **$15 / 17 = 88.24\%$**
  - `road_crack`: **$15 / 16 = 93.75\%$**
- **Pothole Subset**: **$6.0 / 14 = 42.86\%$** [95% CI: 17.86%, 67.86%].
- **Damage Area Baseline Comparison**:
  - Damage Area Baseline Agreement: **$35.0 / 47 = 74.47\%$** [95% CI: 61.70%, 85.11%].
  - Paired Difference (sPCI minus Area): **$+2.13\%$** [95% CI: $-2.13\%, +7.45\%$].
  - Scientific Finding: Because the 95% bootstrap confidence interval contains 0, there is **no detectable difference** between sPCI and the naive damage area baseline on the overall set.

---

## 7. 10-Year Municipal Maintenance Simulation

The simulation framework (`src/simulation/engine.py`, `docs/simulation.md`) models a municipal network of 200 synthetic road segments across two pools:
- **`all` pool**: 200 segments ($668,193\text{ m}^2$), 10-year budget $B_{10} = \$3,340,965$ ($\$334,096.50/\text{year}$).
- **`cracks` pool**: 200 segments ($702,355\text{ m}^2$), 10-year budget $B_{10} = \$3,511,774$ ($\$351,177.39/\text{year}$).

### 7.1 Base Scenario Results (`all` pool, 10% budget, 200 Monte Carlo runs)

| Policy | Logic | Area-Weighted Condition (AW CY) | Traffic-Weighted Condition (TW CY) | Total 10-Yr Spend ($) | Budget Spent (%) | TW Efficiency (CY/$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | No maintenance | 361.64 | 357.01 | $0 | 0.00% | 0.000 |
| **P1: Reactive** | Worst-first on $\text{PCI} < 40$ | 642.80 | 644.01 | $3,068,687 | 91.85% | 0.0935 |
| **P2: Worst-First** | Worst-first by ASTM bands | **660.50** | 661.66 | $3,337,775 | 99.90% | 0.0913 |
| **P2t: Traffic-Aware** | Worst-first by $(100-\text{PCI})\cdot w_i$ | 626.94 | 672.45 | $3,336,948 | 99.88% | 0.0945 |
| **P3: Greedy Benefit** | Maximize $\Delta\text{PCI}\cdot w_i / c_i$ | 655.82 | 730.69 | $3,337,095 | 99.88% | **0.1120** |
| **P4: Integer Linear Program** | Exact 0/1 IP (HiGHS solver) | 656.77 | **731.54** | $3,339,967 | 99.97% | **0.1120** |

### 7.2 Cracks Pool Spend & Mechanics
- **P1 Spend on Cracks Pool**: $\$1,629,136$, which represents **46.39%** of the $\$3,511,774$ pool budget (or **48.76%** of the baseline $\$3,340,965$ reference budget).
- **P2 / P2t Coincidence**: When the budget does not bind (only 49.50% spent), P2 and P2t select the exact same segments, yielding identical AW condition ($774.99\text{ CY}$) and $+0.23\text{ TW CY}$ delta.
- **Mechanism Designations**: Hypotheses explaining policy threshold dynamics (e.g., *[Hypothesis] bumps segments above preventive threshold*, *[Hypothesis] forces P1 to treat borderline segments*) are designated as hypotheses.
- **Bias vs. Noise Domination**: On the cracks pool, tile-level bias ($b = -5.0$) creates $\approx +109\text{ AW CY}$ regret, dominating zero-mean noise ($\lambda \in [0.5, 1.0]$, $+10$ to $+23\text{ AW CY}$ regret) by $5\times$ to $10\times$. Effects are non-additive.
- **Oracle Comparison**: Regret relative to the heuristic oracle can be negative because fixed threshold rules are not globally tuned to the 10-year dynamic horizon.

---

## 8. Command Center Application (`app.py`)

The Streamlit command center integrates all three components into an interactive dashboard:
- **Tab 1: Live Pavement Inspector**: Runs U-Net inference on 4 verified test examples or user uploads under the unified 720 px rule. Displays qualitative overlays, ASTM morphological metrics, sPCI score, and ASTM rating badge.
  - Performance: CPU inference takes ~8 s locally, 15–25 s on free cloud tiers.
  - Privacy: Images are processed in memory and never saved to disk.
- **Tab 2: Double-Blind Human Alignment**: Interactively displays pairwise validation statistics, confusion matrices, and the empirical 76.60% runtime agreement rate.
- **Tab 3: 10-Year Municipal Simulator**: Interactive Monte Carlo simulator enabling users to customize budget constraints ($0–20\%$), network size ($50–300$ segments), traffic weighting, and run comparative policy analyses (P0–P4, P2t).
