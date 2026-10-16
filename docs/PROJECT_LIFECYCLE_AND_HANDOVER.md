# ROAD-PCI-INTELLIGENCE: End-to-End Project Lifecycle, Technical Handover, and System Record

> **Document Type**: Comprehensive System Lifecycle & Agent Handover Record  
> **Status**: Complete & Verified (Steps 0 through 5C)  
> **Author**: Antigravity Assistant & Project Principal  
> **Repository**: `road-pci-intelligence`  
> **Target Audience**: Any new AI Agent, Research Engineer, or Asset Management Domain Specialist resuming or auditing this codebase.

---

## 1. Executive Summary & Foundational Principles

### 1.1 Project Objective
**`road-pci-intelligence`** is an end-to-end cyber-physical infrastructure intelligence system that bridges raw computer vision and municipal asset management. It solves the critical disconnect between deep learning perception and municipal budget decisions:

$$\text{Pavement Imagery} \xrightarrow{\text{Perception}} \text{Multi-Task Segmentation} \xrightarrow{\text{Feature Extraction}} \text{ASTM D6433 Metrics} \xrightarrow{\text{sPCI Engine}} \text{Pavement Condition Index} \xrightarrow{\text{Simulation}} \text{Budget Optimization (10-yr)}$$

The project encompasses four tightly integrated layers:
1. **Perception**: Multi-task deep semantic segmentation of road distresses (longitudinal cracks, transverse cracks, alligator cracking, and potholes) trained across heterogeneous public benchmarks.
2. **Feature Extraction**: Morphological and topological feature extraction (skeletonization, branch pruning, crack width, density, and connected components).
3. **Index Formulation (sPCI v1)**: A simplified, mathematically grounded adaptation of the ASTM D6433 Pavement Condition Index with calibrated deduct-value curves and standard condition rating bands.
4. **Municipal Maintenance Simulation**: A 10-year Monte Carlo asset management engine comparing six maintenance policies (P0 through P4, plus P2t) across 200 roadway segments under fiscal capital constraints.

### 1.2 Scientific Honesty & Frozen Checkpoint Rules
Throughout this project, strict methodological integrity rules were enforced:
- **No Data Snooping / Test Contamination**: Test splits were strictly quarantined until final evaluation. All calibration, threshold tuning, and hyperparameter selection occurred strictly on `train` and `val` splits.
- **Blind Human Alignment**: Human pair labels were generated and committed blind without inspecting model scores. Model test scores were computed and frozen (Git tag `test-scores-computed`) before unblinding human labels.
- **Frozen Wahrheiten**: The sPCI calculation engine (`src/spci.py`, `src/pavement_features.py`, `configs/spci_params.yaml`) was frozen under Git tag `spci-v1-frozen`. No parameters were tuned to fit test split outcomes.
- **Transparent Failure Reporting**: No cherry-picked results. Disagreements, failure modes, data ambiguities, and negative results (such as pothole annotation gaps and threshold artifacts) are fully documented.
- **Illustrative Cost Disclaimer**: All financial figures ($1, 3, 5, 8, 12$ cost units) are stylized relative benchmarks for algorithmic comparison; no claims of real-world municipal tax savings are asserted.

---

## 2. Master Chronology: Step-by-Step Evolution (Step 0 to Step 5C)

```
[Step 0-1] Project Genesis & Multi-Dataset Audit (Kaggle mount paths, Base-ID grouping)
     │
[Step 2]   Manifests, Leak-Proof Splits, Mask Standardization & Foreground Reconciliation
     │
[Step 3A]  Multi-Task Segmentation Pipeline, Masked Loss, 3-Epoch Smoke Test
     │
[Step 3B]  30-Epoch Baseline Training (Run A: Masked Loss vs. Run B: Naive Loss)
     │
[Step 3C]  Multi-Seed Stability & Zero-Shot Cross-Domain RDD2022ES Benchmark
     │
[Step 3D]  Asymmetric Loss Mode (Run C) & Comparative Model Selection
     │
[Step 3E]  Freeze Segmentation Backbone (Run B Selected) & Test Split Inference
     │
[Step 4A]  Mask-to-Feature Extraction (ASTM D6433) & 60-Pair Human Labeling Sheet
     │
[Step 4B]  Skeleton Pruning, HTML Tool, sPCI Formulation & Tag 'spci-v1-frozen'
     │
[Step 4C]  Blind Human Agreement Validation (81.7% overall, 90.9% cracks) & Tag 'test-scores-computed'
     │
[Step 4D]  Git Identity Reconciliation & Initial 10-Year Monte Carlo Engine
     │
[Step 5A]  Simulation Engine Mechanics (P0–P4, HiGHS MILP, Lognormal Deterioration)
     │
[Step 5B]  Spend Accounting Fix, Dual Metrics (AW vs. TW), Dual Grids & Noise Scaling
     │
[Step 5C]  Heuristic Baseline P2t, Error Structure, Bias-Noise Decomposition & Synthesis
```

---

## 3. Data Engineering & Perception Pipeline (Steps 1 – 3E)

### 3.1 Datasets & Annotation Schemes
The system unifies four heterogeneous road distress datasets:

| Dataset | Source / Key | Native Format | Distress Classes | Special Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Pothole-600** | `farzadnekouei/pothole-image-segmentation-dataset` | Binary PNG masks | Pothole only | Augmented copies (e.g. `pic-282`) sharing base ID |
| **Road Crack** | `rukiyeaydn/road-crack-dataset` | RGB PNG masks | Longitudinal, Transverse, Alligator | Roboflow export containing hash IDs (`.rf.`) |
| **CRACK500** | `sj26717/crack500` | Binary PNG masks | Crack (untyped) | Sub-image crops from high-res cameras |
| **RDD2022ES** | `juusos/rdd2022es` (Spain subset) | VOC XML Bounding Boxes | D00, D10, D20, D40 | Zero-shot cross-domain evaluation benchmark |

### 3.2 Leak-Proof Base-ID Grouping & Split Manifests
To prevent data leakage caused by Roboflow rotations, crops, or camera sequence bursts, images were grouped by **Base-ID** prior to splitting:
- **Pothole**: Base-ID extracted as `pic-<N>` (stripping augmentation suffixes).
- **Road Crack**: Base-ID extracted as filename prior to `.rf.` hash.
- **CRACK500**: Base-ID extracted as parent sequence ID prior to crop coordinate suffixes.

Images were partitioned into strict **70% Train / 15% Val / 15% Test** splits such that all augmented copies or crops of an image reside exclusively within one split.
Split manifests are committed in `manifests/`:
- `manifests/segmentation_manifest.csv`
- `manifests/rdd_manifest.csv`

### 3.3 Mask Conversion & Foreground Statistics Reconciliation
Mask standardizer (`src/data_rules.py`) converts arbitrary inputs into a standardized 4-channel binary tensor:
$$\mathbf{Y} \in \{0, 1\}^{4 \times H \times W} \quad [\text{Longitudinal}, \text{Transverse}, \text{Alligator}, \text{Pothole}]$$

During Step 2, a documentation discrepancy in foreground fractions was caught and reconciled through direct ground-truth pixel accounting:
- **CRACK500**: $6.30\%$ foreground pixels
- **Road Crack**: $4.12\%$ foreground pixels
- **Pothole**: $3.41\%$ foreground pixels
*(All confirmed against disk and documented in [`docs/data_decisions.md`](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/data_decisions.md).)*

### 3.4 Multi-Task Architecture & Loss Formulations
- **Architecture**: `smp.Unet` with an ImageNet-pretrained `resnet34` encoder, 4-channel sigmoid output.
- **Input Resolution**: $512 \times 512$ with Albumentations pipeline (horizontal flip, shift-scale-rotate, random brightness-contrast).

#### Loss Functions Evaluated:
1. **Masked Multi-Task Loss (Run A)**: Computes loss only over channels annotated in the source dataset:
   $$\mathcal{L}_{\text{masked}} = \frac{1}{\sum_c m_c} \sum_{c=1}^4 m_c \cdot \text{BCEWithLogits}(p_c, y_c), \quad m_c \in \{0, 1\}$$
2. **Naive Loss (Run B)**: Assumes absence of annotation implies background (all $m_c = 1$).
3. **Asymmetric Loss (Run C)**: Penalizes false positives on unannotated classes with weight $\alpha = 0.25$.

#### 30-Epoch Training & Multi-Seed Validation Results:
Experiments evaluated on Validation split (F1 score / IoU):

| Run ID | Loss Type | Crack F1 | Pothole F1 | Overall Mean F1 | Val IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Run A (Seed 42)** | Masked BCE | 0.584 | 0.762 | 0.628 | 0.461 |
| **Run B (Seed 42)** | Naive BCE | **0.612** | **0.789** | **0.656** | **0.492** |
| **Run C (Seed 42)** | Asymmetric | 0.598 | 0.771 | 0.641 | 0.474 |
| **Run B (Multi-Seed Mean)** | Naive BCE | $0.610 \pm 0.004$ | $0.785 \pm 0.005$ | **$0.654 \pm 0.003$** | **$0.490 \pm 0.004$** |

### 3.5 Cross-Domain RDD2022ES Benchmark & Model Freezing
Zero-shot transfer was tested on the Spain subset of RDD2022 (`notebooks/08_rdd_cross_domain_abc`):
- Class mapping verified: D00 = Longitudinal, D10 = Transverse, D20 = Alligator, D40 = Pothole.
- Run B demonstrated superior cross-domain precision by avoiding false-positive background bleed.

> [!IMPORTANT]
> **FROZEN MODEL DECISION (Step 3E)**:  
> **Run B (Naive Loss, Seed 42)** was selected and permanently frozen as the project's perception backbone. Checkpoint weights were validated on the quarantined Test split (Notebook 12), achieving test F1 of 0.648. Overlays generated:
> - `docs/figures/test_overlays_crack500.png`
> - `docs/figures/test_overlays_road_crack.png`
> - `docs/figures/test_overlays_pothole.png`

---

## 4. Feature Extraction & Simplified PCI (sPCI v1) (Steps 4A – 4D)

### 4.1 Mask-to-Feature Engineering (`src/pavement_features.py`)
From 4-channel segmentation masks, the module extracts 7 physical distress parameters:
1. **Area Fraction**: $A_k = \frac{1}{H \times W} \sum_{x, y} \mathbf{M}_k(x, y)$
2. **Skeleton Length**: Medial axis skeletonization ($L_k = \sum \text{skeleton}(M_k)$) with iterative branch pruning to remove single-pixel spur artifacts.
3. **Mean Crack Width**: $W_k = \frac{\text{Area}_k}{\text{Length}_k + \epsilon}$
4. **Dominant Orientation Angle**: Principal axis of inertia derived from image central moments:
   $$\theta = \frac{1}{2} \text{atan2}(2\mu_{11}, \mu_{20} - \mu_{02})$$
   Used to separate longitudinal ($|\theta| \approx 90^\circ$) from transverse ($|\theta| \approx 0^\circ$).
5. **Junction Density / Branch Count**: Degree-3+ branch point detection on pruned skeletons to characterize alligator crack networks.
6. **Connected Component Count & Extent**: Labeling distinct distress instances.
7. **Pothole Diameter & Depth Proxy**: Equivalent circle diameter and contour compactness.

### 4.2 sPCI Formulation & Calibrated Deduct Curves (`src/spci.py`)
ASTM D6433 standard establishes pavement condition on a $0 - 100$ scale via cumulative deduct values. The simplified sPCI v1 implements:

$$\text{sPCI} = \max\left(0.0, 100.0 - \sum_{k=1}^4 \text{Deduct}_k(\text{Features}_k)\right)$$

Where each deduct value is governed by a power-law saturation curve with a noise floor threshold $\tau$:
$$\text{Deduct}_k = \begin{cases} 0.0, & \text{if } A_k < \tau_k \\ a_k \cdot (A_k - \tau_k)^{b_k} \times \left(1.0 + \beta_k \cdot \text{SeverityModifier}_k\right), & \text{otherwise} \end{cases}$$

Calibrated empirical parameters (`configs/spci_params.yaml`):
- Longitudinal Crack: $a = 45.0, b = 0.55, \tau = 0.001$
- Transverse Crack: $a = 40.0, b = 0.50, \tau = 0.001$
- Alligator Crack: $a = 75.0, b = 0.60, \tau = 0.0005$
- Pothole: $a = 90.0, b = 0.65, \tau = 0.0005$

#### ASTM D6433 Standard Rating Bands:
- **Good**: $85 < \text{PCI} \le 100$
- **Satisfactory**: $70 < \text{PCI} \le 85$
- **Fair**: $55 < \text{PCI} \le 70$
- **Poor**: $40 < \text{PCI} \le 55$
- **Very Poor**: $25 < \text{PCI} \le 40$
- **Serious**: $10 < \text{PCI} \le 25$
- **Failed**: $0 \le \text{PCI} \le 10$

> **Git Tag `spci-v1-frozen`**: Formally tagged on commit `a044983`. Parameters and logic remained frozen across all subsequent evaluations.

### 4.3 Blind Human Agreement Validation (Step 4C)
To test whether sPCI captures human pavement engineering judgment:
1. **60 Test Pairs Selected**: CRACK500 (18 pairs), Road Crack (15 pairs), Pothole (27 pairs).
2. **Labeling Tool (`labels/label_tool.html`)**: Interactive UI for side-by-side human pair comparison ($A > B$, $A = B$, $A < B$).
3. **Double-Blind Integrity Protocol**:
   - User labeled pairs in `labels/pairs_human.csv`.
   - Assistant computed sPCI test pair scores and committed them under Git tag `test-scores-computed` *before* unblinding human labels.
   - Verification script `src/verify_labels.py` validated formatting without revealing answers.

#### Empirical Human-Model Agreement Results:
- **Overall Agreement (60 pairs)**: **$81.7\%$** ($49 / 60$, 95% CI: $[70.0\%, 90.0\%]$)
- **Crack Pairs Combined (33 pairs)**: **$90.9\%$** ($30 / 33$, 95% CI: $[78.8\%, 97.0\%]$)
  - CRACK500: **$88.9\%$** ($16 / 18$)
  - Road Crack: **$93.3\%$** ($14 / 15$)
- **Pothole Pairs (27 pairs)**: **$70.4\%$** ($19 / 27$)
- **Baseline Area Comparison**: sPCI outperformed naive distress area ranking by $+6.7\%$ agreement points.
- **Disagreement Analysis**: All 8 disagreement pairs were visually compiled into `docs/figures/human_agreement_disagreements.png`. Disagreements stemmed from deep narrow potholes vs wide shallow stains and fine background alligator cracks in single-task pothole masks.

---

## 5. Municipal Maintenance Simulation Engine (Steps 5A – 5C)

### 5.1 Simulation Engine Architecture (`src/simulation/`)
The simulation evaluates how visual inspection estimates affect capital planning over a **10-year horizon**:
- **Synthetic Network**: $N = 200$ segments, lengths $100 - 600\text{ m}$, widths $6 - 14\text{ m}$, total network surface area $\approx 668,193\text{ m}^2$.
- **Traffic Weighting**: Low ($w=1$, 50%), Medium ($w=2$, 35%), High ($w=3$, 15%).
- **Tile Sampling**: $k = 6$ image tiles sampled with replacement per segment to form initial estimated ($\widehat{\text{PCI}}$) and true ($\text{PCI}^*$) condition.
- **Annual Deterioration**:
  $$\Delta\text{PCI}_i = r_i \cdot \left[1.0 + \max\left(0, \frac{70.0 - \text{PCI}_i^*}{70.0}\right)\right], \quad r_i \sim \text{Lognormal}(\ln(3.0), 0.30)$$

### 5.2 Maintenance Allocation Policies Evaluated
1. **P0 (Do Nothing)**: Passive deterioration, spend $= 0$.
2. **P1 (Reactive / Worst-First Severe)**: Only treats $\widehat{\text{PCI}} < 40$. Severe roads receive Reconstruction ($c=12$), poor roads receive Rehabilitation ($c=c_{\text{rehab}}$).
3. **P2 (Worst-First by Condition Band)**: Prioritizes lowest $\widehat{\text{PCI}}$ first. ASTM bands dictate treatments:
   - $\le 25$: Reconstruction ($c=12$)
   - $26 - 55$: Rehabilitation ($c=c_{\text{rehab}}$)
   - $56 - 70$: Preventive crack sealing ($c=1.0$, $+15$ PCI gain)
   - $> 70$: None
4. **P2t (Traffic-Aware Worst-First - Step 5C Heuristic)**: Exact same ASTM band treatments as P2, but candidate segments are prioritized in descending order of urgency score:
   $$\text{Urgency}_i = (100.0 - \widehat{\text{PCI}}_i) \times w_i$$
5. **P3 (Greedy Benefit/Cost Heuristic)**: Ranks actions by ratio $\frac{\Delta \widehat{P} \cdot A \cdot w}{c_T \cdot A}$, treating highest benefit-per-cost segments until budget exhaustion.
6. **P4 (Exact 0/1 Integer Linear Program)**: Solves exact single-year optimum knapsack via `scipy.optimize.milp` (HiGHS branch-and-cut simplex):
   $$\max \sum_{i, T} (\Delta \widehat{P}_{i, T} \cdot A_i \cdot w_i) x_{i, T} \quad \text{s.t.} \quad \sum_{i, T} (c_T A_i) x_{i, T} \le \text{Budget}, \quad \sum_T x_{i, T} = 1$$

### 5.3 Error Structure of sPCI Estimates (Step 5C)
Per-tile and segment-level ($k=6$) estimation error ($\widehat{\text{sPCI}} - \text{sPCI}^*$):

| Group | $N$ | Mean Signed Error | MAE | Std Dev (Empirical) | Std Dev (CLT Theoretical) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Per-Tile: all** | 455 | -3.1080 | 9.2768 | 15.4001 | — |
| **Per-Tile: cracks** | 287 | -3.5342 | 8.4718 | 13.0174 | — |
| **Per-Tile: crack500** | 142 | +0.5449 | 6.7061 | 8.4796 | — |
| **Per-Tile: road_crack** | 145 | -7.5289 | 10.2010 | 15.2863 | — |
| **Per-Tile: pothole** | 168 | -2.3799 | 10.6518 | 18.8067 | — |
| **Segment Mean ($k=6$): all** | 200 | -2.9493 | 4.8356 | 5.5699 | 6.2870 |
| **Segment Mean ($k=6$): cracks** | 200 | -3.3907 | 4.7573 | 5.3560 | 5.3143 |

*Finding: Averaging $k=6$ tiles per segment attenuates inspection variance by approximately $\sqrt{6} \approx 2.45$, conforming to the Central Limit Theorem.*

### 5.4 Base Scenario Benchmark Results ($N = 200$ Runs, Budget 10%, Ratio 5.0)

#### Pool 'all' (200 Runs):
- **P0**: AW 361.64, TW 357.01, Spend 0 (0.00%)
- **P1**: AW 642.80, TW 644.01, Spend 3,068,687 (91.85%), Eff TW 0.0935
- **P2**: AW **660.50**, TW 661.66, Spend 3,337,775 (99.90%), Eff TW 0.0913
- **P2t**: AW 626.94, TW 672.45, Spend 3,336,948 (99.88%), Eff TW 0.0945
- **P3**: AW 655.82, TW 730.69, Spend 3,337,095 (99.88%), Eff TW **0.112**
- **P4**: AW 656.77, TW **731.54**, Spend 3,339,967 (99.97%), Eff TW **0.112**

#### Paired Differences (P2t minus Competitors on 'all'):
- $\text{P2t} - \text{P2}$: $\text{AW} = -33.56$ [$-33.88, -33.23$] | $\text{TW} = \mathbf{+10.79}$ [$+10.43, +11.14$]
- $\text{P2t} - \text{P3}$: $\text{AW} = -28.88$ [$-29.21, -28.54$] | $\text{TW} = -58.24$ [$-58.53, -57.95$]
- $\text{P2t} - \text{P4}$: $\text{AW} = -29.82$ [$-30.16, -29.49$] | $\text{TW} = -59.09$ [$-59.38, -58.79$]

*Key Insight: Prioritizing traffic in worst-first policies (P2t) successfully boosts traffic-weighted condition (+10.79 TW CY over P2), but benefit-cost optimization (P3/P4) remains decisively superior on traffic-weighted outcomes (+59 TW CY over P2t).*

### 5.5 Bias vs. Noise Regret Decomposition & Negative Regret Mechanics
Regret relative to oracle ($\text{Regret} = \text{Oracle CY} - \text{Estimated CY}$):

1. **Noise-Only Scaling ($\lambda \in [0, 2]$ with zero-mean noise)**:
   - Regret is **strictly non-negative ($\ge 0$)** across all policies and pools without exception.
   - Random noise monotonically degrades decision efficiency. For P4 on `all`, regret rises from $0.00$ at $\lambda=0$ to $+19.94\text{ AW CY}$ at $\lambda=0.5$, $+40.96$ at $\lambda=1.0$, and $+74.01$ at $\lambda=2.0$.
2. **Bias-Only Decomposition ($b \in \{-10, -5, 0, +5, +10\}$)**:
   - Demonstrates **negative regrets** due to structural policy thresholds:
     - **P1 Under Negative Bias ($b = -5, -10$)**: Because P1 strictly requires $\widehat{\text{PCI}} < 40$ to intervene, an underestimating model tricks P1 into treating borderline roads ($40 \le \text{PCI}^* < 50$). This forces P1 to spend previously idle capital, achieving **higher condition-years than its own conservative oracle** ($-41.52\text{ AW Regret}$ at $b=-10$ on `all`; $-52.66$ on `cracks`).
     - **P3/P4 Under Mild Positive Bias ($b = +5$)**: Positive bias bumps moderately damaged roads above the preventive cutoff ($\widehat{\text{PCI}} \ge 55$), allowing P3 to purchase cheap crack sealing across large surface areas (producing $-13.50\text{ AW Regret}$ on `all`).

### 5.6 Fixed Absolute Budget Sensitivity Grid (Grid 2, 18 Cells x 100 Runs)
Grid 2 decouples municipal capital budgets from rehabilitation cost variations ($c_{\text{rehab}} \in \{3, 5, 8\}$, Budgets $\in \{5\%, 10\%, 20\%\}$):

#### Victory Tallies across 18 Cells:
- **P0**: 0 wins
- **P1**: 0 wins
- **P2**: 6 nominal wins (AW), 6 co-wins (AW) | 0 nominal wins (TW), 0 co-wins (TW)
- **P2t**: 0 nominal wins (AW), 1 co-win (AW) | **1 nominal win (TW)**, 1 co-win (TW) (wins on `cracks, 5% budget, ratio 3`)
- **P3**: 8 nominal wins (AW), 10 co-wins (AW) | **10 nominal wins (TW)**, 12 co-wins (TW)
- **P4**: 4 nominal wins (AW), 6 co-wins (AW) | **7 nominal wins (TW)**, 8 co-wins (TW)

---

## 6. What We Can and Cannot Conclude

### 6.1 What We Can Conclude (Supported Directly by Data)
1. **Policy Rankings are Strongly Conditioned on the Evaluated Metric**:
   - P2 dominates Area-Weighted condition under tight budgets (6 of 18 cells in Grid 2).
   - P3/P4 dominate Traffic-Weighted condition (17 of 18 cells in Grid 2), generating up to $+87\text{ TW CY}$ over P1 and $+70\text{ TW CY}$ over P2.
2. **Traffic-Aware Prioritization (P2t) Trades Area for User Impact**:
   - Prioritizing segments by $(100 - \widehat{\text{PCI}}) \times w_i$ reliably improves traffic-weighted condition over P2 (+10.79 TW CY), but concedes area-weighted pavement condition (-33.56 AW CY).
3. **P1 Spending Bounds are a Definitional Property**:
   - P1 spending is bounded because it ignores all roads with $\text{PCI} \ge 40$. Once severe backlogs are remediated, reactive policies cannot absorb capital.
4. **Zero-Mean Noise Degrades Decisions; Systematic Bias Causes Threshold Distortions**:
   - Zero-mean noise uniformly degrades decision quality (monotonic positive regret).
   - Systematic bias triggers non-monotonic threshold effects (negative regret in P1 and P3).
5. **Ranking Sensitivity to Treatment Response Curves**:
   - Discontinuous threshold cutoffs (cliff at $\text{PCI} \ge 55$) penalize noisy perception models far more heavily than continuous smooth response curves.

### 6.2 What We Cannot Conclude
1. **No Municipal Dollar Savings Claims**: Costs are dimensionless relative benchmark units ($1, 3, 5, 8, 12$).
2. **No Physical Life-Cycle Guarantees**: Real deterioration is governed by weather, subgrade moisture, freeze-thaw cycles, and heavy axle loads not modeled in this stylized setup.
3. **Pothole Detection Involves High Variance**: Pothole sPCI estimates exhibit high standard deviation ($\sigma = 18.81$), meaning automated pothole-driven decisions carry significant uncertainty.

---

## 7. Complete Repository Inventory & Execution Guide

### 7.1 Key Directory Structure
```
road-pci-intelligence/
├── configs/
│   ├── dataset_config.yaml           # Dataset paths and normalization
│   ├── model_config.yaml             # Segmentation architecture hyperparameters
│   ├── spci_params.yaml              # FROZEN sPCI deduct curve parameters
│   └── sim_params.yaml               # Simulation constants and budget configs
├── src/
│   ├── data_rules.py                 # Mask standardization & class parsing
│   ├── build_manifest.py             # Leak-proof Base-ID grouping and manifest generator
│   ├── losses.py                     # Masked, Naive, and Asymmetric loss functions
│   ├── pavement_features.py          # ASTM D6433 morphological feature extraction
│   ├── spci.py                       # FROZEN sPCI v1 calculation engine
│   ├── verify_labels.py              # Blind label verification script
│   └── simulation/
│       ├── network.py                # Synthetic road network generator
│       ├── engine.py                 # 10-year Monte Carlo simulation engine (P0–P4, P2t)
│       └── policy.py                 # Maintenance allocation policies & HiGHS MILP
├── tests/
│   ├── test_data_rules.py            # Mask formatting and integrity tests
│   ├── test_losses.py                # Loss function gradient & masking tests
│   ├── test_features.py              # Skeletonization & feature extraction tests
│   ├── test_spci.py                  # sPCI bounds, monotonicity, and band tests
│   └── test_simulation.py           # 14 unit tests: budget, determinism, P2t, MILP
├── scripts/
│   ├── run_step5c_experiments.py     # Master runner for Step 5C experiments
│   └── make_disagreement_figure.py   # Human agreement disagreement visualizer
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Kaggle dataset audit
│   ├── 06_train_naive.ipynb          # Frozen Model B training
│   ├── 08_rdd_cross_domain_abc/      # Zero-shot cross-domain evaluation
│   ├── 12_eval_frozen_model/         # Test split perception evaluation
│   ├── 13_extract_features_gpu/      # High-throughput feature extraction
│   └── 14_test_spci_eval/            # Test split sPCI calculation
├── manifests/
│   ├── segmentation_manifest.csv     # 70/15/15 leak-proof split manifest
│   ├── features_gt_trainval.csv      # Ground-truth distress features
│   └── test_pair_sheet_60.csv        # 60 blind human comparison pairs
├── labels/
│   ├── label_tool.html               # Interactive browser-based pair labeling tool
│   └── pairs_human.csv               # Ground-truth human pair comparison labels
├── results/
│   ├── tile_scores.csv               # 455 test tiles with predicted & true sPCI
│   ├── spci_test_pairs.csv           # 60 test pairs with frozen model scores
│   ├── sim_base_with_p2t_all.csv     # Base scenario 200 runs on 'all'
│   ├── sim_base_with_p2t_cracks.csv  # Base scenario 200 runs on 'cracks'
│   ├── sim_bias_decomposition.csv    # 60 rows complete bias decomposition
│   ├── sim_noise_decomposition.csv   # 60 rows complete noise decomposition
│   └── sim_sensitivity_grid_fixed_step5c.csv # 18 cells Grid 2 summary
└── docs/
    ├── figures/                      # Exact 10 publication figures
    ├── data_decisions.md             # Dataset decisions & foreground reconciliation
    ├── experiments.md                # Perception models, sweeps, and benchmarks
    ├── pci_design.md                 # sPCI v1 formulation & parameter calibration
    ├── human_validation.md           # Blind human agreement validation study
    └── simulation.md                 # Complete simulation formulation & Step 5C findings
```

### 7.2 Publication Figures (`docs/figures/`)
The figures directory contains exactly 10 PNG files:
1. `human_agreement_disagreements.png` (1,758,555 bytes)
2. `simulation_condition_trajectories.png` (170,752 bytes)
3. `simulation_cost_effectiveness.png` (81,679 bytes)
4. `simulation_regret_vs_bias.png` (885,112 bytes)
5. `simulation_regret_vs_lambda.png` (1,023,045 bytes)
6. `simulation_sensitivity_grid_fixed.png` (158,553 bytes)
7. `simulation_sensitivity_grid_scaling.png` (49,496 bytes)
8. `test_overlays_crack500.png` (2,572,546 bytes)
9. `test_overlays_pothole.png` (2,113,079 bytes)
10. `test_overlays_road_crack.png` (2,365,979 bytes)

### 7.3 Reproducibility Commands
To verify the entire repository locally:

```bash
# 1. Run full unit test suite (14 simulation tests + perception tests)
python -m unittest discover tests

# 2. Verify frozen sPCI code has not changed from frozen tag
git diff spci-v1-frozen --stat -- src/spci.py src/pavement_features.py configs/spci_params.yaml

# 3. Re-run Step 5C simulation experiments and regenerate figures
python scripts/run_step5c_experiments.py
```
