# road-pci-intelligence

An end-to-end deep learning and decision-support system designed to automate road distress detection, calculate pavement health metrics, and optimize infrastructure maintenance. Manual road pavement inspection is dangerous, subjective, and difficult to scale across municipal networks. By leveraging deep semantic segmentation to identify and quantify critical distress categories (longitudinal cracks, transverse cracks, alligator cracking, and potholes) from roadway imagery, this system computes an automated, simplified Pavement Condition Index (PCI) score and runs budget-constrained repair prioritization simulations to help municipal planners allocate road maintenance funds effectively.

---

## Planned Pipeline

```mermaid
flowchart LR
    A[Roadway Imagery] --> B[Deep Learning Segmentation]
    B --> C[Distress Classification & Severity Extraction]
    C --> D[Simplified PCI Calculation]
    D --> E[Budget-Constrained Repair Prioritization]
    E --> F[Interactive Dashboard Application]
```

1. **Data Ingestion & Preprocessing**: Sourcing, exploring, and unifying pavement distress benchmarks with pixel-level segmentation masks and multi-class annotations.
2. **Deep Semantic Segmentation**: Training deep segmentation networks (e.g., U-Net / Feature Pyramid Networks via `segmentation-models-pytorch`) to classify distress types (longitudinal, transverse, alligator cracks, potholes).
3. **Pavement Condition Index (PCI) Engine**: Measuring defect density and severity from predicted masks to formulate an objective, simplified PCI score (0–100 scale).
4. **Maintenance Optimization Simulation**: Formulating budget-constrained resource allocation algorithms (greedy heuristic & linear programming) to prioritize street repairs for maximum network condition improvement.
5. **Interactive Dashboard**: A Streamlit web application enabling users to upload inspection imagery, view segmentation overlays, inspect distress metrics, and simulate maintenance budgets.

---

## Status Checklist

- [x] **Environment & Tooling Verification** (Python 3.14, Git, Kaggle CLI, GitHub CLI authenticated)
- [x] **Project Scaffolding & Version Control** (Directory structure, requirements, git initialization, GitHub repository)
- [x] **Dataset Discovery & Selection** (Selected `pothole-image-segmentation-dataset`, `road-crack-dataset`, `crack500`, and `rdd2022es`)
- [x] **Step 1: Data Exploration & Schema Audit** *(Done)*:
  - Kaggle Notebook: [`vedshah04/road-pci-intelligence-data-exploration`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-data-exploration) ([`notebooks/01_data_exploration/`](notebooks/01_data_exploration/))
  - Directory tree audit, resolution stats, mask value distributions, duplication & base-ID leakage checks.
- [x] **Step 2: Manifests, Mask Standardization & Metric Reconciliation** *(Done)*:
  - Kaggle Notebooks:
    - Manifests & Segmentation Reconciliation: [`vedshah04/road-pci-intelligence-manifests-and-loader`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-manifests-and-loader) ([`notebooks/02_manifest_and_loader/`](notebooks/02_manifest_and_loader/))
    - Parallel RDD Manifest Builder: [`vedshah04/road-pci-intelligence-rdd-manifest-builder`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-rdd-manifest-builder) ([`notebooks/03_build_rdd_manifest/`](notebooks/03_build_rdd_manifest/))
  - Unified CSV manifests in `manifests/` with relative paths (`pothole`, `road_crack`, `crack500`, `rdd2022es`).
  - Standardized mask conversions & documented assumptions (`src/data_rules.py`).
  - Leak-proof group partitioning (pothole `pic-<N>`, road-crack scene 70/15/15, RDD2022ES mirror pairing).
  - Validation test suite (`tests/test_manifest.py`) passing 100% on Kaggle.
  - Rigorous ground-truth foreground metric reconciliation across all 1,686 masks.
  - Comprehensive documentation in [`docs/data_decisions.md`](docs/data_decisions.md).
- [x] **Step 3A: Training Pipeline, Sanity Checks & Smoke Test** *(Done)*:
  - Kaggle Notebook: [`vedshah04/road-pci-intelligence-train-baseline`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-train-baseline) ([`notebooks/04_train_baseline/`](notebooks/04_train_baseline/))
  - Pretrained U-Net (ResNet-34 ImageNet weights, `segmentation-models-pytorch`) with 2 independent sigmoid outputs `[crack, pothole]`.
  - Custom `MaskedMultiTaskLoss` (BCE + Soft Dice) with selective gradient masking for un-supervised channels (`tests/test_losses.py` passed 100%).
  - `SourceBalancedSampler` drawing 600 samples/epoch (200 crack500, 200 road_crack, 200 pothole).
  - Sanity Check 2 Overfit Test: Achieved **0.9601** Mean Dice on 8 samples (crack500: 0.9389, road_crack: 0.9509, pothole: 0.9906), well exceeding the > 0.85 target.
  - Sanity Check 3 Smoke Test: 3 full epochs completed with augmentation, AMP, and cosine decay (Epoch 3 train loss: 1.8012, mean val Dice: 0.4348).
  - Sanity Check 4: Saved 6 validation prediction overlays per source (`image | ground truth | prediction`) with clean titles.
- [x] **Step 3B: Baseline Training & Loss Mode Ablation** *(Done)*:
  - Masked Multi-Task Baseline (Run A, Seed 42, 30 epochs): [`vedshah04/road-pci-intelligence-train-masked-run-a`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-train-masked-run-a) ([`notebooks/05_train_masked/`](notebooks/05_train_masked/))
  - Naive All-Zero Negative Baseline (Run B, Seed 42, 30 epochs): [`vedshah04/road-pci-intelligence-train-naive-run-b`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-train-naive-run-b) ([`notebooks/06_train_naive/`](notebooks/06_train_naive/))
  - Comprehensive threshold sweep across $\{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ ([`notebooks/07_eval_threshold_sweep/`](notebooks/07_eval_threshold_sweep/)).
- [x] **Step 3C: Cross-Domain Evaluation & Multi-Seed Benchmarking** *(Done)*:
  - Cross-Domain RDD2022ES Check (Models A & B on 2,560 validation images, threshold 0.5): [`vedshah04/road-pci-intelligence-rdd-cross-domain`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-rdd-cross-domain) ([`notebooks/08_rdd_cross_domain/`](notebooks/08_rdd_cross_domain/))
  - Multi-seed runs across seeds 42, 43, 44 ([`notebooks/09_seed_runs/`](notebooks/09_seed_runs/)).
- [x] **Step 3D: Asymmetric Partial Supervision, Exhaustive Box Statistics & Combined Evaluation** *(Done)*:
  - Exhaustive RDD box statistics across all 25,599 label files (16 threads).
  - Asymmetric loss mode implementation (`src/losses.py`) with complete unit test suite (`tests/test_losses.py`).
  - Multi-seed asymmetric training (Seeds 42, 43, 44; 30 epochs each): [`vedshah04/road-pci-intelligence-asymmetric-runs`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-asymmetric-runs) ([`notebooks/11_train_asymmetric_and_rdd_eval/`](notebooks/11_train_asymmetric_and_rdd_eval/)).
  - Combined Cross-Domain RDD2022ES Evaluation (Models A, B, C on 2,560 validation images): [`vedshah04/road-pci-intelligence-rdd-cross-domain-eval-abc`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-rdd-cross-domain-eval-abc) ([`notebooks/08_rdd_cross_domain_abc/`](notebooks/08_rdd_cross_domain_abc/)).
- [x] **Step 3E: Segmentation Model Freeze, Official Test Evaluation & Cross-Domain Audit** *(Done)*:
  - Evaluation Notebook: [`vedshah04/road-pci-intelligence-final-evaluation`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-final-evaluation) ([`notebooks/12_final_eval/`](notebooks/12_final_eval/))
  - Frozen Model: Model B (Naive Multi-Task Loss, Run B, Seed 42, Epoch 30).
  - Weights & Model Card: Private Kaggle Dataset [`vedshah04/road-pci-seg-final-weights`](https://www.kaggle.com/datasets/vedshah04/road-pci-seg-final-weights) (SHA-256: `9269c5edb6e5d37b359bd3337fbd83c60901cd2b241f9ef789e9116fee400ef7`, 24,436,514 parameters).
  - Validation TTA Sweep: Flip-TTA improved validation Dice by +0.00637 ($\ge 0.0050$ threshold) $\rightarrow$ Adopted as Official Setting.
  - Test Split Evaluation (Evaluated ONCE, N = 200 images, threshold 0.5):
    - CRACK500 ($N=72$): Dataset IoU 0.4923, Dataset Dice 0.6597, Mean Img Dice 0.6545
    - Pothole ($N=60$): Dataset IoU 0.6805, Dataset Dice 0.8099, Mean Img Dice 0.7689
    - Road Crack ($N=68$): Dataset IoU 0.4495, Dataset Dice 0.6202, Mean Img Dice 0.5319 (Non-Empty Dice 0.6213 on 55 images; 2 / 13 clean images predicted empty).
  - Out-of-Domain RDD Test Audit ($N = 2,560$ images): Clean road false flag rate 27.50% at >1% (pothole) and 0.00% (crack); deep pothole box hit rate 35.42%. Documented as out-of-domain perspective limitation.
- [x] **Step 4A: Mask-to-Feature Module & Human Labeling Sheet** *(Done)*:
  - Feature Extractor: [`src/pavement_features.py`](src/pavement_features.py) with scale-normalized feature extraction (512 px shorter side), crack skeleton metrics (length, density, width, junctions, enclosed cells, network index), and pothole metrics (count, area fraction, largest pothole).
  - Unit Test Suite: [`tests/test_features.py`](tests/test_features.py) (8/8 tests passing on synthetic lines, plus signs, grids, disks, speckles, empty masks, scale invariance).
  - Feature Notebook: [`notebooks/13_features/`](notebooks/13_features/) ([`vedshah04/road-pci-intelligence-features-and-pci`](https://www.kaggle.com/code/vedshah04/road-pci-intelligence-features-and-pci)) running on CPU for ground-truth feature distributions and validation rank correlation with frozen model predictions.
  - Human Labeling Sheet: [`scripts/make_pair_sheet.py`](scripts/make_pair_sheet.py) generating 60 blinded test pairs across 10 pages with [`labels/pairs_template.csv`](labels/pairs_template.csv) (Seed 42).
  - Design Documentation: [`docs/pci_design.md`](docs/pci_design.md) documenting definitions, units, scale-invariance assumptions, and validation plans.
- [x] **Step 4B: Automated Simplified PCI (sPCI) Engine & Freeze** *(Done)*:
  - Rating Scale Alignment: Standard ASTM D6433 7-band pavement condition rating scale (86–100 Good, 71–85 Satisfactory, 56–70 Fair, 41–55 Poor, 26–40 Very Poor, 11–25 Serious, 0–10 Failed).
  - Calibrated Deduct Curves: Calibrated power deduct curves $\text{DV} = \min(100, a \cdot q^b)$ with empirical parameters ($a \in [5, 12, 25], b=0.75$ for cracks; $a \in [100, 160, 240], b=0.60$ for potholes) and network topology multiplier $m$.
  - Multi-Distress Deduct Aggregation: Multiplicative Corrected Deduct Value $\text{CDV} = 100 \cdot (1 - \prod (1 - \text{DV}_i/100))$, noise floor $\tau = 0.00$.
  - Implementation & Configuration: [`src/spci.py`](src/spci.py) and [`configs/spci_params.yaml`](configs/spci_params.yaml) verified and frozen under git tag `spci-v1-frozen`.
  - Comprehensive Unit Tests: [`tests/test_spci.py`](tests/test_spci.py) passing 10/10 tests (boundary values, monotone severity, scale invariance, empty masks).
- [x] **Step 4C: Human-Agreement Validation & Test Split Evaluation** *(Done)*:
  - Blinded Protocol: 60 test image pairs independently labeled by human evaluator in [`labels/pairs_human.csv`](labels/pairs_human.csv) (13 marked identical condition excluded, $N=47$ valid evaluation pairs).
  - Test Evaluation Execution: Model inferences and test pair features computed via Kaggle GPU Notebook 14 ([`notebooks/14_human_agreement/`](notebooks/14_human_agreement/)) with test scores frozen under tag `test-scores-computed` ([`results/spci_test_pairs.csv`](results/spci_test_pairs.csv)).
  - Pairwise Human Agreement: **76.60%** (36/47, 95% CI: [64.89%, 87.23%]), 2 ties ($|\Delta\text{sPCI}| < 1.0$).
    - CRACK500: **88.24%** (15/17)
    - Road Crack: **93.75%** (15/16)
    - Pothole: **42.86%** (6/14, 5 full + 2 ties @ 0.5)
  - Baseline Comparison: Damage Area Baseline **74.47%** (35/47, 95% CI: [61.70%, 85.11%]). Paired difference $+2.13\%$ (95% CI: $[-2.13\%, +7.45\%]$) — confirmed pre-registered interpretation of *no detectable difference*.
  - Full Test Split Benchmark ($N=200$ images, GT vs Predicted sPCI): CRACK500 Spearman $\rho = 0.9038$ ($\text{MAE} = 6.81$), Road Crack $\rho = 0.7547$ ($\text{MAE} = 11.80$), Pothole $\rho = 0.8123$ ($\text{MAE} = 10.28$).
- [x] **Step 5A & 5B: Budget-Constrained Maintenance Simulation & Refinements** *(Done)*:
  - Synthetic Road Network: 200 segments ($668,193\text{ m}^2$) with lognormal accelerated deterioration ($d\text{PCI} = r \cdot f(\text{PCI})$) and $k=6$ sampled inspection tiles from [`results/tile_scores.csv`](results/tile_scores.csv).
  - Maintenance Policies: P0 (do nothing), P1 (reactive worst-first), P2 (band-based worst-first), P3 (greedy benefit/cost heuristic), and P4 (exact 0/1 integer linear program via `scipy.optimize.milp` HiGHS branch-and-cut solver).
  - Spend Accounting & Utilization Audit: Corrected 10-year cumulative spend tracking ($\sum c_T \cdot A_i$). Discovered that P1 (Reactive) utilizes only 91.85% of budget in base scenario and 45.88% at 20% budget ($3.06\text{M}$ spent vs $6.68\text{M}$ available) because it exhausts segments with $\text{PCI} < 40$.
  - Metric Realignment: Evaluated both Area-Weighted (AW) and Traffic-Weighted (TW) condition-years. In traffic-weighted terms (matching municipal route priorities), P4 dominates with 731.55 TW CY (+87.54 over P1, +69.89 over P2).
  - Treatment-Model Sensitivity: Evaluated smooth effectiveness $\gamma = \text{clip}((\text{PCI} - 40)/30, 0, 1)$ showing that threshold attenuation reduces optimization regret on `all` pool while preserving P3/P4 dominance on `cracks`.
  - Information-Quality & Noise Scaling: Benchmarked continuous noise scaling $\lambda \in \{0, 0.5, 1, 1.5, 2\}$ showing monotonic regret growth for benefit-cost policies; detailed P1 physical accounting (166.4 vs 180.1 treated segments, 0 reconstructions, 2.87M vs 3.07M spend).
  - Dual Sensitivity Grids: Benchmarked 100 Monte Carlo runs across 18 parameter cells for both Grid 1 (scaling budget) and Grid 2 (fixed absolute budget) with paired bootstrap co-leader detection.
  - Test Split Overlays: Archived 8-image test split overlays ([`docs/figures/test_overlays_crack500.png`](docs/figures/test_overlays_crack500.png), [`docs/figures/test_overlays_pothole.png`](docs/figures/test_overlays_pothole.png), [`docs/figures/test_overlays_road_crack.png`](docs/figures/test_overlays_road_crack.png)) and human agreement visual audit ([`docs/figures/human_agreement_disagreements.png`](docs/figures/human_agreement_disagreements.png)).
  - Unit Test Suite: [`tests/test_simulation.py`](tests/test_simulation.py) passing 11/11 tests; entire repository suite passing 46/46 unit tests.
  - Comprehensive Documentation: [`docs/simulation.md`](docs/simulation.md) documenting illustrative assumptions, limitations, empirical results, and figures.
- [x] **Step 6: Interactive Streamlit Web Application** *(Done)*:
  - Command Center (`app.py`): Live Pavement Inspector, Double-Blind Human Alignment Explorer, and 10-Year Municipal Simulation Scenario Planner.
  - End-to-End Pipeline (`src/pipeline.py`): In-memory processing, strict SHA-256 weight verification, flip-TTA inference, and scale-normalized ASTM D6433 feature extraction.
  - Test Suite (`tests/test_pipeline.py`): Verified 100% agreement with held-out test split benchmark scores.

---

## Run Locally

### 1. Environment Setup

The application is compatible with Python 3.10 – 3.12:

```bash
# Clone repository
git clone https://github.com/skilloutofbounds/road-pci-intelligence.git
cd road-pci-intelligence

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements-app.txt
```

### 2. Model Weights Configuration

The frozen ResNet-34 U-Net perception backbone requires `best.pt` (SHA-256: `9269c5edb6e5d37b359bd3337fbd83c60901cd2b241f9ef789e9116fee400ef7`).
- The pipeline automatically attempts to load from candidate paths (`outputs/models/best.pt`, `scratch/downloaded_final_weights/best.pt`) or download from the configured Hugging Face Hub repository (`vedshah04/road-pci-intelligence`).
- To provide weights manually, place `best.pt` in `outputs/models/best.pt`.
- *Note: Demo Mode has been completely removed to prevent ungrounded inferences. If weights are missing, the app explicitly displays setup instructions.*

### 3. Launching the Command Center

```bash
streamlit run app.py
```

### 4. Empirical Performance Benchmarks (CPU Deployment)

Measured on a standard CPU workstation:
- **Total Cold-Start Time**: **8.70 s**
  - Dependency & Module Imports: 7.56 s
  - Model Load & SHA-256 Verification: 1.14 s
- **720×1280 Image CPU Inference (Flip-TTA, 3 passes)**: **8.41 s ± 0.08 s**
- **ASTM Features & sPCI Calculation**: Included in inference total (< 0.15 s)

### 5. Running the Automated Test Suite

```bash
# Run end-to-end perception pipeline fidelity tests (asserts <= 0.5 sPCI diff on 6 test examples)
python -m unittest -v tests/test_pipeline.py

# Run 10-year municipal maintenance simulation unit tests (14 passing tests)
python -m unittest -v tests/test_simulation.py
```

---

## Project Structure

```
road-pci-intelligence/
├── app_assets/           # Example held-out test split images & attribution
├── configs/              # Model and sPCI parameter configurations
├── data/                 # Raw and processed datasets (git-ignored)
├── docs/                 # Architectural documentation, facts, & decisions:
│   ├── PROJECT_FACTS.md         # Traceable facts record for all metrics & tags
│   ├── simulation.md            # Step 5C simulation results & error decomposition
│   ├── human_validation.md      # Step 4C double-blind alignment study
│   ├── pci_design.md            # ASTM D6433 feature extraction & sPCI formulation
│   └── experiments.md           # Model training, ablation & RDD limitation
├── labels/               # Human pairwise preference labeling sheets & templates
├── manifests/            # Version-controlled CSV manifests for training & validation
├── notebooks/            # Isolated Jupyter/Kaggle notebook modules (01 to 14)
├── outputs/              # Model checkpoints and reports (git-ignored)
├── results/              # Frozen evaluation scores & tabular benchmark results
├── scripts/              # Reproducible evaluation & simulation scripts
├── src/                  # Core modular source code:
│   ├── pipeline.py              # RoadPCIPipeline inference & overlay generator
│   ├── spci.py                  # Simplified PCI deduct curve calculation
│   ├── pavement_features.py     # ASTM D6433 morphological feature extraction
│   ├── losses.py                # Multi-task masked and naive loss functions
│   └── simulation/              # 10-year Monte Carlo engine, network, & policies
├── tests/                # Automated unit test suite:
│   ├── test_pipeline.py         # App pipeline fidelity on held-out test examples
│   ├── test_simulation.py       # 14 simulation mechanics & policy tests
│   ├── test_spci.py             # sPCI calculation & deduct value tests
│   ├── test_features.py         # ASTM morphological feature extraction tests
│   └── test_losses.py           # Multi-task loss gradient masking tests
├── .gitignore            # Git ignore configuration
├── app.py                # Streamlit Flagship Command Center application
├── requirements-app.txt  # Pinned dependencies for local execution
├── requirements.txt      # Project Python dependencies
└── README.md             # Project overview, roadmap, and local execution guide
```


