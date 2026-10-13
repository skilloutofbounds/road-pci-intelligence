# Human Pairwise Agreement Validation (Step 4C)

## 1. Executive Summary and Protocol

This document reports the empirical validation of the frozen simplified Pavement Condition Index (**sPCI v1**, tag `spci-v1-frozen`) against blinded human pairwise deterioration judgments on the test split.

### Pre-Registered Evaluation Protocol
1. **Blinded Scoring**: Model predictions, morphological features, and sPCI ratings across all 120 test pair images were computed and committed (`results/spci_test_pairs.csv`, SHA-256: `47ce67e07d9b39858be672ef46e0c8d1ea13216b5ec85a13d93a6888a8ae7c91`) under tag `test-scores-computed` **prior to reading, opening, or unblinding** human labels in `labels/pairs_human.csv`.
2. **Scoring Logic**:
   - For each pair where the human labeled image A or B as worse:
     - sPCI selects the image with the **lower sPCI** as worse.
     - **Tie Rule**: If $|\text{sPCI}_A - \text{sPCI}_B| < 1.0$ point, the decision is designated a tie and awarded **0.5 points**.
     - Otherwise, agreement = **1.0 point** if model pick matches human pick, and **0.0 points** if model pick differs.
   - Pairs labeled as **"same"** by the human are excluded from the denominator $N$.
   - **Primary Metric**:
     $$\text{Agreement} = \frac{\sum (\text{Agreements} + 0.5 \times \text{Ties})}{N}$$
3. **Statistical Inference**:
   - 95% Bootstrap Confidence Intervals computed using 10,000 resamples with replacement (fixed seed 42).
   - Paired difference against the **Damage Area baseline** ($\text{crack\_area\_frac} + \text{pothole\_area\_frac}$, larger area = worse).
   - **Pre-registered reading**: sPCI is described as better than the area baseline only if the 95% bootstrap CI strictly excludes 0; otherwise reported as *"no detectable difference"*.

---

## 2. Integrity Verification and Timestamps

The following cryptographic Git commits and timestamps document that the sPCI model and parameter specification were frozen before human labels were committed, and that test scores were committed before human labels were unblinded:

| Milestone / Artifact | Git Ref / File | Commit Hash | Timestamp (Local / UTC) |
| :--- | :--- | :--- | :--- |
| **sPCI Parameter Freeze** | Tag `spci-v1-frozen` | `68a30aa532f9c7ae9eb36890915d957656eaa8d7` | Mon Oct 12 18:14:34 2026 +0530 (12:44:34 UTC) |
| **Human Blind Labels** | `labels/pairs_human.csv` | `372a7389fbd515892ed101ad5892aafea6104b3e` | Mon Oct 12 18:54:47 2026 +0530 (13:24:47 UTC) |
| **Test Scores Computed** | Tag `test-scores-computed` | `3a097be85123ca1a3b45ff4fdb47cd8a7bc98ed8` | Mon Oct 12 21:20:42 2026 +0530 (15:50:42 UTC) |

- `git diff spci-v1-frozen --stat -- src configs`: **Empty** (0 insertions, 0 deletions; confirms bitwise preservation of frozen code and YAML).
- `python -m src.verify_labels labels/pairs_human.csv`: **PASS** (60 rows, valid tokens {'A', 'B', 'same'}, verified without printing labels).

---

## 3. Human Agreement Results

### 3.1 Primary Results Table

- **Total Pairs**: 60
- **Human Label Distribution**: A = 27, B = 20, same = 13
- **Evaluated Pairs ($N$)**: 47 (13 "same" pairs excluded)
- **Ties ($|\Delta\text{sPCI}| < 1.0$)**: 2 (both in the pothole subset, score = 0.5 each)

| Subset / Source | Valid Pairs ($N$) | Ties | Disagreements | Agreements | Agreement Rate | 95% Bootstrap CI (10k draws, seed 42) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (All Distress Types)** | **47** | **2** | **10** | **35** | **76.60%** | **[64.89%, 87.23%]** |
| `crack500` | 17 | 0 | 2 | 15 | **88.24%** | [70.59%, 100.00%] |
| `road_crack` | 16 | 0 | 1 | 15 | **93.75%** | [81.25%, 100.00%] |
| `pothole` | 14 | 2 | 7 | 5 | **42.86%** | [17.86%, 67.86%] |

### 3.2 Comparison with Baselines

| Method / Baseline | Agreement Rate | 95% Bootstrap CI | Comparison vs. sPCI |
| :--- | :---: | :---: | :--- |
| **Frozen sPCI Model** | **76.60%** | **[64.89%, 87.23%]** | — |
| **Damage Area Baseline** | 74.47% | [61.70%, 85.11%] | sPCI is +2.13 percentage points higher |
| **Random Guessing** | 50.00% | — | sPCI is +26.60 percentage points higher |
| **Ground-Truth sPCI Reference** (`crack500` + `road_crack`, $N=33$) | 90.91% | [78.79%, 100.00%] | Predicted sPCI achieves identical 90.91% on this subset |

*(Note: For pothole ground-truth masks, annotations contain only the pothole channel; hence the GT reference comparison is reported on the crack benchmarks).*

### 3.3 Paired Difference Test (sPCI vs. Damage Area Baseline)

- **Mean Paired Difference**: $+2.13\text{ percentage points}$ ($+0.0213$)
- **95% Bootstrap CI (10,000 resamples, seed 42)**: **$[-2.13\%, +7.45\%]$**

| Subset / Source | Valid Pairs ($N$) | Ties | sPCI Agreement (95% CI) | Area Baseline Agreement (95% CI) | Paired Difference sPCI - Area (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 47 | 2 | **76.60%** [64.89%, 87.23%] | **74.47%** [61.70%, 85.11%] | **+2.13%** [-2.13%, +7.45%] |
| `crack500` | 17 | 0 | **88.24%** [70.59%, 100.00%] | **82.35%** [64.71%, 100.00%] | **+5.88%** [+0.00%, +17.65%] |
| `road_crack` | 16 | 0 | **93.75%** [81.25%, 100.00%] | **93.75%** [81.25%, 100.00%] | **+0.00%** [+0.00%, +0.00%] |
| **Cracks Combined** (`crack500` + `road_crack`) | 33 | 0 | **90.91%** [78.79%, 100.00%] | **87.88%** [75.76%, 96.97%] | **+3.03%** [+0.00%, +9.09%] |
| `pothole` | 14 | 2 | **42.86%** [17.86%, 67.86%] | **42.86%** [14.29%, 71.43%] | **+0.00%** [-10.71%, +10.71%] |

**Plain Summary**:
- **Overall**: No detectable difference vs the damage area baseline (+2.13 percentage points, 95% CI: [-2.13%, +7.45%], which includes 0).
- **Crack Distress**: Strong alignment with human judgment across both crack sources: **30 / 33 pairs agreed** (**90.91%**, 95% CI: [78.79%, 100.00%]), performing identically to the ground-truth sPCI reference.
- **Pothole Distress**: Poor alignment with human judgment: **6.0 / 14 pairs agreed** (**42.86%**, 95% CI: [17.86%, 67.86%]). Crucially, the 95% confidence interval spans 50.0% (random chance), meaning sPCI performance on pothole pairs cannot be distinguished from guessing.

---

## 4. Full Test Split Benchmark Metrics

Computed on the full test sets ($N = 200$) across all three benchmarks:

| Benchmark Dataset | Split Size ($N$) | Ground-Truth Channels Available | Spearman $\rho$ (GT vs. Pred sPCI) | Mean Absolute Error (MAE) |
| :--- | :---: | :--- | :---: | :---: |
| **CRACK500** | 72 | Crack | **0.9038** | **6.81 points** |
| **Road Crack** | 68 | Crack | **0.7547** | **11.80 points** |
| **Pothole** | 60 | Pothole only | **0.8123** | **10.28 points** |

---

## 5. Descriptive Error Analysis

A total of 12 pairs had agreement scores $< 1.0$ (10 full disagreements, 2 ties):

| Pair ID | Source | Human Label | Model sPCI A | Model sPCI B | Model Decision | $|\Delta\text{sPCI}|$ | Score | Description / Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `pair_004` | `crack500` | **A** | 65.24 | 62.05 | **B** | 3.19 | 0.0 | Close crack coverage; model penalized slightly longer skeleton in B. |
| `pair_007` | `crack500` | **A** | 59.27 | 44.79 | **B** | 14.48 | 0.0 | Image B had multiple branches triggering network factor multiplier ($m > 1.2$). |
| `pair_037` | `road_crack` | **B** | 37.14 | 40.97 | **A** | 3.83 | 0.0 | Close score difference (3.83 pts); image A had slightly thicker crack width. |
| `pair_042` | `pothole` | **A** | 37.84 | 23.31 | **B** | 14.53 | 0.0 | 2D area in B exceeded A, but human rater selected A as worse. |
| `pair_043` | `pothole` | **A** | 15.59 | 2.83 | **B** | 12.76 | 0.0 | Pothole footprint in B larger; human rater selected A as worse. |
| `pair_044` | `pothole` | **A** | 0.00 | 0.00 | **Tie** | 0.00 | 0.5 | Both images exceeded ~10% pothole area, saturating deduct curve to 0.00 sPCI. |
| `pair_045` | `pothole` | **A** | 73.03 | 53.65 | **B** | 19.38 | 0.0 | Pothole area in B larger than A; human rater selected A as worse. |
| `pair_048` | `pothole` | **B** | 0.00 | 25.08 | **A** | 25.08 | 0.0 | Large surface patch in A saturated sPCI to 0.00; human rater picked B. |
| `pair_049` | `pothole` | **A** | 53.09 | 0.00 | **B** | 53.09 | 0.0 | Image B had substantial surface hole saturating sPCI to 0.00; human rater picked A. |
| `pair_053` | `pothole` | **A** | 28.94 | 0.00 | **B** | 28.94 | 0.0 | Large footprint in B saturated sPCI to 0.00; human rater picked A. |
| `pair_054` | `pothole` | **B** | 2.20 | 22.71 | **A** | 20.51 | 0.0 | Image A had larger 2D area; human rater selected B as worse. |
| `pair_058` | `pothole` | **A** | 0.00 | 0.00 | **Tie** | 0.00 | 0.5 | Both images saturated deduct curve to 0.00 sPCI (exact tie). |

A visual grid of representative disagreeing pairs is saved at:
- File path: `docs/figures/human_agreement_disagreements.png`

### Qualitative Observations on Disagreements
1. **Deduct Curve Saturation on Potholes**: Deduct curves are power functions:
   $$\text{DV} = \min(100.0, a \cdot q^b)$$
   where for potholes $a \in [100, 160, 240]$ and $b = 0.60$. Because $a \ge 100$, when aggregate pothole area fraction $q_{\text{pot}} \ge 0.10$ (10%), $\text{DV} \ge 100 \cdot (0.1)^{0.6} \approx 25.1$ even at low severity, and at high severity reaches 100.0, rapidly driving sPCI toward $0.00$. When two severely deteriorated roads were paired (e.g. `pair_044`, `pair_058`), both saturated to 0.00 sPCI, making distinction impossible under sPCI v1.
2. **Missing 3D Depth Cues (Untested Hypothesis)**: 2D monocular segmentation measures only projected surface area ($q_{\text{pot}}$). One plausible but untested hypothesis is that human raters may heavily weight vertical cavity depth, shadowing, and sharp vertical edges, which cannot be measured from 2D RGB imagery alone.
3. **Alternative Hypothesis (Camera Perspective & Scale Ambiguity)**: Because there is no metric pixel scale (ground-sampling distance, GSD) across diverse uncalibrated smartphone and dashboard images, image-relative pothole size (fraction of image area) is not physically comparable between photos taken with different camera distances, heights, or angles. A small pothole photographed close to the ground can occupy 25% of the frame, whereas a massive crater photographed from a vehicle dashboard 5 meters away may occupy only 5% of the frame.
4. **High Fidelity on Crack Distress**: In contrast to potholes, crack rankings demonstrated excellent alignment (88.2% on CRACK500, 93.8% on Road Crack; 30/33 combined = 90.91%), validating the effectiveness of normalized skeleton length, width severity thresholding, and topological network multipliers.

---

## 6. Threats to Validity and Limitations

1. **Sample Size**: The human validation set is modest ($N = 60$ pairs total, $N = 47$ non-tie/non-same evaluations). While sufficient for exploratory verification, statistical power to detect small performance deltas ($\le 5\%$) is limited.
2. **Single Human Labeler**: All 60 pairs were annotated by a single rater. Inter-rater agreement (Fleiss' $\kappa$) could not be assessed.
3. **Within-Source Pairing**: Pairs were sampled strictly within benchmark sources (CRACK500 vs. CRACK500, Road Crack vs. Road Crack, Pothole vs. Pothole). Cross-domain cross-distress pairs (e.g. cracks vs. potholes) were not evaluated.
4. **No Metric Scale**: The system operates on normalized 512 px image fields without ground-sampling distance (GSD) calibration.
