# Pavement Distress Feature Extraction & Simplified PCI (sPCI) Design

## 1. Overview and Objectives

The objective of the pavement feature extraction module (`src/pavement_features.py`) and simplified Pavement Condition Index engine (`src/spci.py`) is to bridge raw pixel-level semantic segmentation masks (`crack` and `pothole` channels) and civil engineering pavement condition metrics.

Standard pavement condition rating systems (e.g., ASTM D6433) evaluate distresses based on three orthogonal dimensions:
1. **Distress Type**: Longitudinal cracking, transverse cracking, alligator (fatigue) cracking, and potholes.
2. **Distress Extent (Quantity / Density)**: Linear length of cracking (normalized short sides) and surface area of potholes (area fraction).
3. **Distress Severity**: Crack width (Low, Medium, High) and pothole diameter (Low, Medium, High).

Because roadway imagery sourced across heterogeneous public benchmarks lacks metric ground-sampling distance (GSD) calibration, the feature pipeline standardizes every mask to a fixed resolution (shorter side = 512 px) and extracts scale-invariant morphological and topological features.

---

## 2. Feature Definitions and Units

Every mask is preprocessed by rescaling its shorter side to **512 pixels** (nearest-neighbor interpolation followed by binary thresholding $>0$). Crack connected components smaller than **30 pixels** and pothole components smaller than **50 pixels** are removed as sensor/label noise.

### 2.1 Crack Distress Features & Pruned Topology

To eliminate false spurs caused by rough boundary pixelation in raster segmentation masks, skeletons undergo **morphological pruning** (`prune_skeleton`): end-branches shorter than **12 pixels** attached to junction nodes are stripped before topological feature calculation.

| Feature Name | Type | Mathematical Definition | Physical Interpretation & Units |
| :--- | :--- | :--- | :--- |
| `crack_area_frac` | float | $\frac{\sum \text{Crack Pixels}}{H_{512} \times W_{512}}$ | Dimensionless fraction of roadway surface covered by crack distress. |
| `crack_length_px` | float | $\sum_{(u,v) \in E} w(u,v)$ where $w=1.0$ (cardinal) or $\sqrt{2}$ (diagonal) | Pruned skeleton centerline length in normalized pixels using 8-connected steps. |
| `crack_length_short_sides` | float | $\frac{\text{crack\_length\_px}}{512.0}$ | Normalized crack linear extent expressed in units of 512 px image short sides. *(Formerly aliased as `crack_length_density`)*. |
| `mean_width_px` | float | $\frac{\text{crack\_area\_px}}{\text{crack\_length\_px}}$ (if length $> 0$, else $0.0$) | Mean physical crack width in normalized pixels. |
| `relative_width` | float | $\frac{\text{mean\_width\_px}}{512.0}$ | Dimensionless scale-invariant relative crack width normalized to image scale. |
| `max_width_px` | float | $\max_{(x,y) \in \text{skel}} 2 \cdot d(x, y)$ | Maximum local crack diameter from Euclidean distance transform to background. |
| `width_p90_px` | float | 90th percentile of $2 \cdot d(x, y)$ along skeleton | 90th percentile local width, robust to localized blobbing or intersections. |
| `n_crack_components` | int | Count of 8-connected crack components $\ge 30$ px | Number of distinct, isolated crack distress instances. |
| `n_junctions` | int | Count of merged clusters with $\ge 3$ 8-neighbors on pruned skeleton | Number of crack branching nodes (intersection points) after removing spurs $< 12$ px. |
| `cells` | int | Count of enclosed background regions $\ge 100$ px inside closed crack mask | Number of closed pavement blocks / polygons formed by interconnecting cracks. |
| `network_index` | float | $\frac{100 \times (\text{n\_junctions} + \text{cells})}{\text{crack\_length\_px}}$ (if length $> 0$, else $0.0$) | Topological meshing density: $(junctions + cells)$ per 100 px of pruned crack length. |

### 2.2 Pothole Distress Features

| Feature Name | Type | Mathematical Definition | Physical Interpretation & Units |
| :--- | :--- | :--- | :--- |
| `pothole_count` | int | Count of 8-connected pothole components $\ge 50$ px | Number of distinct pothole defects. |
| `pothole_area_frac` | float | $\frac{\sum_{\text{comp} \ge 50} \text{Area}}{H_{512} \times W_{512}}$ | Dimensionless surface fraction of pavement occupied by potholes. |
| `largest_pothole_frac` | float | $\frac{\max_{\text{comp} \ge 50} \text{Area}}{H_{512} \times W_{512}}$ | Surface fraction of the single most severe pothole defect. |

---

## 3. Simplified PCI (sPCI) Formulation

The simplified Pavement Condition Index ($\text{sPCI} \in [0, 100]$) translates detected distress extent, severity, and topology into a unified index:

### 3.1 Distress Severity Categorization

1. **Crack Severity**:
   - Each skeleton pixel is classified by local width ($w = 2 \cdot \text{distanceTransform}$):
     - **Low**: $w \le T_{\text{low}}$ ($33^{\text{rd}}$ percentile over crack-bearing train+val GT = **4.394 px**)
     - **Medium**: $T_{\text{low}} < w \le T_{\text{high}}$ ($80^{\text{th}}$ percentile over crack-bearing train+val GT = **11.588 px**)
     - **High**: $w > T_{\text{high}}$ ($> 11.588$ px)
   - Extent per severity tier:
     $$q_{\text{crack}, s} = \frac{N_{\text{skel}, s}}{N_{\text{skel}}} \times \frac{\text{crack\_length\_px}}{512.0} \quad \text{(units: 512 px short sides)}$$

2. **Network Multiplier ($m$)**:
   - Captures severe structural degradation from fatigue/alligator cracking:
     $$m = 1.0 + \beta \cdot \text{clip}\left(\frac{\text{network\_index}}{n_{\text{ref}}}, 0.0, 1.0\right)$$
     where $\beta = 0.5$ and $n_{\text{ref}}$ is the $90^{\text{th}}$ percentile of `network_index` over crack-bearing train+val GT (**1.275**).

3. **Pothole Severity**:
   - Equivalent planar diameter normalized to 512 px short side ($d = 2 \sqrt{\text{Area}/\pi} / 512$):
     - **Low**: $d < 0.08$
     - **Medium**: $0.08 \le d \le 0.20$
     - **High**: $d > 0.20$
     *(Note: Pothole depth cannot be measured from single 2D monocular RGB images and is treated as unknown).*
   - Extent per severity tier:
     $$q_{\text{pot}, s} = \frac{\sum_{i \in s} \text{Area}_i}{512 \times 512} \quad \text{(surface area fraction, } 0 \text{ to } 1\text{)}$$

### 3.2 Deduct Curves & Combination

Deduct values per distress and severity tier are modeled as power-law functions:
$$\text{DV} = \min\left(100.0, a \cdot q^b\right)$$
For cracks, the network factor $m$ scales the deduct value: $\text{DV}_{\text{crack}, s} = \min(100.0, m \cdot a_s \cdot q_s^{b_s})$.

**Calibrated Defaults (`configs/spci_params.yaml`)**:
- Low crack: $a = 5.0, b = 0.75$ (1.0 short-side length $\rightarrow \text{DV} = 5.0$)
- Medium crack: $a = 12.0, b = 0.75$
- High crack: $a = 25.0, b = 0.75$
- Low pothole: $a = 100.0, b = 0.60$
- Medium pothole: $a = 160.0, b = 0.60$
- High pothole: $a = 240.0, b = 0.60$ (10% surface coverage $\rightarrow \text{DV} \approx 60.3$)

**Optional Noise Floor ($\tau$)**:
An optional parameter $\tau$ (in normalized short sides) sets crack deducts to 0 when total skeleton length is below $\tau$:
$$\text{If } \frac{\text{crack\_length\_px}}{512.0} < \tau \implies \text{DV}_{\text{crack}, \text{low}} = \text{DV}_{\text{crack}, \text{med}} = \text{DV}_{\text{crack}, \text{high}} = 0.0$$
Empirical evaluation across $\tau \in \{0, 0.1, 0.2, 0.3, 0.5\}$ on VAL showed that road_crack $\rho$ improved by at most $+0.0014$ (failing the $+0.03$ requirement). Hence, the noise floor remains **$\tau = 0.0$**.

**Corrected Deduct Value (CDV) & sPCI**:
$$\text{CDV} = 100.0 \cdot \left(1.0 - \prod_{i} \left(1.0 - \frac{\text{DV}_i}{100.0}\right)\right)$$
$$\text{sPCI} = 100.0 - \text{CDV} = 100.0 \cdot \prod_{i} \left(1.0 - \frac{\text{DV}_i}{100.0}\right)$$

### 3.3 Condition Rating Bands (ASTM D6433)

**Source**: ASTM D6433 / D6433M-20, *"Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys"*, ASTM International, West Conshohocken, PA, Section 4 (Summary of Practice) and Figure 1 (*Pavement Condition Index (PCI) and Condition Rating Scale*).

Quote of Figure 1 Rating Scale:
| PCI Integer Range | Condition Rating Category |
| :---: | :---: |
| **86 – 100** | **Good** |
| **71 – 85** | **Satisfactory** |
| **56 – 70** | **Fair** |
| **41 – 55** | **Poor** |
| **26 – 40** | **Very Poor** |
| **11 – 25** | **Serious** |
| **0 – 10** | **Failed** |

Per ASTM D6433 Section 10, continuous computed PCI values are rounded to the nearest integer ($0 \le \text{PCI} \le 100$), ensuring strictly non-overlapping integer rating categories without boundary ambiguity.

---

## 4. Sensitivity Analysis (VAL Split, 7 Groups Separated)

Evaluated over 50 random draws per parameter group with uniform perturbations in $[0.7, 1.3]$ on validation predicted masks ($N = 255$):

| Parameter Group | Min Spearman $\rho$ | 5th Percentile $\rho$ | Median Spearman $\rho$ |
| :--- | :---: | :---: | :---: |
| `crack a` | 0.9938 | 0.9946 | 0.9975 |
| `crack b` | 0.9975 | 0.9976 | 0.9989 |
| `pothole a` | 0.9822 | 0.9827 | 0.9963 |
| `pothole b` | **0.9521** | **0.9699** | **0.9854** |
| `severity thresholds` | 0.9836 | 0.9844 | 0.9933 |
| `beta` | 0.9994 | 0.9995 | 0.9998 |
| `network n_ref` | 0.9992 | 0.9992 | 0.9998 |

**Finding**: The ranking is most sensitive to **`pothole b`** (lowest min $\rho = 0.9521$, lowest median $\rho = 0.9854$), followed by `pothole a` and `severity thresholds`. Network topology parameters ($\beta, n_{\text{ref}}$) exhibit minimal sensitivity ($\rho > 0.999$).

---

## 5. Validation Distributions and Ground-Truth Comparison

### 5.1 Distribution by Dataset Source (VAL Split)

| Source | Mask Type | Min | P10 | Median | P90 | Max | sPCI $\ge 99$ | Condition Bands (Good / Satisfactory / Fair / Poor / Very Poor / Serious / Failed) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRACK500** ($N=70$) | Ground-Truth | 12.60 | 33.51 | 66.90 | 84.81 | 95.26 | 0 | 5 / 22 / 19 / 11 / 11 / 2 / 0 |
| | Predicted | 21.32 | 34.64 | 65.73 | 84.06 | 91.65 | 0 | 5 / 19 / 22 / 14 / 9 / 1 / 0 |
| **Road Crack** ($N=77$) | Ground-Truth | 72.99 | 79.98 | 100.00 | 100.00 | 100.00 | 39 | 60 / 17 / 0 / 0 / 0 / 0 / 0 |
| | Predicted | 55.60 | 62.42 | 86.98 | 98.95 | 100.00 | 8 | 42 / 20 / 15 / 0 / 0 / 0 / 0 |
| **Pothole** ($N=108$) | Ground-Truth | 0.00 | 0.00 | 41.31 | 80.18 | 86.01 | 0 | 2 / 26 / 13 / 14 / 14 / 18 / 21 |
| | Predicted | 0.00 | 0.00 | 35.07 | 78.17 | 86.06 | 0 | 1 / 21 / 9 / 12 / 23 / 22 / 20 |

### 5.2 Road Crack Subset Analysis
- **38 Images with $\ge 1$ GT Crack Pixel**:
  - Spearman $\rho = 0.8274$
  - Mean Absolute Error (MAE) = $3.00$ points
- **39 Empty-GT Images**:
  - Images with predicted $\text{sPCI} < 95$: **17 / 39 (43.6%)**
  - Distribution of predicted sPCI:
    - Min = 55.60, P10 = 58.30, Median = 96.00, P90 = 99.35, Max = 100.00

---

## 6. Noise Floor ($\tau$) Tuning Curve

Evaluated on VAL split for $\tau \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$ (units: normalized short sides):

| $\tau$ (Short Sides) | CRACK500 Spearman $\rho$ | CRACK500 MAE | Road Crack Spearman $\rho$ | Road Crack MAE |
| :---: | :---: | :---: | :---: | :---: |
| **0.0** | **0.9046** | **6.60** | **0.3554** | **8.79** |
| 0.1 | 0.9046 | 6.60 | 0.3554 | 8.77 |
| 0.2 | 0.9046 | 6.60 | 0.3568 | 8.63 |
| 0.3 | 0.9046 | 6.60 | 0.3143 | 8.63 |
| 0.5 | 0.9046 | 6.60 | 0.3012 | 8.69 |

**Decision Rule**: Adopt $\tau > 0$ only if road_crack $\rho$ improves by $\ge 0.03$ and CRACK500 $\rho$ drops by $\le 0.01$; otherwise keep $\tau = 0.0$.
**Result**: At $\tau = 0.2$, road_crack $\rho$ improved by only $+0.0014$ (below the $+0.03$ threshold). For $\tau \ge 0.3$, road_crack $\rho$ deteriorated. CRACK500 $\rho$ was unchanged. Therefore, **$\tau = 0.0$ is maintained**.

---

## 7. Architectural Assumptions & Limitations

1. **Relative Scale (No Metric GSD)**:
   - Images are gathered from diverse uncalibrated cameras, mobile phones, and vehicles without ground-sampling distance (GSD).
   - All spatial metrics (widths, lengths, areas) are computed relative to the 512 px normalized field of view.
2. **Pothole Depth Unknown**:
   - 2D monocular RGB cameras cannot measure defect depth or volume. Pothole severity is estimated solely from horizontal surface area and equivalent diameter.
3. **Approximated Deduct Curves**:
   - The deduct curves ($\text{DV} = \min(100, a \cdot q^b)$) are empirical monotonic approximations designed to provide continuous relative deterioration rankings.
   - **These curves are NOT calibrated to official ASTM deduct charts** and should not be used as legal engineering compliance records without field calibration.
4. **Morphological Continuum of Cracks**:
   - Rather than fragile multiclass labeling (longitudinal vs transverse vs alligator), cracks exist on a continuum:
     - Linear cracks: $\text{network\_index} \approx 0, \text{cells} = 0, m \approx 1.0$
     - Alligator/fatigue cracks: high $\text{network\_index}, \text{cells} \ge 1, m \rightarrow 1.5$.
5. **Within-Source Validation Scope**:
   - Validation is performed within-source on benchmark validation splits. Cross-domain out-of-distribution transfer (e.g., to RDD dashboard-camera views) exhibits perspective and lighting shifts.
6. **Strict Test Split Isolation**:
   - All parameter design, percentile calculations, and model freezes use **TRAIN and VAL splits only**.
   - The test split ($N = 200$) is used strictly for compiling the blinded human pairwise preference labeling sheet (`labels/pairs_template.csv`).

---

## 8. Parameter Freeze Audit & Final Tag Declaration

### Audit of Earlier `spci-frozen` Commit (`a044983`)
The earlier commit `a044983` tagged as `spci-frozen` was **premature**: it froze documentation while placeholder values remained in `configs/spci_params.yaml` ($T_{\text{low}} = 5.0, T_{\text{high}} = 9.0, n_{\text{ref}} = 1.933$). The empirical thresholds were computed during the subsequent Notebook 13 run on the full 1,486 train+val GT masks, after which `configs/spci_params.yaml` was updated in commit `a66335d` ($T_{\text{low}} = 4.394, T_{\text{high}} = 11.588, n_{\text{ref}} = 1.275$).

### Final Frozen Parameters (`spci-v1-frozen`)
As of git tag **`spci-v1-frozen`**, the following parameters and configurations are permanently frozen:
- **Feature normalization**: Shorter side = 512 px
- **Noise filters**: crack components $< 30$ px, pothole components $< 50$ px, enclosed cells $< 100$ px
- **Morphological pruning**: Spurs $< 12$ px stripped from junction nodes
- **Crack severity thresholds**:
  - $T_{\text{low}} = 4.394$ px (33rd percentile over crack-bearing train+val GT)
  - $T_{\text{high}} = 11.588$ px (80th percentile over crack-bearing train+val GT)
- **Network factor**:
  - $n_{\text{ref}} = 1.275$ (90th percentile over crack-bearing train+val GT)
  - $\beta = 0.5$
- **Pothole severity thresholds**:
  - $T_{\text{low}} = 0.08$ normalized diameter
  - $T_{\text{high}} = 0.20$ normalized diameter
- **Deduct curve coefficients**:
  - Crack Low: $a = 5.0, b = 0.75$
  - Crack Medium: $a = 12.0, b = 0.75$
  - Crack High: $a = 25.0, b = 0.75$
  - Pothole Low: $a = 100.0, b = 0.60$
  - Pothole Medium: $a = 160.0, b = 0.60$
  - Pothole High: $a = 240.0, b = 0.60$
- **Noise floor**: $\tau = 0.0$ short sides
- **Rating scale**: ASTM D6433 Figure 1 non-overlapping integer bands (Good: 86–100, Satisfactory: 71–85, Fair: 56–70, Poor: 41–55, Very Poor: 26–40, Serious: 11–25, Failed: 0–10).
