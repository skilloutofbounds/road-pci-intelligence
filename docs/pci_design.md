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
     - **Low**: $w \le T_{\text{low}}$ ($33^{\text{rd}}$ percentile over crack-bearing train+val GT, $\approx 5.0$ px)
     - **Medium**: $T_{\text{low}} < w \le T_{\text{high}}$ ($80^{\text{th}}$ percentile over crack-bearing train+val GT, $\approx 9.0$ px)
     - **High**: $w > T_{\text{high}}$ ($> 9.0$ px)
   - Extent per severity tier:
     $$q_{\text{crack}, s} = \frac{N_{\text{skel}, s}}{N_{\text{skel}}} \times \frac{\text{crack\_length\_px}}{512.0} \quad \text{(units: 512 px short sides)}$$

2. **Network Multiplier ($m$)**:
   - Captures severe structural degradation from fatigue/alligator cracking:
     $$m = 1.0 + \beta \cdot \text{clip}\left(\frac{\text{network\_index}}{n_{\text{ref}}}, 0.0, 1.0\right)$$
     where $\beta = 0.5$ (default) and $n_{\text{ref}}$ is the $90^{\text{th}}$ percentile of `network_index` over crack-bearing train+val GT ($\approx 1.933$).

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

**Corrected Deduct Value (CDV) & sPCI**:
$$\text{CDV} = 100.0 \cdot \left(1.0 - \prod_{i} \left(1.0 - \frac{\text{DV}_i}{100.0}\right)\right)$$
$$\text{sPCI} = 100.0 - \text{CDV} = 100.0 \cdot \prod_{i} \left(1.0 - \frac{\text{DV}_i}{100.0}\right)$$

### 3.3 Condition Rating Bands (ASTM D6433-20)

Pavement condition ratings follow the standard ASTM D6433 rating scales:
- **Good**: $85 \le \text{sPCI} \le 100$
- **Satisfactory**: $70 \le \text{sPCI} < 85$
- **Fair**: $55 \le \text{sPCI} < 70$
- **Poor**: $40 \le \text{sPCI} < 55$
- **Very Poor**: $25 \le \text{sPCI} < 40$
- **Serious**: $10 \le \text{sPCI} < 25$
- **Failed**: $0 \le \text{sPCI} < 10$

*Citation*: ASTM D6433-20, "Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys", ASTM International, West Conshohocken, PA, 2020, Section 4 & Table 1.

---

## 4. Architectural Assumptions & Limitations

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

## 5. Parameter Freeze Declaration

As of commit `spci-frozen`, the following parameters are strictly frozen:
- Feature extraction short-side dimension: 512 px
- Noise filters: crack $< 30$ px, pothole $< 50$ px, enclosed cells $< 100$ px
- Skeleton spur pruning threshold: $< 12$ px
- sPCI parameters in `configs/spci_params.yaml`:
  - `crack_severity`: Low $\le 5.0$ px, High $> 9.0$ px
  - `network_factor`: $\beta = 0.5, n_{\text{ref}} = 1.933$
  - `pothole_severity`: Low $< 0.08$, High $> 0.20$
  - `deduct_curves`: crack $(a \in [5, 12, 25], b = 0.75)$, pothole $(a \in [100, 160, 240], b = 0.60)$
  - Combination formula: Multiplicative product CDV.
