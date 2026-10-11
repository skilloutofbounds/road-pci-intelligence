# Pavement Distress Feature Extraction & PCI Design

## 1. Overview and Objectives

The objective of the pavement feature extraction module (`src/pavement_features.py`) is to bridge raw pixel-level semantic segmentation masks (`crack` and `pothole` channels) and civil engineering pavement condition metrics (simplified Pavement Condition Index, PCI).

Standard pavement condition rating systems (e.g., ASTM D6433) evaluate distresses based on three orthogonal dimensions:
1. **Distress Type**: Longitudinal cracking, transverse cracking, alligator (fatigue) cracking, and potholes.
2. **Distress Extent (Severity / Quantity)**: Linear length of cracking (feet or meters) and surface area of potholes (square feet or square meters).
3. **Distress Severity**: Crack width (hairline vs. moderate vs. severe) and pothole depth/diameter.

Because roadway imagery sourced across heterogeneous public benchmarks lacks metric ground-sampling distance (GSD) calibration, the feature pipeline standardizes every mask to a fixed resolution and extracts scale-invariant morphological and topological features.

---

## 2. Feature Definitions and Units

Every mask is preprocessed by rescaling its shorter side to **512 pixels** (nearest-neighbor interpolation followed by binary thresholding $>0$). Crack connected components smaller than **30 pixels** and pothole components smaller than **50 pixels** are removed as sensor/label noise.

### 2.1 Crack Distress Features

| Feature Name | Type | Mathematical Definition | Physical Interpretation & Units |
| :--- | :--- | :--- | :--- |
| `crack_area_frac` | float | $\frac{\sum \text{Crack Pixels}}{H_{512} \times W_{512}}$ | Dimensionless fraction of roadway surface covered by crack distress. |
| `crack_length_px` | float | $\sum_{(u,v) \in E} w(u,v)$ where $w=1.0$ (cardinal) or $\sqrt{2}$ (diagonal) | Skeleton centerline length in normalized pixels using 8-connected steps. |
| `crack_length_density` | float | $\frac{\text{crack\_length\_px}}{512 \times 512} \times 512 = \frac{\text{crack\_length\_px}}{512.0}$ | Normalized crack linear density: pixels of crack length per 512 px image dimension. |
| `mean_width_px` | float | $\frac{\text{crack\_area\_px}}{\text{crack\_length\_px}}$ (if length $> 0$, else $0.0$) | Mean physical crack width in normalized pixels. |
| `relative_width` | float | $\frac{\text{mean\_width\_px}}{512.0}$ | Dimensionless scale-invariant relative crack width normalized to image scale. |
| `max_width_px` | float | $\max_{(x,y) \in \text{skel}} 2 \cdot d(x, y)$ | Maximum local crack diameter from Euclidean distance transform to background. |
| `width_p90_px` | float | 90th percentile of $2 \cdot d(x, y)$ along skeleton | 90th percentile local width, robust to localized blobbing or intersections. |
| `n_crack_components` | int | Count of 8-connected crack components $\ge 30$ px | Number of distinct, isolated crack distress instances. |
| `n_junctions` | int | Count of merged clusters with $\ge 3$ 8-neighbors | Number of crack branching nodes (intersection points). |
| `cells` | int | Count of enclosed background regions $\ge 100$ px inside closed crack mask | Number of closed pavement blocks / polygons formed by interconnecting cracks. |
| `network_index` | float | $\frac{100 \times (\text{n\_junctions} + \text{cells})}{\text{crack\_length\_px}}$ (if length $> 0$, else $0.0$) | Topological meshing density: $(junctions + cells)$ per 100 px of crack length. |

### 2.2 Pothole Distress Features

| Feature Name | Type | Mathematical Definition | Physical Interpretation & Units |
| :--- | :--- | :--- | :--- |
| `pothole_count` | int | Count of 8-connected pothole components $\ge 50$ px | Number of distinct pothole defects. |
| `pothole_area_frac` | float | $\frac{\sum_{\text{comp} \ge 50} \text{Area}}{H_{512} \times W_{512}}$ | Dimensionless surface fraction of pavement occupied by potholes. |
| `largest_pothole_frac` | float | $\frac{\max_{\text{comp} \ge 50} \text{Area}}{H_{512} \times W_{512}}$ | Surface fraction of the single most severe pothole defect. |

---

## 3. Core Architectural Assumptions

1. **Absence of Absolute Metric Scale**:
   - The benchmarks (`CRACK500`, `road-crack-dataset`, `pothole-image-segmentation-dataset`) contain photos taken at variable camera heights, angles, and focal lengths without ground-sampling distance (GSD) calibration.
   - Normalizing each image so that its shorter side is 512 px establishes an empirical baseline where all spatial quantities (densities, widths, lengths) are strictly comparable relative to the field of view.
2. **Continuous Topological Collapse of Crack Categories**:
   - Rather than forcing rigid discrete multiclass classification into longitudinal vs. transverse vs. alligator cracking, we model crack morphology on a continuum:
     - **Isolated linear cracks** (longitudinal / transverse): characterized by $\text{cells} = 0$, $\text{n\_junctions} \le 1$, and $\text{network\_index} \approx 0$.
     - **Branched / Block cracking**: characterized by moderate $\text{n\_junctions}$ and small $\text{network\_index}$.
     - **Alligator (fatigue) cracking**: characterized by high $\text{n\_junctions}$, numerous enclosed $\text{cells} \ge 1$, and high $\text{network\_index}$.
3. **Strict Test Split Segregation**:
   - The test splits ($N = 200$ images across the three datasets) are completely isolated from feature extraction design, parameter tuning, and threshold selection.
   - Test split data is utilized solely to compile the blinded human labeling sheet (`labels/pairs_template.csv`) for independent external validation.

---

## 4. Planned Validation & Alignment

1. **Feature Extraction Unit Testing**:
   - Synthetic masks with known analytical answers are validated in `tests/test_features.py` (lines, plus-signs, grids, disks, speckles, empty masks, scale invariance).
2. **Ground-Truth vs. Prediction Spearman Rank Correlation**:
   - On the validation split ($N = 255$ images), features extracted from model predictions (ResNet-34 U-Net, Naive Loss, Flip-TTA) are compared against ground-truth features via Spearman rank correlation ($\rho$) to ensure high monotonic fidelity before PCI deduction curve formulation.
3. **Human Pairwise Preference Validation**:
   - Human ranking on 60 blinded test image pairs (`labels/pairs_template.csv`) will be compared against automated PCI score rankings to evaluate model alignment with professional inspection judgment.
