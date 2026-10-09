# Road PCI Intelligence — Data Decisions, Schemas & Leak-Proof Splitting

This document details the architectural data engineering decisions, annotation standardization rules, group-aware partitioning policies, masked-loss formulation, and known dataset limitations for the **Road PCI Intelligence** project.

---

## 1. Dataset Registry & Licensing

| Dataset Key | Dataset Source / URL | License (Kaggle Metadata / Verified) | Role in Project Pipeline |
| :--- | :--- | :--- | :--- |
| **`pothole`** | [`farzadnekouei/pothole-image-segmentation-dataset`](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset) | `CC0: Public Domain` | Pothole semantic segmentation channel ground truth. YOLOv8 normalized polygon vectors rasterized to binary masks. |
| **`road_crack`** | [`rukiyeaydn/road-crack-dataset`](https://www.kaggle.com/datasets/rukiyeaydn/road-crack-dataset) | `CC0: Public Domain` | Pavement crack semantic segmentation channel ground truth. Concrete pavement orthomosaic tiles; Class 1 mapped to crack channel. |
| **`crack500`** | [`sj26717/crack500`](https://www.kaggle.com/datasets/sj26717/crack500) | `CC0-1.0` *(metadata quoted)*<br>Subtitle: *not found*<br>Description: *not found*<br>userSpecifiedSources: *not found*<br>*(Benchmark: Temple University FPHB/CRACK500)* | Asphalt pavement crack semantic segmentation channel ground truth. Binary grayscale masks mapped to crack channel; official benchmark train/val/test splits preserved. |
| **`rdd2022es`** | [`juusos/rdd2022es`](https://www.kaggle.com/datasets/juusos/rdd2022es) | `CC-BY-NC-SA-4.0` *(metadata quoted)*<br>Subtitle: *"Severity Extended RDD2022"*<br>Description: *Detailed 8 damage categories across USA, India, Japan, Czech*<br>userSpecifiedSources: *Quoted Arya et al., CRDDC 2022, IEEE Big Data* | Multi-country bounding box benchmark & held-out detection evaluation. Kept in a separate manifest; **not** used for pixel-level segmenter loss. |

---

## 2. Standardized Mask Conversion Rules & Assumptions

All segmentation annotations are standardized into a multi-channel binary tensor representation:
- **Channel 0**: Crack (`0` = background, `1` = crack)
- **Channel 1**: Pothole (`0` = background, `1` = pothole)

### A. Road-Crack Dataset (`rukiyeaydn/road-crack-dataset`)
- **Raw Format**: 2D PNG masks with pixel values in `{0, 1, 2, 3}`.
- **Conversion Rule**:
  - Value `1` $\to$ `1` (Crack)
  - Value `2` $\to$ `0` (Background)
  - Value `3` $\to$ `0` (Background)
  - Value `0` $\to$ `0` (Background)
- **Recorded Assumption**:
  > *"Value-to-class mapping inferred from the Kaggle description order and mask geometry, not documented in the files."*
- **Empirical & Geometric Evidence**:
  - The Kaggle dataset description lists the classes as: 1. *Concrete Crack*, 2. *Construction Joints*, 3. *Drilling*. However, the repository contains an empty `README.md` (0 bytes) and no configuration files.
  - Geometric analysis across all 435 masks confirms:
    - **Value 1 (Crack)**: Irregular, branching, tortuous crack networks (72.4% intersect image boundaries).
    - **Value 2 (Joint)**: Perfectly straight, thick linear construction slab joints / saw cuts (94.8% intersect image boundaries).
    - **Value 3 (Drill)**: Isolated, small circular or semi-circular core-drilling extraction holes (0.0% boundary intersection).
  - Mapping Value 1 to crack and Values 2 & 3 to background ensures the model learns true structural pavement distress rather than engineered construction joints or intentional core samples.

### B. CRACK500 Dataset (`sj26717/crack500`)
- **Raw Format**: Binary grayscale PNG masks with values in `{0, 255}`.
- **Conversion Rule**: Any non-zero pixel ($>0$) $\to$ `1` (Crack); `0` $\to$ `0` (Background).

### C. Pothole Dataset (`farzadnekouei/pothole-image-segmentation-dataset`)
- **Raw Format**: YOLOv8 text annotations containing normalized polygon coordinates:
  $$\text{class\_id } x_1\ y_1\ x_2\ y_2\ \dots\ x_n\ y_n \quad (x_i, y_i \in [0, 1])$$
- **Conversion Rule**:
  - Denormalize vertex coordinates to target image dimensions: $(p_{x, i}, p_{y, i}) = (\text{round}(x_i \cdot W), \text{round}(y_i \cdot H))$.
  - Rasterize closed polygon contours into a 2D binary uint8 mask using `cv2.fillPoly`: Pothole $\to$ `1`, Background $\to$ `0`.

---

## 3. Leak-Proof, Group-Aware Splitting Policy

Naive random splitting causes severe data leakage in road distress datasets due to image tiling, multi-angle augmentation, and horizontal mirroring. We enforce group-aware partitioning across all sources:

| Source | Group Identifier (`group_id`) | Split Strategy | Partition Counts |
| :--- | :--- | :--- | :--- |
| **`pothole`** | Base scene prefix `pic-<N>` stripped of augmentation hash (e.g., `pic-282`). | Test = official unaugmented `valid/` directory.<br>Val = 15% of train group IDs (deterministic, seed 42).<br>Train = remaining 85% of train groups.<br>*(All 3 augmented variants of any group strictly share the same split).* | • **Train**: 612 images (204 groups)<br>• **Val**: 108 images (36 groups)<br>• **Test**: 60 images (60 groups)<br>• **Total**: 780 images (300 groups) |
| **`road_crack`** | Physical scene stem (`road_<N>` or `Production_1_ortho_merge`). | Scene-level group split targeting ~70% train / 15% val / 15% test by image count (seed 42). All tiles of a physical road segment stay together. | • **Train**: 294 images (15 scenes, 67.6%)<br>• **Val**: 71 images (4 scenes, 16.3%)<br>• **Test**: 70 images (3 scenes, 16.1%)<br>• **Total**: 435 images (22 scenes) |
| **`crack500`** | Base capture image stem (e.g., `20160222_165402`). | Preserves official benchmark split lists verbatim (`train.lst`, `val.lst`, `test.lst`). | • **Train**: 329 images<br>• **Val**: 70 images<br>• **Test**: 72 images<br>• **Total**: 471 images |
| **`rdd2022es`** | Filename stem with `xmirror_` prefix removed (e.g., `Czech_000000`). | 80% train / 10% val / 10% test by group ID (seed 42). Mirrored pairs (`Czech_...` and `xmirror_Czech_...`) strictly share the same split. | • **Train**: 20,480 images (10,240 groups)<br>• **Val**: 2,560 images (1,280 groups)<br>• **Test**: 2,560 images (1,280 groups)<br>• **Total**: 25,600 images (12,800 groups) |

---

## 4. Per-Source Masked-Loss Plan (Multi-Task Weak Supervision)

### The Partial Supervision Problem
In pavement distress inspection, no publicly available dataset provides simultaneous pixel-level annotations for both cracks and potholes:
- `road_crack` and `crack500` only annotate crack networks (potholes are unannotated or absent).
- `pothole` only annotates pothole boundaries (cracks are unannotated or absent).

If treated as full supervision, unannotated potholes in crack datasets would be penalized as false positives, and unannotated cracks in pothole datasets would be penalized as false positives, degrading multi-task convergence.

### Multi-Channel Masked Loss Formulation
Our multi-head segmentation architecture predicts two independent sigmoid logits:
$$\hat{\mathbf{Y}} \in \mathbb{R}^{H \times W \times 2} \quad \text{where } \hat{y}_{\text{crack}} = \sigma(z_0), \quad \hat{y}_{\text{pothole}} = \sigma(z_1)$$

Each manifest entry provides explicit binary supervision capability flags:
- `labels_crack` $\in \{0, 1\}$
- `labels_pothole` $\in \{0, 1\}$

The total training loss per sample is a masked combination of Binary Cross-Entropy (BCE) and Soft Dice loss:
$$\mathcal{L}_{\text{total}} = \mathbb{I}_{\text{labels\_crack}} \cdot \mathcal{L}_{\text{seg}}(y_{\text{crack}}, \hat{y}_{\text{crack}}) + \mathbb{I}_{\text{labels\_pothole}} \cdot \mathcal{L}_{\text{seg}}(y_{\text{pothole}}, \hat{y}_{\text{pothole}})$$

where:
$$\mathcal{L}_{\text{seg}}(y, \hat{y}) = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}(y, \hat{y})$$

- When training on `road_crack` or `crack500`: $\mathbb{I}_{\text{labels\_pothole}} = 0$. Zero gradients backpropagate through the pothole head.
- When training on `pothole`: $\mathbb{I}_{\text{labels\_crack}} = 0$. Zero gradients backpropagate through the crack head.
- At inference time, both heads run concurrently across any input road inspection frame, producing a unified multi-distress overlay.

---

## 5. Known Limitations & Mitigation Strategies

### 1. Concrete vs. Asphalt Domain Shift
- **Domain Gap**: `road_crack` is sourced from concrete slab pavements (high albedo, aggregate exposure, saw-cut joints), whereas `crack500` and `pothole` are predominantly asphalt (dark bitumen, fine bitumen aggregate, oil stains).
- **Impact**: Feature representations tuned purely to asphalt fail to segment cracks on concrete, and vice versa.
- **Mitigation**: Color jittering, random gamma transforms, grayscale normalization, and multi-source balanced batch sampling during training.

### 2. Lack of Physical Pixel Scale for PCI Computation
- **Gap**: Standard dashcam and handheld smartphone images lack calibrated camera intrinsics, LiDAR depth, and ground sampling distance ($\text{mm}/\text{pixel}$).
- **Impact**: ASTM D6433 Pavement Condition Index (PCI) requires physical engineering quantities (crack width in millimeters, crack length in linear meters, pothole area in square meters, and pothole depth in centimeters).
- **Mitigation**:
  1. We implement a reference physical calibration prior (e.g., standard lane width prior $\approx 3.65\text{ m}$ or standard road marking dimensions) to compute an estimated ground scale.
  2. For downstream simulation, distress severity is parameterized using relative fractional area, morphological skeleton length, and contour diameter priors until calibrated metric inputs are supplied.

### 3. Foreground Class Imbalance
- Cracks occupy approximately $1.5\% - 3.0\%$ of total image pixels.
- Potholes occupy approximately $2.0\% - 4.5\%$ of total image pixels.
- **Mitigation**: Using compound loss with Dice loss ($\text{Dice} = 1 - \frac{2|X \cap Y| + \epsilon}{|X| + |Y| + \epsilon}$) alongside focal/BCE loss to prevent background collapse.
