# Budget-Constrained Pavement Maintenance Simulation (Step 5C)

## 1. Executive Summary & Objective

This document specifies the formulation, mathematical models, empirical simulation results, error decomposition, and policy comparisons of the budget-constrained pavement maintenance simulation engine (`src/simulation/`).

The objective is to evaluate how an automated pavement inspection system (specifically, the frozen deep multi-task segmentation model + simplified Pavement Condition Index, **sPCI v1**) interacts with municipal asset management policies under annual capital budget constraints. The simulation models a synthetic road network over a 10-year planning horizon, comparing six distinct maintenance policies (P0 through P4, plus heuristic baseline P2t) under both estimated condition data and oracle ground-truth conditions.

> [!IMPORTANT]
> **EXPLICIT DISCLAIMER**: All financial costs, treatment costs per square meter, treatment effectiveness thresholds, and deterioration rate parameters in this module are **ILLUSTRATIVE BENCHMARK ASSUMPTIONS**. They are established for algorithmic benchmarking, sensitivity analysis, and comparative policy ranking. **No claims are made regarding real municipality accounting, actual tax dollar savings, or physical infrastructure life-cycle guarantees.**

---

## 2. Mathematical Formulation & Engine Mechanics

### 2.1 Synthetic Network Topology
The network consists of $N = 200$ synthetic roadway segments. Each segment $i \in \{0, \dots, N-1\}$ is characterized by:
- Segment ID: $\text{seg\_000}$ to $\text{seg\_199}$
- Length: $L_i \sim \mathcal{U}(100.0, 600.0)\text{ meters}$
- Width: $W_i \sim \mathcal{U}(6.0, 14.0)\text{ meters}$
- Surface Area: $A_i = L_i \times W_i\text{ m}^2$. Note that because segment geometry is instantiated per pool, the two pools have different total network surface areas ($668,192.9\text{ m}^2$ on `all` vs. $702,354.8\text{ m}^2$ on `cracks`) and therefore different annual budgets ($334,096.5$ units/yr on `all` vs. $351,177.4$ units/yr on `cracks`).
- Traffic Classification: Categorical variable drawn with municipal priority weights $w_i$:
  - Low traffic (50% probability, priority weight $w_i = 1.0$)
  - Medium traffic (35% probability, priority weight $w_i = 2.0$)
  - High traffic (15% probability, priority weight $w_i = 3.0$)
- Image Tile Sampling: Each segment samples $k = 6$ tiles with replacement from the tile score pool (`results/tile_scores.csv`):
  - **Estimated Condition**:
    $$\widehat{\text{PCI}}_{i, 0} = \frac{1}{k} \sum_{j=1}^{k} \text{predicted\_sPCI}_{i, j}$$
  - **True Condition**:
    $$\text{PCI}_{i, 0}^* = \frac{1}{k} \sum_{j=1}^{k} \text{gt\_sPCI}_{i, j}$$

### 2.2 Annual Deterioration Model
Without intervention, pavement condition deteriorates annually according to an accelerated log-normal deterioration curve based on current true PCI:
$$\Delta\text{PCI}_i = r_i \cdot f(\text{PCI}_i^*)$$
where:
- Base deterioration rate $r_i$ follows a log-normal distribution across segments:
  $$r_i \sim \text{Lognormal}(\mu = \ln(3.0), \sigma = 0.30)$$
  yielding a median annual degradation of $3.0\text{ PCI points/year}$.
- Acceleration factor $f(\text{PCI})$ models moisture penetration and structural loss below fair condition ($\text{PCI} < 70$):
  $$f(\text{PCI}) = \begin{cases} 1.0 + \frac{70.0 - \text{PCI}}{70.0}, & \text{if } \text{PCI} < 70.0 \\ 1.0, & \text{otherwise} \end{cases}$$
- Pavement condition is floored at 0:
  $$\text{PCI}_{t+1}^* = \max(0.0, \text{PCI}_t^* - \Delta\text{PCI}_i)$$

### 2.3 Treatment Options & Effectiveness Variants
Four treatment categories are modeled in relative illustrative cost units per square meter:

| Treatment Type ($T$) | Cost per $\text{m}^2$ ($c_T$) | Pavement Condition Update ($\text{PCI}_{\text{new}}^*$) | Expected Planning Gain ($\Delta \widehat{P}$) |
| :--- | :---: | :--- | :--- |
| **None** | $0.0$ | $\text{PCI}^*$ (no change) | $0.0$ |
| **Preventive (Crack Seal)** | $1.0$ | $\min(95.0, \text{PCI}^* + 15.0 \times \gamma)$ | $\max(0.0, \min(95.0, \widehat{\text{PCI}} + 15.0 \times \gamma) - \widehat{\text{PCI}})$ |
| **Rehabilitation** | $c_{\text{rehab}} \in \{3, 5, 8\}$ | $\max(\text{PCI}^*, u_i), \quad u_i \sim \mathcal{U}(85.0, 95.0)$ | $\max(0.0, 90.0 - \widehat{\text{PCI}})$ |
| **Reconstruction** | $12.0$ | $100.0$ | $\max(0.0, 100.0 - \widehat{\text{PCI}})$ |

**Preventive Effectiveness Factor ($\gamma$)**:
1. **Cliff Variant (Standard)**: $\gamma = 1.0$ if $\text{PCI} \ge 55.0$; $\gamma = 0.0$ if $\text{PCI} < 55.0$.
2. **Smooth Variant**: Continuous linear scaling:
   $$\gamma = \text{clip}\left(\frac{\text{PCI} - 40.0}{30.0}, 0.0, 1.0\right)$$
   providing full effect (+15 PCI) at $\ge 70.0$, half effect (+7.5 PCI) at $55.0$, and zero effect at $\le 40.0$.

> **Rule of Monotonicity**: A treatment never degrades pavement condition ($\text{PCI}_{\text{new}}^* \ge \text{PCI}^*$).

### 2.4 Budget Formulations & Real Spend Accounting
The simulation models two alternative annual budget specifications:
1. **Grid 1: Budget Scales with Rehab Cost** (Relative capital allocation):
   $$\text{Budget}_{\text{annual}} = c_{\text{rehab}} \times \left(\sum_{i=0}^{N-1} A_i\right) \times \frac{X}{100}$$
2. **Grid 2: Fixed Absolute Budget** (Decoupled capital allocation):
   $$\text{Budget}_{\text{annual}} = 5.0 \times \left(\sum_{i=0}^{N-1} A_i\right) \times \frac{X}{100}$$
   where $5.0$ is the reference rehabilitation unit cost, keeping the total dollar budget constant as $c_{\text{rehab}}$ varies.

**Real Spend Tracking**:
Annual expenditure is tracked strictly from applied interventions:
$$\text{Spent}_t = \sum_{i=0}^{N-1} c_{T_{i, t}} \times A_i \le \text{Budget}_{\text{annual}}$$
Unspent budget does **not** roll over to subsequent fiscal years.

### 2.5 Dual Evaluation Metrics: Area-Weighted and Traffic-Weighted
Because municipal stakeholders care both about physical pavement preservation and road user impact, all condition-years are reported in both metrics:
- **Area-Weighted Network Condition**:
  $$\text{PCI}_{\text{aw}, t} = \frac{\sum_{i=0}^{N-1} \text{PCI}_{i, t}^* \cdot A_i}{\sum_{i=0}^{N-1} A_i}, \quad \text{Condition-Years}_{\text{aw}} = \sum_{t=1}^{10} \text{PCI}_{\text{aw}, t}$$
- **Traffic-Weighted Network Condition**:
  $$\text{PCI}_{\text{tw}, t} = \frac{\sum_{i=0}^{N-1} \text{PCI}_{i, t}^* \cdot A_i \cdot w_i}{\sum_{i=0}^{N-1} A_i \cdot w_i}, \quad \text{Condition-Years}_{\text{tw}} = \sum_{t=1}^{10} \text{PCI}_{\text{tw}, t}$$

### 2.6 Maintenance Allocation Policies (P0 – P4, plus P2t)
In each year $t$, policies evaluate candidate actions based on observed condition ($\widehat{\text{PCI}}$ in deployment; $\text{PCI}^*$ in oracle mode):
1. **P0 (Do Nothing)**: Zero maintenance interventions; passive deterioration only. Total spend = 0.
2. **P1 (Reactive / Worst-First Severe)**:
   - Intervenes only on segments in poor condition ($\widehat{\text{PCI}} < 40.0$).
   - If $\widehat{\text{PCI}} < 15.0$: assign Reconstruction ($c = 12$).
   - If $15.0 \le \widehat{\text{PCI}} < 40.0$: assign Rehabilitation ($c = c_{\text{rehab}}$).
   - Ranked strictly worst-first (lowest $\widehat{\text{PCI}}$ first) until annual budget is exhausted.
3. **P2 (Worst-First by Condition Band)**:
   - All segments sorted worst-first (ascending $\widehat{\text{PCI}}$).
   - ASTM D6433 condition bands dictate treatment:
     - Failed / Serious ($\le 25$): Reconstruction
     - Very Poor / Poor ($26 - 55$): Rehabilitation
     - Fair ($56 - 70$): Preventive
     - Satisfactory / Good ($> 70$): None
   - Funded sequentially until annual budget is exhausted.
4. **P2t (Traffic-Aware Worst-First)**:
   - Heuristic baseline assigning the exact same ASTM D6433 condition-band treatments as P2.
   - However, candidate segments are prioritized in descending order of urgency score:
     $$\text{Urgency}_i = (100.0 - \widehat{\text{PCI}}_i) \times w_i$$
   - Funded sequentially until annual budget is exhausted.
5. **P3 (Greedy Benefit/Cost Heuristic)**:
   - For all segments $i$ and eligible treatments $T$:
     $$\text{Benefit}_{i, T} = \Delta \widehat{P}_{i, T} \times A_i \times w_i, \quad \text{Cost}_{i, T} = c_T \times A_i, \quad \text{Ratio}_{i, T} = \frac{\text{Benefit}_{i, T}}{\text{Cost}_{i, T}}$$
   - Greedily selects candidate with highest benefit/cost ratio, at most one treatment per segment per year, until budget is exhausted.
6. **P4 (Exact 0/1 Integer Linear Program)**:
   - Solves the exact single-year optimum knapsack allocation:
     $$\max_{x_{i, T} \in \{0, 1\}} \sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Benefit}_{i, T} \cdot x_{i, T}$$
     subject to:
     $$\sum_{T \in \mathcal{T}} x_{i, T} = 1 \quad \forall i, \quad \sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Cost}_{i, T} \cdot x_{i, T} \le \text{Budget}_{\text{annual}}$$
   - Solved via `scipy.optimize.milp` using the exact HiGHS branch-and-cut simplex backend.

---

## 3. Error Structure of the Estimates

To ground the simulation behavior, we report the empirical error structure of the vision model's sPCI estimates ($\text{Predicted sPCI} - \text{GT sPCI}$) across both tile pools and individual data sources.

### 3.1 Per-Tile Error Metrics
*Evaluated on individual image tiles ($N$ tiles):*

| Group | $N$ | Mean Signed Error | MAE | Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| **pool: all** | 455 | -3.1080 | 9.2768 | 15.4001 |
| **pool: cracks** | 287 | -3.5342 | 8.4718 | 13.0174 |
| **source: crack500** | 142 | +0.5449 | 6.7061 | 8.4796 |
| **source: road_crack** | 145 | -7.5289 | 10.2010 | 15.2863 |
| **source: pothole** | 168 | -2.3799 | 10.6518 | 18.8067 |

### 3.2 Segment Means Error Metrics ($k = 6$ Tiles per Segment)
*Evaluated across $N = 200$ synthetic roadway segments formed by averaging $k=6$ tiles sampled with replacement:*

| Group | $N$ Segments | Mean Signed Error | MAE | Std Dev (Empirical) | Std Dev (CLT Theoretical: $\sigma/\sqrt{6}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **pool: all** | 200 | -2.9493 | 4.8356 | 5.5699 | 6.2870 |
| **pool: cracks** | 200 | -3.3907 | 4.7573 | 5.3560 | 5.3143 |
| **source: crack500** | 200 | +1.1518 | 2.8673 | 3.4311 | 3.4618 |
| **source: road_crack** | 200 | -8.2588 | 8.7019 | 6.5123 | 6.2406 |
| **source: pothole** | 200 | -1.9974 | 5.8503 | 8.0079 | 7.6778 |

**Key Observations on Error Structure**:
1. In both pools, the model has a slight negative bias (underestimating sPCI by $-3.11$ points on `all`, and $-3.53$ points on `cracks`).
2. Source-level breakdown shows distinct behavior: CRACK500 has near-zero bias ($+0.54$) and low variance ($\sigma = 8.48$), whereas Road Crack has a significant negative bias ($-7.53$) and Pothole has high variance ($\sigma = 18.81$).
3. Segment-level aggregation ($k=6$) reduces standard deviation by approximately $1/\sqrt{6}$, matching Central Limit Theorem predictions closely (e.g., $5.36$ empirical vs $5.31$ theoretical on `cracks`).

---

## 4. Base Scenario Results with P2t Baseline

*Annual Budget: 10% of network rehab ($334,096.5$ units/yr on `all`, $351,192.5$ on `cracks`), Rehab Ratio: 5.0, $N = 200$ Monte Carlo runs (seeds 1000–1199):*

### 4.1 Base Table — Pool 'all' ($N = 200$ Runs)

| Policy | AW CY [95% CI] | TW CY [95% CI] | Real Spend [95% CI] | Budget Util (%) | Eff AW (CY/1k) [95% CI] | Eff TW (CY/1k) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 361.64 [361.34, 361.94] | 357.01 [356.69, 357.33] | 0 [0, 0] | 0.00% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **P1: Reactive** | 642.80 [642.28, 643.31] | 644.01 [643.48, 644.53] | 3,068,687 [3,063,713, 3,073,385] | 91.85% | 0.0916 [0.0914, 0.0919] | 0.0935 [0.0933, 0.0938] |
| **P2: Worst-First by Band** | 660.50 [660.22, 660.77] | 661.66 [661.33, 661.98] | 3,337,775 [3,337,667, 3,337,878] | 99.90% | 0.0895 [0.0894, 0.0896] | 0.0913 [0.0912, 0.0914] |
| **P2t: Traffic-Aware Worst-First** | 626.94 [626.62, 627.25] | 672.45 [672.16, 672.73] | 3,336,948 [3,336,839, 3,337,055] | 99.88% | 0.0795 [0.0794, 0.0796] | 0.0945 [0.0944, 0.0946] |
| **P3: Greedy Benefit/Cost** | 655.82 [655.60, 656.04] | 730.69 [730.52, 730.86] | 3,337,095 [3,336,986, 3,337,203] | 99.88% | 0.0882 [0.0881, 0.0882] | 0.112 [0.112, 0.112] |
| **P4: Integer Programming** | 656.77 [656.54, 656.99] | 731.54 [731.37, 731.70] | 3,339,967 [3,339,943, 3,339,990] | 99.97% | 0.0884 [0.0883, 0.0884] | 0.112 [0.112, 0.112] |

#### Paired Differences on Pool 'all' ($N = 200$ Runs):
- **P2t minus P2**:
  - Area-Weighted: $-33.56\text{ CY}$ [95% CI: $-33.88, -33.23$]
  - Traffic-Weighted: $+10.79\text{ CY}$ [95% CI: $+10.43, +11.14$]
- **P2t minus P3**:
  - Area-Weighted: $-28.88\text{ CY}$ [95% CI: $-29.21, -28.54$]
  - Traffic-Weighted: $-58.24\text{ CY}$ [95% CI: $-58.53, -57.95$]
- **P2t minus P4**:
  - Area-Weighted: $-29.82\text{ CY}$ [95% CI: $-30.16, -29.49$]
  - Traffic-Weighted: $-59.09\text{ CY}$ [95% CI: $-59.38, -58.79$]

---

### 4.2 Base Table — Pool 'cracks' ($N = 200$ Runs)

*Note on Budget Accounting: In the simulation execution, the annual budget for the `cracks` network (surface area = $702,354.79\text{ m}^2$) is $351,177.39\text{ units/yr}$ ($3,511,774$ over 10 years). Against this pool-specific budget, P1 spend of $1,629,136$ represents $46.39\%$ budget utilization. If evaluated against the baseline `all`-pool reference budget of $3,340,965$ ($334,096.5\text{ units/yr}$), $1,629,136$ represents $48.76\%$.*

| Policy | AW CY [95% CI] | TW CY [95% CI] | Real Spend [95% CI] | Budget Util (% of Pool Budget) | Eff AW (CY/1k) [95% CI] | Eff TW (CY/1k) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 545.44 [545.18, 545.70] | 548.37 [548.08, 548.66] | 0 [0, 0] | 0.00% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **P1: Reactive** | 618.69 [618.36, 619.01] | 620.41 [620.08, 620.74] | 1,629,136 [1,623,010, 1,635,307] | 46.39% (48.76% of 3.34M) | 0.0450 [0.0448, 0.0451] | 0.0442 [0.0440, 0.0444] |
| **P2: Worst-First by Band** | 774.99 [774.90, 775.09] | 776.28 [776.18, 776.39] | 1,738,324 [1,735,801, 1,740,889] | 49.50% | 0.132 [0.132, 0.132] | 0.131 [0.131, 0.131] |
| **P2t: Traffic-Aware Worst-First** | 774.99 [774.90, 775.09] | 776.51 [776.41, 776.62] | 1,738,493 [1,735,935, 1,741,079] | 49.50% | 0.132 [0.132, 0.132] | 0.131 [0.131, 0.131] |
| **P3: Greedy Benefit/Cost** | 810.89 [810.51, 811.28] | 845.26 [845.03, 845.49] | 3,507,799 [3,507,662, 3,507,935] | 99.89% | 0.0757 [0.0756, 0.0758] | 0.0846 [0.0845, 0.0847] |
| **P4: Integer Programming** | 810.87 [810.48, 811.27] | 845.21 [844.97, 845.44] | 3,511,596 [3,511,587, 3,511,606] | 99.99% | 0.0756 [0.0755, 0.0757] | 0.0845 [0.0844, 0.0846] |

#### Coincidence of P2 and P2t on Cracks Pool:
Note that P2 and P2t coincide on the `cracks` pool whenever the budget does not bind. Because initial pavement condition on `cracks` is higher and P2/P2t only exhaust $49.50\%$ of the annual budget ($1,738,324$), all eligible segments needing intervention are funded under both policies regardless of whether candidate segments are prioritized strictly worst-first (P2) or by traffic-weighted urgency score (P2t). Their performance difference is virtually zero (AW difference: $-0.00\text{ CY}$ [$-0.01, +0.00$], TW difference: $+0.23\text{ CY}$ [$+0.22, +0.23$]).

#### Paired Differences on Pool 'cracks' ($N = 200$ Runs):
- **P2t minus P2**:
  - Area-Weighted: $-0.00\text{ CY}$ [95% CI: $-0.01, +0.00$]
  - Traffic-Weighted: $+0.23\text{ CY}$ [95% CI: $+0.22, +0.23$]
- **P2t minus P3**:
  - Area-Weighted: $-35.90\text{ CY}$ [95% CI: $-36.28, -35.51$]
  - Traffic-Weighted: $-68.75\text{ CY}$ [95% CI: $-68.98, -68.51$]
- **P2t minus P4**:
  - Area-Weighted: $-35.88\text{ CY}$ [95% CI: $-36.27, -35.49$]
  - Traffic-Weighted: $-68.69\text{ CY}$ [95% CI: $-68.93, -68.46$]

---

## 5. Bias vs. Noise Decomposition Study

To isolate the effects of systematic calibration error (bias) versus random inspection variance (noise), we run controlled decompositions across all policies on both pools ($N = 100$ runs per setting, synchronized random seeds).

1. **Bias-Only Decomposition**:
   $$\widehat{\text{PCI}} = \text{True PCI} + b, \quad b \in \{-10.0, -5.0, 0.0, +5.0, +10.0\}$$
2. **Noise-Only Decomposition**:
   $$\widehat{\text{PCI}} = \text{True PCI} + \lambda \times \left(\text{Predicted sPCI} - \text{True sPCI} - \text{MeanSignedError}_{\text{pool}}\right), \quad \lambda \in \{0.0, 0.5, 1.0, 1.5, 2.0\}$$
   where subtracting $\text{MeanSignedError}_{\text{pool}}$ ensures the inspection noise is strictly zero-mean.
3. Regret is defined relative to the oracle ($b = 0.0$ or $\lambda = 0.0$):
   $$\text{Regret} = \text{Condition-Years}_{\text{oracle}} - \text{Condition-Years}_{\text{setting}}$$

### 5.1 Complete Table 5.1: Bias-Only Decomposition ($N = 100$ Runs)

| Pool | Bias $b$ | Policy | Regret AW [95% CI] | Regret TW [95% CI] | Mean AW CY | Mean TW CY |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `all` | -10.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | -10.0 | P1: Reactive | -41.52 [-42.30, -40.74] | -41.54 [-42.34, -40.76] | 662.63 | 663.51 |
| `all` | -10.0 | P2: Worst-First by Band | +21.19 [+19.41, +23.05] | +20.97 [+19.15, +22.84] | 644.28 | 645.31 |
| `all` | -10.0 | P2t: Traffic-Aware Worst-First | +17.17 [+15.92, +18.51] | -0.45 [-1.36, +0.52] | 618.05 | 673.13 |
| `all` | -10.0 | P3: Greedy Benefit/Cost | +115.02 [+114.54, +115.50] | +91.74 [+91.38, +92.11] | 603.87 | 682.73 |
| `all` | -10.0 | P4: Integer Programming | +115.08 [+114.61, +115.54] | +90.77 [+90.42, +91.14] | 605.22 | 684.60 |
| `all` | -5.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | -5.0 | P1: Reactive | -35.99 [-36.84, -35.15] | -36.00 [-36.86, -35.16] | 657.10 | 657.97 |
| `all` | -5.0 | P2: Worst-First by Band | +0.76 [+0.36, +1.16] | +0.92 [+0.48, +1.37] | 664.71 | 665.36 |
| `all` | -5.0 | P2t: Traffic-Aware Worst-First | +7.31 [+6.21, +8.54] | -3.40 [-4.17, -2.55] | 627.91 | 676.09 |
| `all` | -5.0 | P3: Greedy Benefit/Cost | +70.41 [+69.90, +70.92] | +46.33 [+45.99, +46.68] | 648.48 | 728.14 |
| `all` | -5.0 | P4: Integer Programming | +70.28 [+69.88, +70.70] | +45.71 [+45.44, +45.98] | 650.02 | 729.67 |
| `all` | +0.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | +0.0 | P1: Reactive | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 621.11 | 621.97 |
| `all` | +0.0 | P2: Worst-First by Band | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 665.47 | 666.28 |
| `all` | +0.0 | P2t: Traffic-Aware Worst-First | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 635.22 | 672.69 |
| `all` | +0.0 | P3: Greedy Benefit/Cost | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 718.89 | 774.48 |
| `all` | +0.0 | P4: Integer Programming | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 720.30 | 775.38 |
| `all` | +5.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | +5.0 | P1: Reactive | +40.38 [+39.18, +41.60] | +40.22 [+38.99, +41.45] | 580.73 | 581.75 |
| `all` | +5.0 | P2: Worst-First by Band | +0.95 [+0.73, +1.16] | +0.84 [+0.60, +1.09] | 664.52 | 665.43 |
| `all` | +5.0 | P2t: Traffic-Aware Worst-First | -14.06 [-16.16, -11.86] | -1.01 [-2.34, +0.41] | 649.28 | 673.70 |
| `all` | +5.0 | P3: Greedy Benefit/Cost | -13.50 [-13.96, -13.05] | +0.11 [-0.18, +0.40] | 732.39 | 774.36 |
| `all` | +5.0 | P4: Integer Programming | -12.08 [-12.53, -11.64] | +1.08 [+0.79, +1.36] | 732.38 | 774.30 |
| `all` | +10.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | +10.0 | P1: Reactive | +79.58 [+78.45, +80.73] | +79.35 [+78.26, +80.46] | 541.53 | 542.62 |
| `all` | +10.0 | P2: Worst-First by Band | +7.45 [+6.95, +7.96] | +7.38 [+6.85, +7.90] | 658.02 | 658.90 |
| `all` | +10.0 | P2t: Traffic-Aware Worst-First | -13.80 [-15.16, -12.32] | +13.11 [+12.17, +14.12] | 649.02 | 659.58 |
| `all` | +10.0 | P3: Greedy Benefit/Cost | +4.29 [+3.66, +4.93] | +22.05 [+21.61, +22.49] | 714.60 | 752.42 |
| `all` | +10.0 | P4: Integer Programming | +5.14 [+4.54, +5.74] | +22.89 [+22.46, +23.31] | 715.16 | 752.49 |
| `cracks` | -10.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | -10.0 | P1: Reactive | -52.66 [-53.27, -52.05] | -51.38 [-52.04, -50.73] | 662.45 | 663.71 |
| `cracks` | -10.0 | P2: Worst-First by Band | +13.98 [+13.64, +14.30] | +14.33 [+13.95, +14.69] | 732.96 | 733.04 |
| `cracks` | -10.0 | P2t: Traffic-Aware Worst-First | -54.11 [-54.45, -53.77] | -68.92 [-69.17, -68.66] | 801.05 | 816.28 |
| `cracks` | -10.0 | P3: Greedy Benefit/Cost | +149.17 [+148.88, +149.46] | +93.20 [+93.01, +93.39] | 717.13 | 785.92 |
| `cracks` | -10.0 | P4: Integer Programming | +148.87 [+148.60, +149.13] | +92.00 [+91.82, +92.17] | 717.74 | 787.31 |
| `cracks` | -5.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | -5.0 | P1: Reactive | -25.26 [-25.74, -24.79] | -24.65 [-25.19, -24.11] | 635.06 | 636.98 |
| `cracks` | -5.0 | P2: Worst-First by Band | -43.57 [-43.73, -43.41] | -43.56 [-43.74, -43.39] | 790.51 | 790.92 |
| `cracks` | -5.0 | P2t: Traffic-Aware Worst-First | -43.49 [-43.67, -43.29] | -44.90 [-45.10, -44.70] | 790.42 | 792.26 |
| `cracks` | -5.0 | P3: Greedy Benefit/Cost | +109.59 [+109.00, +110.18] | +62.77 [+62.41, +63.12] | 756.70 | 816.35 |
| `cracks` | -5.0 | P4: Integer Programming | +109.06 [+108.34, +109.80] | +62.22 [+61.79, +62.65] | 757.54 | 817.09 |
| `cracks` | +0.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | +0.0 | P1: Reactive | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 609.80 | 612.33 |
| `cracks` | +0.0 | P2: Worst-First by Band | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 746.94 | 747.36 |
| `cracks` | +0.0 | P2t: Traffic-Aware Worst-First | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 746.94 | 747.36 |
| `cracks` | +0.0 | P3: Greedy Benefit/Cost | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 866.29 | 879.12 |
| `cracks` | +0.0 | P4: Integer Programming | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 866.61 | 879.31 |
| `cracks` | +5.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | +5.0 | P1: Reactive | +21.25 [+20.79, +21.73] | +21.05 [+20.55, +21.57] | 588.55 | 591.27 |
| `cracks` | +5.0 | P2: Worst-First by Band | +44.76 [+44.58, +44.95] | +44.37 [+44.18, +44.56] | 702.17 | 703.00 |
| `cracks` | +5.0 | P2t: Traffic-Aware Worst-First | +44.76 [+44.58, +44.95] | +44.37 [+44.18, +44.56] | 702.17 | 703.00 |
| `cracks` | +5.0 | P3: Greedy Benefit/Cost | -7.09 [-7.29, -6.90] | -0.31 [-0.45, -0.18] | 873.38 | 879.43 |
| `cracks` | +5.0 | P4: Integer Programming | -5.15 [-5.37, -4.94] | +1.05 [+0.90, +1.19] | 871.76 | 878.26 |
| `cracks` | +10.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | +10.0 | P1: Reactive | +38.11 [+37.64, +38.60] | +37.94 [+37.42, +38.48] | 571.69 | 574.39 |
| `cracks` | +10.0 | P2: Worst-First by Band | +84.22 [+84.01, +84.42] | +83.41 [+83.17, +83.65] | 662.72 | 663.96 |
| `cracks` | +10.0 | P2t: Traffic-Aware Worst-First | +84.22 [+84.01, +84.42] | +83.41 [+83.17, +83.65] | 662.72 | 663.96 |
| `cracks` | +10.0 | P3: Greedy Benefit/Cost | +5.10 [+4.92, +5.27] | +13.43 [+13.30, +13.56] | 861.20 | 865.69 |
| `cracks` | +10.0 | P4: Integer Programming | +7.37 [+7.17, +7.56] | +15.38 [+15.22, +15.52] | 859.24 | 863.93 |

---

### 5.2 Complete Table 5.2: Noise-Only Decomposition ($N = 100$ Runs)

| Pool | $\lambda$ | Policy | Regret AW [95% CI] | Regret TW [95% CI] | Mean AW CY | Mean TW CY |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `all` | 0.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | 0.0 | P1: Reactive | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 621.11 | 621.97 |
| `all` | 0.0 | P2: Worst-First by Band | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 665.47 | 666.28 |
| `all` | 0.0 | P2t: Traffic-Aware Worst-First | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 635.22 | 672.69 |
| `all` | 0.0 | P3: Greedy Benefit/Cost | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 718.89 | 774.48 |
| `all` | 0.0 | P4: Integer Programming | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 720.30 | 775.38 |
| `all` | 0.5 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | 0.5 | P1: Reactive | +3.48 [+2.44, +4.53] | +2.86 [+1.82, +3.92] | 617.63 | 619.11 |
| `all` | 0.5 | P2: Worst-First by Band | +1.57 [+1.23, +1.90] | +1.35 [+0.99, +1.71] | 663.90 | 664.93 |
| `all` | 0.5 | P2t: Traffic-Aware Worst-First | +1.81 [+0.55, +3.05] | +1.19 [+0.32, +2.03] | 633.40 | 671.50 |
| `all` | 0.5 | P3: Greedy Benefit/Cost | +20.21 [+19.75, +20.66] | +14.06 [+13.79, +14.33] | 698.68 | 760.42 |
| `all` | 0.5 | P4: Integer Programming | +19.94 [+19.51, +20.36] | +13.93 [+13.68, +14.18] | 700.36 | 761.45 |
| `all` | 1.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | 1.0 | P1: Reactive | +2.13 [+1.24, +3.02] | +1.48 [+0.55, +2.41] | 618.98 | 620.50 |
| `all` | 1.0 | P2: Worst-First by Band | +4.96 [+4.56, +5.36] | +4.67 [+4.21, +5.13] | 660.50 | 661.61 |
| `all` | 1.0 | P2t: Traffic-Aware Worst-First | +8.35 [+7.17, +9.63] | +5.32 [+4.50, +6.21] | 626.87 | 667.37 |
| `all` | 1.0 | P3: Greedy Benefit/Cost | +40.35 [+39.92, +40.79] | +30.24 [+29.94, +30.54] | 678.54 | 744.24 |
| `all` | 1.0 | P4: Integer Programming | +40.96 [+40.49, +41.43] | +30.52 [+30.22, +30.81] | 679.34 | 744.86 |
| `all` | 1.5 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | 1.5 | P1: Reactive | +1.82 [+1.02, +2.62] | +1.46 [+0.63, +2.29] | 619.29 | 620.51 |
| `all` | 1.5 | P2: Worst-First by Band | +10.09 [+9.75, +10.45] | +9.99 [+9.58, +10.42] | 655.37 | 656.29 |
| `all` | 1.5 | P2t: Traffic-Aware Worst-First | +16.24 [+15.03, +17.56] | +10.04 [+9.20, +10.96] | 618.97 | 662.65 |
| `all` | 1.5 | P3: Greedy Benefit/Cost | +56.86 [+56.38, +57.33] | +45.76 [+45.43, +46.10] | 662.03 | 728.71 |
| `all` | 1.5 | P4: Integer Programming | +58.65 [+58.21, +59.09] | +46.86 [+46.57, +47.16] | 661.65 | 728.51 |
| `all` | 2.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 361.45 | 357.05 |
| `all` | 2.0 | P1: Reactive | +4.17 [+3.30, +5.03] | +3.95 [+3.04, +4.87] | 616.94 | 618.02 |
| `all` | 2.0 | P2: Worst-First by Band | +15.84 [+15.49, +16.20] | +16.00 [+15.58, +16.44] | 649.62 | 650.28 |
| `all` | 2.0 | P2t: Traffic-Aware Worst-First | +23.46 [+22.23, +24.82] | +15.13 [+14.27, +16.06] | 611.75 | 657.56 |
| `all` | 2.0 | P3: Greedy Benefit/Cost | +73.51 [+73.06, +73.96] | +62.11 [+61.81, +62.42] | 645.37 | 712.36 |
| `all` | 2.0 | P4: Integer Programming | +74.01 [+73.59, +74.44] | +62.41 [+62.14, +62.70] | 646.29 | 712.96 |
| `cracks` | 0.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | 0.0 | P1: Reactive | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 609.80 | 612.33 |
| `cracks` | 0.0 | P2: Worst-First by Band | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 746.94 | 747.36 |
| `cracks` | 0.0 | P2t: Traffic-Aware Worst-First | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 746.94 | 747.36 |
| `cracks` | 0.0 | P3: Greedy Benefit/Cost | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 866.29 | 879.12 |
| `cracks` | 0.0 | P4: Integer Programming | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 866.61 | 879.31 |
| `cracks` | 0.5 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | 0.5 | P1: Reactive | +5.15 [+4.68, +5.63] | +5.34 [+4.86, +5.81] | 604.64 | 606.99 |
| `cracks` | 0.5 | P2: Worst-First by Band | +2.84 [+2.67, +3.01] | +2.40 [+2.22, +2.59] | 744.10 | 744.96 |
| `cracks` | 0.5 | P2t: Traffic-Aware Worst-First | +2.84 [+2.67, +3.01] | +2.40 [+2.22, +2.59] | 744.10 | 744.96 |
| `cracks` | 0.5 | P3: Greedy Benefit/Cost | +9.87 [+9.56, +10.18] | +7.16 [+6.97, +7.35] | 856.42 | 871.96 |
| `cracks` | 0.5 | P4: Integer Programming | +10.22 [+9.84, +10.60] | +7.38 [+7.15, +7.62] | 856.39 | 871.92 |
| `cracks` | 1.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | 1.0 | P1: Reactive | +8.29 [+7.80, +8.76] | +8.59 [+8.10, +9.08] | 601.51 | 603.73 |
| `cracks` | 1.0 | P2: Worst-First by Band | +4.42 [+4.22, +4.62] | +3.46 [+3.24, +3.68] | 742.52 | 743.90 |
| `cracks` | 1.0 | P2t: Traffic-Aware Worst-First | +4.42 [+4.22, +4.62] | +3.46 [+3.24, +3.68] | 742.52 | 743.90 |
| `cracks` | 1.0 | P3: Greedy Benefit/Cost | +23.29 [+22.87, +23.73] | +17.79 [+17.53, +18.06] | 843.01 | 861.33 |
| `cracks` | 1.0 | P4: Integer Programming | +20.77 [+20.29, +21.27] | +16.17 [+15.87, +16.47] | 845.83 | 863.14 |
| `cracks` | 1.5 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | 1.5 | P1: Reactive | +8.99 [+8.44, +9.51] | +9.27 [+8.70, +9.80] | 600.81 | 603.06 |
| `cracks` | 1.5 | P2: Worst-First by Band | +6.49 [+6.28, +6.70] | +4.68 [+4.45, +4.90] | 740.45 | 742.68 |
| `cracks` | 1.5 | P2t: Traffic-Aware Worst-First | +6.49 [+6.28, +6.70] | +4.68 [+4.45, +4.90] | 740.45 | 742.68 |
| `cracks` | 1.5 | P3: Greedy Benefit/Cost | +37.51 [+37.09, +37.93] | +29.51 [+29.25, +29.77] | 828.78 | 849.61 |
| `cracks` | 1.5 | P4: Integer Programming | +36.18 [+35.66, +36.71] | +28.84 [+28.51, +29.16] | 830.42 | 850.47 |
| `cracks` | 2.0 | P0: Do Nothing | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 545.32 | 548.10 |
| `cracks` | 2.0 | P1: Reactive | +8.81 [+8.15, +9.50] | +8.66 [+7.97, +9.36] | 600.99 | 603.67 |
| `cracks` | 2.0 | P2: Worst-First by Band | +10.45 [+10.23, +10.67] | +8.32 [+8.07, +8.56] | 736.49 | 739.04 |
| `cracks` | 2.0 | P2t: Traffic-Aware Worst-First | +10.45 [+10.23, +10.67] | +8.32 [+8.07, +8.56] | 736.49 | 739.04 |
| `cracks` | 2.0 | P3: Greedy Benefit/Cost | +47.35 [+46.95, +47.76] | +39.30 [+39.05, +39.55] | 818.94 | 839.82 |
| `cracks` | 2.0 | P4: Integer Programming | +48.90 [+48.40, +49.41] | +40.28 [+39.96, +40.61] | 817.70 | 839.03 |

---

### 5.3 Explanation of Negative Regrets

In decision theory, regret is defined as the performance deficit of an estimated policy relative to the oracle ($\text{Oracle} - \text{Estimated}$). In this simulation, negative regrets (where the policy under estimated conditions achieves *higher* condition-years than its own oracle) occur because **the heuristic policy thresholds are fixed a priori rather than globally optimized for the dynamic simulation**. 

Because thresholds like $\widehat{\text{PCI}} < 40.0$ or $\widehat{\text{PCI}} \ge 55.0$ are external rules of thumb, applying them to ground-truth PCI (the "oracle") does not compute the mathematically best achievable 10-year maintenance trajectory. Consequently, an imperfect or biased perception estimate can accidentally trigger earlier, broader, or more aggressive interventions that fortuitously outperform the heuristic oracle.

Specific mechanism hypotheses for observed negative regrets:

1. **P1 (Reactive) Under Negative Bias ($b = -5, -10$)**:
   - `[Hypothesis]`: P1 definition strictly forbids intervention on segments with $\widehat{\text{PCI}} \ge 40.0$. Under an oracle, many segments sit between true PCI 40 and 50 and are ignored by P1, leaving substantial budget unspent (utilization $\le 91.85\%$ on `all`, $\le 46.39\%$ on `cracks`). Under negative bias ($b = -5$ or $-10$), these borderline segments appear to have $\widehat{\text{PCI}} < 40.0$. This `[Hypothesis]` forces P1 to treat borderline segments with structural rehabilitation, consuming otherwise idle budget and boosting true condition-years (e.g., negative regret of $-41.52\text{ CY}$ at $b = -10$ on `all`, and $-52.66\text{ CY}$ on `cracks`).
2. **P3/P4 Under Mild Positive Bias ($b = +5$)**:
   - `[Hypothesis]`: P3 and P4 maximize benefit-cost ratio $\Delta P \cdot A \cdot w / c_T$. Preventive maintenance (+15 gain) costs only $c=1.0$, giving a high benefit-cost ratio, but requires $\widehat{\text{PCI}} \ge 55.0$. Under mild positive bias ($b = +5$), this `[Hypothesis]` bumps segments above the preventive threshold, allowing P3 to purchase cheap preventive treatments across wider surface areas. On area-weighted condition, this produces a slight negative regret ($-13.50\text{ CY}$ on `all`, $-7.09\text{ CY}$ on `cracks`), while traffic-weighted regret remains positive or near zero.
3. **P2t Under Positive Bias ($b = +5, +10$) on Pool 'all'**:
   - `[Hypothesis]`: P2t orders segments by $(100 - \widehat{\text{PCI}}) \times w$. Positive bias compresses the $(100 - \widehat{\text{PCI}})$ spread, allowing P2t to allocate treatments across larger surface areas rather than concentrating only on the worst segments, gaining $+14.06$ AW CY over oracle while conceding TW performance. In general, `[Hypothesis]` hard cutoffs amplify small perception errors into discrete policy shifts.
4. **Noise-Only Regret is Strictly Non-Negative**:
   - In Table 5.2, where inspection errors are strictly zero-mean, regret is $\ge 0$ across all policies and pools without exception. Random inspection noise uniformly degrades decision quality.

---

### 5.4 Empirical Comparison of Bias vs. Noise

Grounding the decomposition in the measured tile-level error structure (Section 3: mean signed error on `all` is $-3.11$, on `cracks` is $-3.53$; per-source signed error: CRACK500 $+0.54$, Road Crack $-7.53$, Pothole $-2.38$):

1. **Magnitude Asymmetry**:
   - For optimization policies P3 and P4 on the `cracks` pool, a systematic negative bias of $b = -5.0$ produces an Area-Weighted regret of approximately **$+109\text{ condition-years}$** ($+109.59\text{ CY}$ for P3, $+109.06\text{ CY}$ for P4; Table 5.1).
   - In contrast, zero-mean inspection noise at baseline and half-baseline scales ($\lambda \in [0.5, 1.0]$) produces an Area-Weighted regret of only **$+10\text{ to }+23\text{ condition-years}$** ($+9.87$ to $+23.29\text{ CY}$ for P3, $+10.22$ to $+20.77\text{ CY}$ for P4; Table 5.2).
   - Consequently, systematic negative bias dominates zero-mean inspection noise by approximately $5\times$ to $10\times$ on benefit-cost policies on the `cracks` pool. Elimination of systematic perceptual calibration bias yields far greater decision-theoretic benefit than reducing random zero-mean inspection variance.

---

## 6. Treatment-Model Sensitivity: Smooth Preventive Effectiveness

To test sensitivity to the sharp eligibility threshold ($\text{PCI} \ge 55$), the smooth effectiveness variant $\gamma = \text{clip}((\text{PCI} - 40)/30, 0, 1)$ was simulated across 100 runs on both pools:

### Pool 'all' (Budget 10%, Ratio 5.0, 100 Runs):
| Policy | Est CY (AW) | Orc CY (AW) | Regret AW [95% CI] | Rank (Est) |
| :--- | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 361.50 | 361.50 | +0.00 [+0.00, +0.00] | Rank 5 |
| **P1: Reactive** | 642.58 | 620.51 | -22.07 [-23.17, -20.95] | Rank 2 |
| **P2: Worst-First by Band** | 660.32 | 664.06 | +3.74 [+3.37, +4.11] | Rank 1 |
| **P3: Greedy Benefit/Cost** | 637.75 | 679.98 | +42.23 [+41.77, +42.70] | Rank 3 |
| **P4: Integer Programming** | 637.03 | 680.20 | +43.16 [+42.77, +43.55] | Rank 4 |

### Pool 'cracks' (Budget 10%, Ratio 5.0, 100 Runs):
| Policy | Est CY (AW) | Orc CY (AW) | Regret AW [95% CI] | Rank (Est) |
| :--- | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 545.23 | 545.23 | +0.00 [+0.00, +0.00] | Rank 5 |
| **P1: Reactive** | 618.62 | 609.70 | -8.92 [-9.48, -8.37] | Rank 4 |
| **P2: Worst-First by Band** | 770.33 | 743.04 | -27.29 [-27.47, -27.11] | Rank 3 |
| **P3: Greedy Benefit/Cost** | 778.87 | 845.03 | +66.16 [+65.65, +66.67] | Rank 1 |
| **P4: Integer Programming** | 775.87 | 843.72 | +67.85 [+67.36, +68.35] | Rank 2 |

---

## 7. Sensitivity Grids: Grid 1 and Grid 2 (Fixed Absolute Budget with P2t)

To evaluate how fiscal capital rules interact with unit cost variations, two separate 18-cell sensitivity grids were evaluated across 100 Monte Carlo runs per cell with synchronized random seeds (totaling 36 cells and 18,000 policy-year evaluations). In each cell, nominal winners and statistical co-leaders (policies whose paired difference 95% bootstrap CI relative to the nominal winner includes 0) were determined.

### 7.1 Grid 1: Budget Scales with Rehab Cost ($\text{Budget} = c_{\text{rehab}} \times \text{Area} \times X\%$)
*In Grid 1, the total capital budget scales proportionally with the unit cost of rehabilitation:*

| Pool | Budget (%) | Cost Ratio | P0 | P1 | P2 | P3 | P4 | Winner (Area-Weighted) | Co-Leaders (95% CI Contains 0) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `all` | 5% | 3 | 361.9 | 527.2 | 501.5 | 508.1 | 509.9 | **P1: Reactive** | P1 |
| `all` | 5% | 5 | 361.9 | 528.4 | 511.2 | 534.0 | 532.9 | **P3: Greedy Benefit/Cost** | P3 |
| `all` | 5% | 8 | 361.9 | 529.9 | 524.5 | 565.3 | 565.3 | **P3: Greedy Benefit/Cost** | P3, P4 |
| `all` | 10% | 3 | 361.9 | 642.8 | 660.1 | 614.9 | 616.2 | **P2: Worst-First by Band** | P2 |
| `all` | 10% | 5 | 361.9 | 642.8 | 660.4 | 655.8 | 656.9 | **P2: Worst-First by Band** | P2 |
| `all` | 10% | 8 | 361.9 | 642.8 | 661.2 | 733.4 | 736.8 | **P4: Integer Programming** | P4 |
| `all` | 20% | 3 | 361.9 | 653.5 | 755.1 | 781.0 | 784.8 | **P4: Integer Programming** | P4 |
| `all` | 20% | 5 | 361.9 | 653.5 | 755.6 | 837.8 | 837.8 | **P3: Greedy Benefit/Cost** | P3, P4 |
| `all` | 20% | 8 | 361.9 | 653.5 | 757.1 | 870.9 | 870.0 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 5% | 3 | 545.7 | 595.3 | 663.7 | 641.9 | 643.3 | **P2: Worst-First by Band** | P2 |
| `cracks` | 5% | 5 | 545.7 | 595.3 | 747.2 | 693.3 | 690.8 | **P2: Worst-First by Band** | P2 |
| `cracks` | 5% | 8 | 545.7 | 595.3 | 771.3 | 748.7 | 748.5 | **P2: Worst-First by Band** | P2 |
| `cracks` | 10% | 3 | 545.7 | 618.4 | 770.3 | 719.0 | 718.3 | **P2: Worst-First by Band** | P2 |
| `cracks` | 10% | 5 | 545.7 | 618.4 | 775.0 | 810.8 | 810.9 | **P4: Integer Programming** | P3, P4 |
| `cracks` | 10% | 8 | 545.7 | 618.4 | 775.4 | 892.2 | 892.5 | **P4: Integer Programming** | P4 |
| `cracks` | 20% | 3 | 545.7 | 625.6 | 775.4 | 867.6 | 863.1 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 20% | 5 | 545.7 | 625.6 | 775.4 | 907.2 | 906.0 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 20% | 8 | 545.7 | 625.6 | 775.4 | 913.1 | 911.8 | **P3: Greedy Benefit/Cost** | P3 |

---

### 7.2 Grid 2: Fixed Absolute Budget ($\text{Budget} = 5.0 \times \text{Area} \times X\%$, Including P2t)
*In Grid 2, the budget is decoupled from unit cost changes and fixed to the reference cost $c_{\text{ref}} = 5.0$. Both Area-Weighted and Traffic-Weighted winners and co-leaders are reported across the 18 cells:*

| Pool | Budget (%) | Cost Ratio | Nominal Winner AW | Co-Leaders AW | Nominal Winner TW | Co-Leaders TW |
| :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| `all` | 5% | 3 | P2_worst_first_by_band | P2_worst_first_by_band | P4_integer_programming | P4_integer_programming |
| `all` | 5% | 5 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `all` | 5% | 8 | P4_integer_programming | P4_integer_programming | P4_integer_programming | P4_integer_programming |
| `all` | 10% | 3 | P2_worst_first_by_band | P2_worst_first_by_band | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `all` | 10% | 5 | P2_worst_first_by_band | P2_worst_first_by_band | P4_integer_programming | P4_integer_programming |
| `all` | 10% | 8 | P3_greedy_benefit_cost | P3, P4 | P4_integer_programming | P4_integer_programming |
| `all` | 20% | 3 | P4_integer_programming | P3, P4 | P4_integer_programming | P3, P4 |
| `all` | 20% | 5 | P3_greedy_benefit_cost | P3, P4 | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `all` | 20% | 8 | P4_integer_programming | P4_integer_programming | P4_integer_programming | P4_integer_programming |
| `cracks` | 5% | 3 | P2_worst_first_by_band | P2_worst_first_by_band | P2t_traffic_aware_worst_first | P2t, P4 |
| `cracks` | 5% | 5 | P2_worst_first_by_band | P2_worst_first_by_band | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 5% | 8 | P2_worst_first_by_band | P2, P2t | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 10% | 3 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 10% | 5 | P4_integer_programming | P3, P4 | P4_integer_programming | P3, P4 |
| `cracks` | 10% | 8 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 20% | 3 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 20% | 5 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |
| `cracks` | 20% | 8 | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost | P3_greedy_benefit_cost |

#### Grid 2 Victory Tallies across 18 Cells:

| Policy | Nominal Wins (AW) | Co-Wins (AW) | Nominal Wins (TW) | Co-Wins (TW) |
| :--- | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 0 | 0 | 0 | 0 |
| **P1: Reactive** | 0 | 0 | 0 | 0 |
| **P2: Worst-First by Band** | 6 | 6 | 0 | 0 |
| **P2t: Traffic-Aware Worst-First** | 0 | 1 | 1 | 1 |
| **P3: Greedy Benefit/Cost** | 8 | 10 | 10 | 12 |
| **P4: Integer Programming** | 4 | 6 | 7 | 8 |

---

## 8. What We Can and Cannot Conclude

### 8.1 What We Can Conclude (Supported Directly by Data)

1. **Policy Rankings Depend Heavily on Evaluation Metric (AW vs. TW)**:
   - Under the area-weighted (AW) metric in the base scenario (Table 4.1), P2 leads with $660.50\text{ CY}$ compared to $656.77$ for P4 and $626.94$ for P2t.
   - Under the traffic-weighted (TW) metric, P4 and P3 dominate with $731.54$ and $730.69\text{ TW CY}$, outperforming P2 ($661.66$) and P2t ($672.45$).
   - Across Grid 2's 18 cells, P2 achieves 6 nominal wins in AW condition but 0 nominal wins in TW condition. P3/P4 win 17 of 18 cells in TW condition.
2. **P2t Improves Traffic-Weighted Condition Over P2 at the Expense of Area-Weighted Condition**:
   - On the `all` pool (Table 4.1), sorting by $(100 - \widehat{\text{PCI}}) \times w_i$ rather than PCI alone gains $+10.79\text{ TW CY}$ [95% CI: $+10.43, +11.14$] over P2, but loses $-33.56\text{ AW CY}$ [$-33.88, -33.23$].
   - In Grid 2, P2t captures a nominal win in TW condition on `cracks, 5% budget, ratio 3` and co-wins in AW on `cracks, 5% budget, ratio 8`.
3. **P1 Spending Cap is a Mathematical Property of Its Definition**:
   - P1 definition strictly forbids intervention on segments with $\widehat{\text{PCI}} \ge 40.0$.
   - In the base scenario on the `all` pool, P1 spends $3,068,687$ ($91.85\%$ of budget). On the `cracks` pool (where initial conditions are better), P1 spends only $1,629,136$ ($46.39\%$ of the $3,511,774$ pool budget, or $48.76\%$ if evaluated against the $3,340,965$ baseline reference budget).
   - This spend restriction is a direct definitional property: reactive worst-first policies cannot absorb municipal capital allocations when severe backlog is cleared.
4. **Information Regret Differs Fundamentally Between Bias and Noise**:
   - **Zero-mean noise ($\lambda$)**: Regret is strictly non-negative ($\ge 0$) across all policies and pools (Table 5.2). Regret scales monotonically with noise scale $\lambda$ for P3/P4, reaching $+74.01\text{ AW CY}$ and $+62.41\text{ TW CY}$ at $\lambda = 2.0$.
   - **Systematic bias ($b$)**: Creates non-monotonic regret curves with negative regrets due to threshold mechanics:
     - *Hypothesis*: Negative bias ($b < 0$) forces P1 to treat borderline segments ($40 \le \text{PCI} < 50$), artificially spending unallocated capital and outperforming its own oracle by $-41.52\text{ CY}$ ($b=-10$ on `all`) and $-52.66\text{ CY}$ on `cracks`.
     - *Hypothesis*: Positive bias ($b = +5$) bumps segments above the preventive threshold ($55.0$), allowing P3/P4 to buy cheap crack sealing across large areas and outperforming their AW oracle by $-13.50\text{ CY}$ on `all`.
5. **Ranking Sensitivity to Treatment Response Curves (Cliff vs. Smooth)**:
   - Under the sharp cliff threshold ($\text{PCI} \ge 55$), P2 ranks #1 on AW in the base scenario ($660.50$ vs $656.77$ for P4).
   - Under the smooth response curve (Section 6), information regret for P3/P4 decreases from $+63.5\text{ CY}$ to $+43.16\text{ CY}$, consistent with the hypothesis that hard cutoffs amplify inspection noise penalties.

---

### 8.2 What We Cannot Conclude

1. **No Real-World Municipal Financial Claims**:
   - All costs ($1, 3, 5, 8, 12$) are dimensionless relative benchmark units. We cannot claim that implementing deep-learning inspection "saves $X$ million dollars in taxpayer funds."
2. **No Physical Infrastructure Guarantees**:
   - The simulation utilizes stylized lognormal degradation curves and immediate condition resets. Actual physical pavement deterioration depends on climate, freeze-thaw cycles, asphalt binder aging, and heavy commercial axle-load spectra not modeled here.
3. **No Unconditional Policy Superiority**:
   - No single policy wins across all scenarios. Under low budgets (5%) and area-weighting, condition-band heuristics (P2) dominate; under traffic-weighting or higher budgets, benefit-cost optimizers (P3/P4) dominate.
4. **Pothole Inferences Carry High Variance**:
   - As documented in Section 3, pothole tiles have an error standard deviation of $\sigma = 18.81$ sPCI points (more than double CRACK500's $\sigma = 8.48$). Deployed maintenance decisions driven by automated pothole detection entail substantially higher uncertainty.

---

## 9. Visualizations and Reference Figures

All publication figures are generated and stored in `docs/figures/`:

1. `human_agreement_disagreements.png` (1,758,555 bytes): Qualitative overlay visual of the 8 human disagreement pairs from Step 4C.
2. `simulation_condition_trajectories.png` (170,752 bytes): 10-year trajectory of network condition across policies P0–P4.
3. `simulation_cost_effectiveness.png` (81,679 bytes): Condition-years gained per 1,000 units of real spend.
4. `simulation_regret_vs_bias.png` (885,112 bytes): Area- and traffic-weighted regret vs systematic bias $b \in [-10, +10]$ across pools.
5. `simulation_regret_vs_lambda.png` (1,023,045 bytes): Area- and traffic-weighted regret vs noise scale $\lambda \in [0, 2]$ across pools.
6. `simulation_sensitivity_grid_fixed.png` (158,553 bytes): Sensitivity Grid 2 heatmap with P2t co-leaders and winners across 18 cells.
7. `simulation_sensitivity_grid_scaling.png` (49,496 bytes): Sensitivity Grid 1 heatmap (scaling budget).
8. `test_overlays_crack500.png` (2,572,546 bytes): 8-image test split segmentation predictions on CRACK500.
9. `test_overlays_pothole.png` (2,113,079 bytes): 8-image test split segmentation predictions on Pothole-600.
10. `test_overlays_road_crack.png` (2,365,979 bytes): 8-image test split segmentation predictions on Road Crack.

**Total Figure Count in `docs/figures/`**: 10 files.
