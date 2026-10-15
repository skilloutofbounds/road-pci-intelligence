# Budget-Constrained Pavement Maintenance Simulation (Step 5B)

## 1. Executive Summary & Objective

This document specifies the formulation, mathematical models, empirical simulation results, and architectural limitations of the budget-constrained pavement maintenance simulation engine (`src/simulation/`).

The objective is to evaluate how an automated pavement inspection system (specifically, the frozen deep multi-task segmentation model + simplified Pavement Condition Index, **sPCI v1**) interacts with municipal asset management policies under annual capital budget constraints. The simulation models a synthetic road network over a 10-year planning horizon, comparing five distinct maintenance policies (P0 through P4) under both estimated condition data and oracle ground-truth conditions.

> [!IMPORTANT]
> **EXPLICIT DISCLAIMER**: All financial costs, treatment costs per square meter, treatment effectiveness thresholds, and deterioration rate parameters in this module are **ILLUSTRATIVE BENCHMARK ASSUMPTIONS**. They are established for algorithmic benchmarking, sensitivity analysis, and comparative policy ranking. **No claims are made regarding real municipality accounting, actual tax dollar savings, or physical infrastructure life-cycle guarantees.**

---

## 2. Mathematical Formulation & Engine Mechanics

### 2.1 Synthetic Network Topology
The network consists of $N = 200$ synthetic roadway segments. Each segment $i \in \{0, \dots, N-1\}$ is characterized by:
- Segment ID: $\text{seg\_000}$ to $\text{seg\_199}$
- Length: $L_i \sim \mathcal{U}(100.0, 600.0)\text{ meters}$
- Width: $W_i \sim \mathcal{U}(6.0, 14.0)\text{ meters}$
- Surface Area: $A_i = L_i \times W_i\text{ m}^2$ (total network surface area $\approx 668,192.9\text{ m}^2$)
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

### 2.6 Maintenance Allocation Policies (P0 – P4)
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
4. **P3 (Greedy Benefit/Cost Heuristic)**:
   - For all segments $i$ and eligible treatments $T$:
     $$\text{Benefit}_{i, T} = \Delta \widehat{P}_{i, T} \times A_i \times w_i, \quad \text{Cost}_{i, T} = c_T \times A_i, \quad \text{Ratio}_{i, T} = \frac{\text{Benefit}_{i, T}}{\text{Cost}_{i, T}}$$
   - Greedily selects candidate with highest benefit/cost ratio, at most one treatment per segment per year, until budget is exhausted.
5. **P4 (Exact 0/1 Integer Linear Program)**:
   - Solves the exact single-year optimum knapsack allocation:
     $$\max_{x_{i, T} \in \{0, 1\}} \sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Benefit}_{i, T} \cdot x_{i, T}$$
     subject to:
     $$\sum_{T \in \mathcal{T}} x_{i, T} = 1 \quad \forall i, \quad \sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Cost}_{i, T} \cdot x_{i, T} \le \text{Budget}_{\text{annual}}$$
   - Solved via `scipy.optimize.milp` using the exact HiGHS branch-and-cut simplex backend.

---

## 3. Spending Bug Audit & Base Scenario Results

### 3.1 Spending Discrepancy Investigation
In the initial Step 5A report, the "Total Spent" column reported $3,340,965.0$ identically across P1 through P4. 
- **Root Cause**: The previous report inadvertently transcribed the 10-year cumulative budget figure ($334,096.5 \times 10$) into the markdown table rather than the real spend column tracked by the simulation engine.
- **Empirical Real Spend Finding**:
  - The simulation engine's internal accounting tracks actual expenditure $\sum_i c_{T_i} A_i$.
  - While P2, P3, and P4 utilize $>99.8\%$ of the annual budget, **P1 (Reactive) uses only 91.85% of the budget** ($3,068,687$ spent out of $3,340,965$), leaving approximately $272,000$ cost units unspent over 10 years because all segments with $\widehat{\text{PCI}} < 40$ are repaired.
  - When budget is doubled to 20% ($6,681,929$), **P1 real spend remains at 3,065,995** (utilization drops to $45.88\%$). Reactive worst-first policies cannot productively absorb high capital allocations without expanding treatment eligibility to preventive preservation.

### 3.2 Base Scenario Performance Table (200 Monte Carlo Runs, Pool 'all')
*Annual Budget: 10% of network rehab ($334,096.5$ units/yr), Rehab Ratio: 5.0, 200 runs (seeds 1000–1199):*

| Policy | Condition-Years (AW) [95% CI] | Condition-Years (TW) [95% CI] | Real Spend [95% CI] | Budget Util (%) | Eff AW (CY/1k) [95% CI] | Eff TW (CY/1k) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 361.64 [361.34, 361.94] | 357.01 [356.69, 357.33] | 0 [0, 0] | 0.00% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| **P1: Reactive** | 642.80 [642.28, 643.31] | 644.01 [643.48, 644.53] | 3,068,687 [3,063,713, 3,073,385] | 91.85% | 0.0916 [0.0914, 0.0919] | 0.0935 [0.0933, 0.0938] |
| **P2: Worst-First by Band** | 660.50 [660.22, 660.77] | 661.66 [661.33, 661.98] | 3,337,775 [3,337,667, 3,337,878] | 99.90% | 0.0895 [0.0894, 0.0896] | 0.0913 [0.0912, 0.0914] |
| **P3: Greedy Benefit/Cost** | 655.83 [655.60, 656.05] | 730.70 [730.54, 730.87] | 3,337,105 [3,336,997, 3,337,212] | 99.88% | 0.0882 [0.0881, 0.0882] | 0.112 [0.112, 0.112] |
| **P4: Integer Programming** | 656.77 [656.54, 656.99] | 731.55 [731.38, 731.71] | 3,339,967 [3,339,943, 3,339,990] | 99.97% | 0.0884 [0.0883, 0.0884] | 0.112 [0.112, 0.112] |

*Efficiency is defined as condition-years gained relative to P0 per 1,000 cost units of expenditure.*

### 3.3 Paired Differences (Synchronized Random Seeds Across Policies)
- **P1 (Reactive) vs. P0**:
  - Area-Weighted: $+281.17\text{ CY}$ [95% CI: $+280.57, +281.75$]
  - Traffic-Weighted: $+287.00\text{ CY}$ [95% CI: $+286.38, +287.60$]
- **P2 (Worst-First by Band) vs. P0**:
  - Area-Weighted: $+298.86\text{ CY}$ [95% CI: $+298.51, +299.20$]
  - Traffic-Weighted: $+304.65\text{ CY}$ [95% CI: $+304.25, +305.05$]
  - $\rightarrow$ vs. P1: $\text{AW} = +17.69$ [$+17.17, +18.23$], $\text{TW} = +17.65$ [$+17.10, +18.21$]
- **P3 (Greedy Benefit/Cost) vs. P0**:
  - Area-Weighted: $+294.19\text{ CY}$ [95% CI: $+293.91, +294.47$]
  - Traffic-Weighted: $+373.69\text{ CY}$ [95% CI: $+373.42, +373.98$]
  - $\rightarrow$ vs. P1: $\text{AW} = +13.02$ [$+12.47, +13.58$], $\text{TW} = +86.69$ [$+86.16, +87.23$]
- **P4 (0/1 Integer Programming) vs. P0**:
  - Area-Weighted: $+295.13\text{ CY}$ [95% CI: $+294.84, +295.42$]
  - Traffic-Weighted: $+374.54\text{ CY}$ [95% CI: $+374.23, +374.83$]
  - $\rightarrow$ vs. P1: $\text{AW} = +13.97$ [$+13.42, +14.52$], $\text{TW} = +87.54$ [$+87.00, +88.09$]

### 3.4 Resolution of Metric Mismatch
When evaluating area-weighted condition-years, P2 appears nominally superior (660.50 vs 656.77 for P4). However, P3 and P4 explicitly optimize traffic-weighted benefit $\Delta P \cdot A \cdot w$. When evaluating the traffic-weighted metric:
- P4 achieves **731.55 TW CY**, outperforming P2 (661.66 TW CY) by **+69.89 TW CY** ($+10.6\%$) and outperforming P1 by **+87.54 TW CY**.
- P4 efficiency in traffic-weighted terms reaches **0.112 CY/1k cost**, compared to 0.0913 for P2 and 0.0935 for P1.

---

## 4. Treatment-Model Sensitivity: Smooth Preventive Effectiveness

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

**Key Findings on Smooth Treatment Formulation**:
1. On the `all` pool, ranking shifts: P1 moves to Rank 2 ahead of P3/P4 because preventive gains on moderately damaged segments (PCI 40–55) are attenuated by the smooth ramp.
2. On the `cracks` pool, ranking is preserved: P3 and P4 remain dominant (778.87 and 775.87 CY), followed by P2 (770.33 CY) and P1 (618.62 CY).
3. Information regret for P3 and P4 on the `all` pool decreases from $+63.5\text{ CY}$ under the cliff model to $+43.16\text{ CY}$ under the smooth model, confirming that the hard cliff at PCI 55 had amplified observation error penalties.

---

## 5. Information-Quality Study (Redone)

### 5.1 Oracle vs. Estimated across Tile Pools (200 Runs)
> [!NOTE]
> Ground-truth masks for pothole tiles lack crack annotations. Consequently, in the `all` pool, tiles containing both potholes and fine cracks have partial ground truth, which introduces unmodeled observation discrepancies.

| Tile Pool | Policy | Oracle AW | Est AW | Regret AW [95% CI] | Regret TW [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **cracks only** | **P1: Reactive** | 609.71 | 618.69 | -8.97 [-9.40, -8.55] | -8.26 [-8.69, -7.83] |
| **cracks only** | **P2: Worst-First by Band** | 746.94 | 774.99 | -28.06 [-28.18, -27.93] | -28.93 [-29.07, -28.79] |
| **cracks only** | **P3: Greedy Benefit/Cost** | 865.90 | 810.87 | +55.03 [+54.64, +55.43] | +33.65 [+33.42, +33.89] |
| **cracks only** | **P4: Integer Programming** | 866.33 | 810.80 | +55.53 [+55.13, +55.93] | +34.00 [+33.76, +34.24] |
| **all** | **P1: Reactive** | 620.61 | 642.80 | -22.19 [-22.94, -21.44] | -22.26 [-23.01, -21.49] |
| **all** | **P2: Worst-First by Band** | 665.32 | 660.50 | +4.82 [+4.54, +5.11] | +4.48 [+4.12, +4.85] |
| **all** | **P3: Greedy Benefit/Cost** | 719.27 | 655.83 | +63.44 [+63.13, +63.75] | +44.00 [+43.80, +44.20] |
| **all** | **P4: Integer Programming** | 720.34 | 656.77 | +63.57 [+63.23, +63.89] | +43.81 [+43.59, +44.02] |

### 5.2 Continuous Noise Scaling Analysis: $\widehat{\text{PCI}}(\lambda) = \text{True} + \lambda \times (\text{Pred} - \text{True})$
*Regret measured as $\text{Oracle CY} - \text{Estimated}(\lambda)\text{ CY}$ across 100 runs per $\lambda$:*

| Pool | $\lambda$ | Policy | Regret AW [95% CI] | Regret TW [95% CI] |
| :---: | :---: | :--- | :---: | :---: |
| `all` | 0.0 | P1: Reactive | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `all` | 0.0 | P3: Greedy | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `all` | 0.0 | P4: Integer Prog | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `all` | 0.5 | P1: Reactive | -9.15 [-10.05, -8.26] | -9.40 [-10.36, -8.45] |
| `all` | 0.5 | P3: Greedy | +34.84 [+34.40, +35.29] | +21.73 [+21.46, +22.01] |
| `all` | 0.5 | P4: Integer Prog | +34.43 [+33.98, +34.90] | +21.43 [+21.16, +21.71] |
| `all` | 1.0 | P1: Reactive | -22.07 [-23.17, -20.95] | -22.17 [-23.30, -21.01] |
| `all` | 1.0 | P3: Greedy | +63.30 [+62.86, +63.72] | +43.92 [+43.64, +44.18] |
| `all` | 1.0 | P4: Integer Prog | +63.32 [+62.87, +63.78] | +43.66 [+43.36, +43.95] |
| `all` | 1.5 | P1: Reactive | -29.81 [-30.78, -28.87] | -29.69 [-30.68, -28.72] |
| `all` | 1.5 | P3: Greedy | +83.41 [+82.91, +83.88] | +61.85 [+61.53, +62.16] |
| `all` | 1.5 | P4: Integer Prog | +83.06 [+82.53, +83.57] | +61.40 [+61.05, +61.76] |
| `all` | 2.0 | P1: Reactive | -26.34 [-27.20, -25.50] | -26.03 [-26.86, -25.20] |
| `all` | 2.0 | P3: Greedy | +96.75 [+96.29, +97.21] | +75.34 [+75.00, +75.68] |
| `all` | 2.0 | P4: Integer Prog | +96.61 [+96.08, +97.15] | +74.98 [+74.61, +75.36] |
| `cracks` | 0.5 | P3: Greedy | +26.42 [+25.94, +26.90] | +16.00 [+15.71, +16.29] |
| `cracks` | 0.5 | P4: Integer Prog | +25.53 [+24.90, +26.16] | +15.54 [+15.17, +15.92] |
| `cracks` | 1.0 | P3: Greedy | +55.03 [+54.51, +55.56] | +33.64 [+33.33, +33.96] |
| `cracks` | 1.0 | P4: Integer Prog | +55.53 [+55.00, +56.04] | +34.00 [+33.69, +34.31] |
| `cracks` | 1.5 | P3: Greedy | +80.62 [+80.11, +81.13] | +49.91 [+49.61, +50.22] |
| `cracks` | 1.5 | P4: Integer Prog | +81.14 [+80.69, +81.59] | +50.30 [+50.03, +50.57] |
| `cracks` | 2.0 | P3: Greedy | +102.17 [+101.69, +102.65] | +65.15 [+64.86, +65.45] |
| `cracks` | 2.0 | P4: Integer Prog | +103.31 [+102.87, +103.77] | +65.93 [+65.66, +66.22] |

*Regret scales monotonically with inspection noise $\lambda$ for both benefit/cost optimization policies (P3 and P4).*

### 5.3 P1 Physical Intervention Accounting (Numbers Only)
*Comparison of physical intervention counts and expenditure for P1 (Reactive) over 10 years (200 runs):*

| Metric | Oracle Ground Truth | Estimated Inspection |
| :--- | :---: | :---: |
| **Mean Total Segments Treated (10-yr)** | 166.4 | 180.1 |
| **Mean Total Reconstructions (10-yr)** | 0.0 | 0.0 |
| **Mean Total Spend (10-yr)** | 2,871,030.3 | 3,068,687.4 |

---

## 6. Dual Sensitivity Grids: Scaling vs. Fixed Absolute Budget

To evaluate how fiscal capital rules interact with unit cost variations, two separate 18-cell sensitivity grids were evaluated across 100 Monte Carlo runs per cell with synchronized random seeds (totaling 36 cells and 18,000 policy-year evaluations). In each cell, nominal winners and statistical co-leaders (policies whose paired difference 95% bootstrap CI relative to the nominal winner includes 0) were determined.

### 6.1 Grid 1: Budget Scales with Rehab Cost ($\text{Budget} = c_{\text{rehab}} \times \text{Area} \times X\%$)
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

### 6.2 Grid 2: Fixed Absolute Budget ($\text{Budget} = 5.0 \times \text{Area} \times X\%$)
*In Grid 2, the budget is decoupled from unit cost changes and fixed to the reference cost $c_{\text{ref}} = 5.0$:*

| Pool | Budget (%) | Cost Ratio | P0 | P1 | P2 | P3 | P4 | Winner (Area-Weighted) | Co-Leaders (95% CI Contains 0) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `all` | 5% | 3 | 361.9 | 618.8 | 622.5 | 583.7 | 585.5 | **P2: Worst-First by Band** | P2 |
| `all` | 5% | 5 | 361.9 | 528.4 | 511.2 | 534.0 | 532.9 | **P3: Greedy Benefit/Cost** | P3 |
| `all` | 5% | 8 | 361.9 | 469.7 | 465.5 | 505.8 | 507.7 | **P4: Integer Programming** | P4 |
| `all` | 10% | 3 | 361.9 | 653.5 | 743.2 | 735.9 | 736.0 | **P2: Worst-First by Band** | P2 |
| `all` | 10% | 5 | 361.9 | 642.8 | 660.4 | 655.8 | 656.9 | **P2: Worst-First by Band** | P2 |
| `all` | 10% | 8 | 361.9 | 567.1 | 564.4 | 602.5 | 602.1 | **P3: Greedy Benefit/Cost** | P3, P4 |
| `all` | 20% | 3 | 361.9 | 653.5 | 773.5 | 873.3 | 873.5 | **P4: Integer Programming** | P3, P4 |
| `all` | 20% | 5 | 361.9 | 653.5 | 755.6 | 837.8 | 837.8 | **P3: Greedy Benefit/Cost** | P3, P4 |
| `all` | 20% | 8 | 361.9 | 652.0 | 709.1 | 798.8 | 799.8 | **P4: Integer Programming** | P4 |
| `cracks` | 5% | 3 | 545.7 | 613.2 | 759.0 | 701.5 | 701.9 | **P2: Worst-First by Band** | P2 |
| `cracks` | 5% | 5 | 545.7 | 595.3 | 747.2 | 693.3 | 690.8 | **P2: Worst-First by Band** | P2 |
| `cracks` | 5% | 8 | 545.7 | 580.9 | 700.7 | 689.3 | 685.9 | **P2: Worst-First by Band** | P2 |
| `cracks` | 10% | 3 | 545.7 | 625.5 | 775.4 | 823.0 | 822.1 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 10% | 5 | 545.7 | 618.4 | 775.0 | 810.8 | 810.9 | **P4: Integer Programming** | P3, P4 |
| `cracks` | 10% | 8 | 545.7 | 603.3 | 774.1 | 813.1 | 812.3 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 20% | 3 | 545.7 | 625.6 | 775.4 | 907.6 | 904.3 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 20% | 5 | 545.7 | 625.6 | 775.4 | 907.2 | 906.0 | **P3: Greedy Benefit/Cost** | P3 |
| `cracks` | 20% | 8 | 545.7 | 623.4 | 775.4 | 906.5 | 905.5 | **P3: Greedy Benefit/Cost** | P3 |

---

### 6.3 Policy Victory & Co-Leader Tallies (18 Cells per Grid)

| Policy | Grid 1 Won (Nominal) | Grid 1 Co-Won | Grid 2 Won (Nominal) | Grid 2 Co-Won |
| :--- | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 0 | 0 | 0 | 0 |
| **P1: Reactive** | 1 | 1 | 0 | 0 |
| **P2: Worst-First by Band** | 6 | 6 | 6 | 6 |
| **P3: Greedy Benefit/Cost** | 7 | 8 | 8 | 10 |
| **P4: Integer Programming** | 4 | 6 | 4 | 6 |

---

### 6.4 Comparative Analysis of Grid 1 vs. Grid 2
1. **The Cost-Ratio Confound in Grid 1**:
   In Grid 1, the municipal budget was formulated as $c_{\text{rehab}} \times \text{area} \times X\%$. When $c_{\text{rehab}}$ increases from 3 to 8, the available budget scales by $8/3 = 2.67\times$. This budget expansion allows policies like P3 and P4 to purchase vast quantities of cheap preventive maintenance ($c = 1.0$), artificially boosting condition-years up to 870–913 CY.
2. **True Cost Pressure in Grid 2**:
   In Grid 2, fixing the absolute budget reveals the true trade-off: when rehabilitation cost increases from 3 to 8, fewer square meters of structural pavement can be repaired within the constant dollar cap. Consequently, on the `all` pool under 10% budget, condition drops from 736.0 CY (at Ratio 3) to 602.1 CY (at Ratio 8).
3. **Robustness of P2 Across Capital Formulations**:
   P2 (Worst-First by Band) achieves exactly 6 nominal wins in both Grid 1 and Grid 2. It consistently wins when budgets are restricted (5% budget) or when initial distress is moderate (`cracks` pool at 5% and 10% Ratio 3).
4. **Statistical Ties Between P3 and P4**:
   On a 200-segment network with continuous area budgets, greedy benefit/cost (P3) is a near-optimal continuous relaxation of the 0/1 knapsack integer program (P4). In 8 cells across the two grids, P3 and P4 are statistical co-leaders (their paired difference 95% bootstrap CI includes 0). P4 nominally leads when discrete combinatorial constraints require branch-and-cut precision, while P3 leads when marginal fractional investments are decisive.

---

## 7. Threats to Validity, Limitations, and Conclusions

### 7.1 What We Can Conclude
1. **Model Inaccuracy Penalizes Preventive Optimization Far More than Worst-First**:
   Policies that optimize benefit-to-cost ratios (P3 and P4) depend sensitively on threshold boundaries (e.g., whether a road is above or below PCI 55 for crack sealing). Model inspection errors cause significant regret (+55 to +63 CY loss). In contrast, reactive policies (P1) and condition-band policies (P2) are relatively robust to inspection noise.
2. **Metric Alignment Matters**:
   When evaluated in area-weighted terms, P2 appears nominally competitive with P4. However, when evaluated on traffic-weighted condition-years (matching the municipal objective), P4 and P3 achieve +87 TW CY higher performance than P1 and +70 TW CY higher than P2.
3. **Reactive Allocations Cannot Scale With Capital Budgets**:
   P1 (Reactive) exhausts its pool of severely damaged roads at higher budget allocations (20% budget), leaving over 54% of available capital unspent. Expanding treatment eligibility to preventive preservation is mathematically necessary to utilize larger budgets.

### 7.2 What We Cannot Conclude
1. **No Real-World Municipal Financial Claims**:
   The costs ($1, 5, 12$) are dimensionless relative units. We cannot claim that implementing this system "saves millions of tax dollars".
2. **No Physical Infrastructure Guarantees**:
   The simulation employs an empirical lognormal deterioration curve with stylized maintenance resets. Real asphalt deterioration depends on climate, subgrade freeze-thaw, axle loads, and drainage conditions not captured in this model.
3. **Unvalidated Pothole Decisions**:
   Given that sPCI agreement on pothole pairs in Step 4C was only 42.9%, automated maintenance decisions on potholed streets in deployed settings carry high uncertainty.

---

## 8. Visualizations and Reference Figures

### 8.1 Network Condition Trajectories over Time (10-Year Horizon)
![Condition Trajectories](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_condition_trajectories.png)
*Figure 1: Mean area-weighted and traffic-weighted True PCI across 10 years for policies P0 through P4 (10% budget, cost ratio 5, tile pool `all`), with 95% Monte Carlo confidence bands.*

### 8.2 Cost-Effectiveness Chart
![Cost Effectiveness](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_cost_effectiveness.png)
*Figure 2: Condition-Years Gained relative to P0 versus real cumulative expenditure over 10 years across policies.*

### 8.3 Information Regret vs. Inspection Noise Level ($\lambda$)
![Regret vs Lambda](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_regret_vs_lambda.png)
*Figure 3: Area-weighted regret ($\text{Oracle} - \text{Estimated}$) as a function of noise scaling parameter $\lambda \in [0, 2]$.*

### 8.4 Sensitivity Grid Heatmaps
![Grid 1 Scaling](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_sensitivity_grid_scaling.png)
*Figure 4: Sensitivity Grid 1 (Budget Scales with Rehab Cost).*

![Grid 2 Fixed](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_sensitivity_grid_fixed.png)
*Figure 5: Sensitivity Grid 2 (Fixed Absolute Budget).*

### 8.5 Test Split Overlays and Disagreements
- CRACK500 8-Image Test Split Overlay: `docs/figures/test_overlays_crack500.png`
- Pothole 8-Image Test Split Overlay: `docs/figures/test_overlays_pothole.png`
- Road Crack 8-Image Test Split Overlay: `docs/figures/test_overlays_road_crack.png`
- Human Agreement Disagreements: `docs/figures/human_agreement_disagreements.png`
