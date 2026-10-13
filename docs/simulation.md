# Budget-Constrained Pavement Maintenance Simulation (Step 5A)

## 1. Executive Summary & Objective

This document specifies the formulation, mathematical models, experimental results, and architectural limitations of the budget-constrained pavement maintenance simulation engine (`src/simulation/`).

The objective is to evaluate how an automated pavement inspection system (such as the frozen deep segmentation + simplified Pavement Condition Index, **sPCI v1**) interacts with municipal asset management policies under annual capital budget constraints. The simulation models a synthetic road network over a 10-year planning horizon, comparing five distinct maintenance policies (P0 through P4) under both estimated condition data and oracle ground-truth conditions.

> [!IMPORTANT]
> **DISCLAIMER**: All financial costs, treatment costs per square meter, treatment effectiveness thresholds, and deterioration rate parameters in this module are **ILLUSTRATIVE ASSUMPTIONS**. They are established for algorithmic benchmarking, sensitivity analysis, and comparative policy ranking. **No claims are made regarding real municipality accounting, actual tax dollar savings, or physical infrastructure life-cycle guarantees.**

---

## 2. Mathematical Formulation

### 2.1 Synthetic Network Topology
The network consists of $N = 200$ synthetic roadway segments. Each segment $i \in \{0, \dots, N-1\}$ is characterized by:
- Segment ID: $\text{seg\_000}$ to $\text{seg\_199}$
- Length: $L_i \sim \mathcal{U}(100.0, 600.0)\text{ meters}$
- Width: $W_i \sim \mathcal{U}(6.0, 14.0)\text{ meters}$
- Surface Area: $A_i = L_i \times W_i\text{ m}^2$
- Traffic Classification: Categorical variable drawn from:
  - Low traffic (50% probability, priority weight $w_i = 1.0$)
  - Medium traffic (35% probability, priority weight $w_i = 2.0$)
  - High traffic (15% probability, priority weight $w_i = 3.0$)
- Image Tile Sampling: Each segment samples $k = 6$ tiles with replacement from the tile pool (`results/tile_scores.csv`).
  - **Estimated Condition**:
    $$\widehat{\text{PCI}}_{i, 0} = \frac{1}{k} \sum_{j=1}^{k} \text{predicted\_sPCI}_{i, j}$$
  - **True Condition**:
    $$\text{PCI}_{i, 0}^* = \frac{1}{k} \sum_{j=1}^{k} \text{gt\_sPCI}_{i, j}$$
  - **Initial Estimation Error**: $\delta_i = \widehat{\text{PCI}}_{i, 0} - \text{PCI}_{i, 0}^*$

### 2.2 Annual Deterioration Model
Without intervention, pavement condition deteriorates annually according to an acceleration curve based on current PCI:
$$\Delta\text{PCI}_i = r_i \cdot f(\text{PCI}_i^*)$$
where:
- Base deterioration rate $r_i$ follows a log-normal distribution across segments:
  $$r_i \sim \text{Lognormal}(\mu = \ln(3.0), \sigma = 0.30)$$
  yielding a median deterioration rate of $3.0\text{ PCI points/year}$.
- Acceleration factor $f(\text{PCI})$ accelerates deterioration once pavement degrades below fair condition ($\text{PCI} < 70$):
  $$f(\text{PCI}) = \begin{cases} 1.0 + \frac{70.0 - \text{PCI}}{70.0}, & \text{if } \text{PCI} < 70.0 \\ 1.0, & \text{otherwise} \end{cases}$$
- Pavement condition is floored at 0:
  $$\text{PCI}_{t+1}^* = \max(0.0, \text{PCI}_t^* - \Delta\text{PCI}_i)$$

### 2.3 Treatment Options & Effects
Four treatment categories are modeled in relative illustrative cost units per square meter:

| Treatment Type ($T$) | Cost per $\text{m}^2$ ($c_T$) | Eligibility Condition | Pavement Condition Update ($\text{PCI}_{\text{new}}^*$) | Expected Planning Gain ($\Delta \widehat{P}$) |
| :--- | :---: | :--- | :--- | :--- |
| **None** | $0.0$ | Always eligible | $\text{PCI}^*$ (no change) | $0.0$ |
| **Preventive (Crack Seal)** | $1.0$ | $\widehat{\text{PCI}} \ge 55.0$ | $\max(\text{PCI}^*, \min(95.0, \text{PCI}^* + 15.0))$ | $\max(0.0, \min(95.0, \widehat{\text{PCI}} + 15.0) - \widehat{\text{PCI}})$ |
| **Rehabilitation** | $5.0$ (varies: 3, 5, 8) | Always eligible | $\max(\text{PCI}^*, u_i), \quad u_i \sim \mathcal{U}(85.0, 95.0)$ | $\max(0.0, 90.0 - \widehat{\text{PCI}})$ |
| **Reconstruction** | $12.0$ | Always eligible | $100.0$ | $\max(0.0, 100.0 - \widehat{\text{PCI}})$ |

> **Rule of Monotonicity**: A treatment never degrades pavement condition ($\text{PCI}_{\text{new}}^* \ge \text{PCI}^*$).

### 2.4 Annual Budget & Rollover
The annual capital budget is defined as the cost of rehabilitating $X\%$ of the total network surface area:
$$\text{Budget}_{\text{annual}} = c_{\text{rehab}} \times \left(\sum_{i=0}^{N-1} A_i\right) \times \frac{X}{100}$$
where $X \in \{5, 10, 20\}\%$. Unspent budget in any year does **not** roll over to subsequent fiscal years.

### 2.5 Maintenance Allocation Policies (P0 – P4)
In each year $t$, policies evaluate candidate actions based on the observed condition ($\widehat{\text{PCI}}$ in deployment; $\text{PCI}^*$ in oracle mode):

1. **P0 (Do Nothing)**: Zero maintenance interventions; passive deterioration only.
2. **P1 (Reactive / Worst-First Severe)**:
   - Intervenes only on segments in poor condition ($\widehat{\text{PCI}} < 40.0$).
   - If $\widehat{\text{PCI}} < 15.0$: assign **Reconstruction** ($c = 12$).
   - If $15.0 \le \widehat{\text{PCI}} < 40.0$: assign **Rehabilitation** ($c = 5$).
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
     $$\text{Benefit}_{i, T} = \Delta \widehat{P}_{i, T} \times A_i \times w_i$$
     $$\text{Cost}_{i, T} = c_T \times A_i$$
     $$\text{Ratio}_{i, T} = \frac{\text{Benefit}_{i, T}}{\text{Cost}_{i, T}}$$
   - All pairs $(i, T)$ with $\Delta \widehat{P} > 0$ sorted by $\text{Ratio}_{i, T}$ descending.
   - Greedily selects highest benefit/cost candidate, ensuring at most one treatment per segment per year, until budget is exhausted.
5. **P4 (Exact 0/1 Integer Linear Program)**:
   - Solves the exact single-year optimum knapsack allocation:
     $$\max_{x_{i, T} \in \{0, 1\}} \sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Benefit}_{i, T} \cdot x_{i, T}$$
     subject to:
     $$\sum_{T \in \mathcal{T}} x_{i, T} = 1 \quad \forall i \in \{0, \dots, N-1\}$$
     $$\sum_{i=0}^{N-1} \sum_{T \in \mathcal{T}} \text{Cost}_{i, T} \cdot x_{i, T} \le \text{Budget}_{\text{annual}}$$
   - **Solver**: Solved via `scipy.optimize.milp` using the exact HiGHS branch-and-cut simplex backend.

---

## 3. Assumptions and Sensitivity Audit

| Parameter / Assumption | Modeled Value | Rationale / Source | Sensitivity & Impact |
| :--- | :---: | :--- | :--- |
| **Network Size ($N$)** | 200 segments | Sufficient statistical size for Monte Carlo while allowing fast exact MILP execution | Low; scales linearly in total area and budget |
| **Segment Length** | $\mathcal{U}(100, 600)\text{ m}$ | Representative municipal street block lengths | Low; area-weighted normalization makes PCI scale-invariant |
| **Segment Width** | $\mathcal{U}(6, 14)\text{ m}$ | Standard 2-lane to 4-lane urban street cross-sections | Low; preserves realistic segment areas |
| **Traffic Weighting** | 1.0 (Low), 2.0 (Med), 3.0 (High) | Standard municipal prioritization of transit and arterial routes | Moderate; guides greedy and IP priorities toward high-volume corridors |
| **Tile Pool Sampling ($k=6$)** | 6 tiles/segment | Simulates spatial sampling density along an urban street segment | Moderate; higher $k$ reduces tile variance via central limit theorem |
| **Base Deterioration Rate ($r$)** | Median $3.0\text{ pts/yr}$ | Standard asphalt pavement life-cycle assumptions (PCI 100 to 40 in ~15-20 yrs) | High; controls the degradation speed of the network |
| **Deterioration Curve Shape ($f$)** | Linear below PCI 70 | Captures accelerated deterioration due to moisture infiltration and structural cracking | High; penalizes delayed maintenance |
| **Preventive Treatment Cost** | $1.0\text{ cost unit/m}^2$ | Crack seal is an order of magnitude cheaper than structural overlays | High; drives the high benefit-cost ratio of P3 and P4 |
| **Rehabilitation Cost** | $5.0\text{ cost units/m}^2$ | Standard surface mill & asphalt overlay | Tested across $\{3, 5, 8\}$; controls ratio of preventive to rehab volume |
| **Reconstruction Cost** | $12.0\text{ cost units/m}^2$ | Full-depth base repair and repaving | High; makes reactive worst-first policies capital-intensive |
| **Annual Budget ($X$)** | $10\%\text{ of network rehab}$ | Typical municipal maintenance capability | Tested across $\{5, 10, 20\}\%$; dictates backlog accumulation |

---

## 4. Threats to Validity & Architectural Limitations

1. **Synthetic Network & Disjoint Tiles**:
   The 200 segments are synthetic abstractions. The $k=6$ tiles sampled per segment come from public vision datasets (CRACK500, Road Crack, Pothole) collected across different cameras, countries, and angles. They **do not represent real contiguous physical streets**.
2. **Illustrative Unit Costs**:
   The costs ($1, 5, 12$) are dimensionless relative ratios, not dollars or euros. Actual pavement treatment costs vary drastically by region, material availability, labor, and traffic control.
3. **Unvalidated Pothole Rankings**:
   Step 4C demonstrated that while crack sPCI agrees well with human pairwise rankings (90.9%), pothole sPCI achieved only 42.9% agreement (statistically indistinguishable from random chance). Simulations run on the "all" tile pool inherit this uncalibrated pothole measurement noise.
4. **Independent Annual Decision Horizon**:
   P3 and P4 optimize single-year returns greedily or via 1-year integer programming rather than solving a full 10-year dynamic programming / Markov decision process (MDP) optimal control problem.
5. **No Environmental or Climate Stressors**:
   Freeze-thaw cycles, subgrade soil settlement, utility cuts, and heavy axle loads are not physically modeled.

---

## 5. Base Scenario Simulation Results

**Simulation Setup**:
- Network: 200 segments ($668,192.9\text{ m}^2$)
- Tile Pool: `all` ($N = 455$ images)
- Annual Budget: $10\%$ of network rehabilitation cost ($334,096.5\text{ cost units/year}$)
- Cost Ratio: Rehabilitation cost = $5.0\times$ preventive cost
- Monte Carlo: $N = 200$ independent 10-year runs (seeds 1000–1199)
- Decision Making: Policies decide using **Estimated sPCI**; evaluated on **True Ground-Truth PCI**

### 5.1 Primary Performance Metrics (Mean with 95% Bootstrap CI)

| Policy Code & Name | Condition-Years (CY) [95% CI] | Final Network PCI (Year 10) [95% CI] | Share of Area $< 40$ PCI (%) | Efficiency (CY gained / cost unit) |
| :--- | :---: | :---: | :---: | :---: |
| **P0: Do Nothing** | 361.64 [361.34, 361.94] | 16.38 [16.33, 16.42] | 91.73% | 0.0000 [0.0000, 0.0000] |
| **P1: Reactive (<40 PCI)** | 642.80 [642.28, 643.31] | 70.46 [70.38, 70.53] | 2.00% | 0.0001 [0.0001, 0.0001] |
| **P2: Worst-First by Band** | **660.50** [660.22, 660.77] | **72.55** [72.50, 72.59] | **0.02%** | 0.0001 [0.0001, 0.0001] |
| **P3: Greedy Benefit/Cost** | 655.83 [655.60, 656.05] | 68.26 [68.21, 68.31] | 27.14% | 0.0001 [0.0001, 0.0001] |
| **P4: 0/1 Integer Linear Program** | 656.77 [656.54, 656.99] | 68.56 [68.51, 68.61] | 26.58% | 0.0001 [0.0001, 0.0001] |

*(Note: Efficiency units reflect condition-years gained divided by total unweighted illustrative cost units across the 10-year horizon).*

### 5.2 Paired Differences Across Policies (Identical Simulation Seeds)

| Comparison Pair | Paired Difference in Condition-Years ($\Delta \text{CY}$) | 95% Bootstrap CI (10,000 resamples) |
| :--- | :---: | :---: |
| **P1 vs. P0** (Reactive vs. Do Nothing) | **+281.17 CY** | [+280.57, +281.75] |
| **P2 vs. P0** (Band Worst-First vs. Do Nothing) | **+298.86 CY** | [+298.51, +299.20] |
| **P3 vs. P0** (Greedy Benefit/Cost vs. Do Nothing) | **+294.19 CY** | [+293.91, +294.47] |
| **P4 vs. P0** (Exact 0/1 IP vs. Do Nothing) | **+295.13 CY** | [+294.84, +295.42] |
| **P2 vs. P1** (Band Worst-First vs. Reactive) | **+17.69 CY** | [+17.17, +18.23] |
| **P3 vs. P1** (Greedy Benefit/Cost vs. Reactive) | **+13.02 CY** | [+12.47, +13.58] |
| **P4 vs. P1** (Exact 0/1 IP vs. Reactive) | **+13.97 CY** | [+13.42, +14.52] |
| **P4 vs. P3** (Exact 0/1 IP vs. Greedy) | **+0.94 CY** | [+0.81, +1.07] |

---

## 6. Information-Quality Study: Regret Analysis

To determine how observation errors in automated condition assessment degrade maintenance efficacy, each policy was executed twice on identical seeds:
1. **Estimated Deployment**: Policy decides using estimated sPCI $\widehat{\text{PCI}}$, evaluated on True PCI.
2. **Oracle Baseline**: Policy decides using perfect True PCI $\text{PCI}^*$, evaluated on True PCI.

$$\text{Regret} = \text{Oracle Condition-Years} - \text{Estimated Condition-Years}$$

| Policy Code | Oracle True PCI (CY) | Estimated sPCI (CY) | Regret ($\Delta\text{CY}$ Loss) | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **P1: Reactive (<40 PCI)** | 620.61 | 642.80 | **-22.19** | [-22.94, -21.44] |
| **P2: Worst-First by Band** | 665.32 | 660.50 | **+4.82** | [+4.54, +5.11] |
| **P3: Greedy Benefit/Cost** | 719.27 | 655.83 | **+63.44** | [+63.13, +63.75] |
| **P4: 0/1 Integer Linear Program** | 720.34 | 656.77 | **+63.57** | [+63.23, +63.89] |

### Key Insight on Information Regret:
- **Optimization Policies (P3, P4)** suffer substantial regret ($\approx +63.5\text{ CY}$ loss, ~9% penalty). Because greedy and IP policies precisely target segments at the threshold of preventive treatment ($\widehat{\text{PCI}} \ge 55$), errors in sPCI cause treatments to be misallocated to segments whose true condition is already degraded below 55 (where crack sealing is ineffective) or already near 95. When given oracle information, P3 and P4 jump from 656 CY to **720 CY**, demonstrating high sensitivity to measurement quality.
- **P1 Reactive Policy** displays negative apparent regret (Estimated > Oracle) because estimation noise on severely damaged segments causes P1 to treat slightly higher-condition segments earlier, inadvertently providing preventive preservation that the oracle reactive rule strictly ignores.

---

## 7. Full Sensitivity Grid (3 x 3 x 2 = 18 Cells)

A full grid across 18 parameter combinations was evaluated (50 Monte Carlo runs per cell = 4,500 year-long policy evaluations):
- **Annual Budget ($X$)**: $\{5\%, 10\%, 20\%\}$ of network rehabilitation area
- **Cost Ratio ($c_{\text{rehab}} / c_{\text{prev}}$)**: $\{3, 5, 8\}$
- **Tile Pool**: `all` (cracks + potholes) vs. `cracks only` (`crack500` + `road_crack`)

### 7.1 Mean Condition-Years per Policy and Rank 1 Policy

| Tile Pool | Budget (%) | Cost Ratio | P0 (None) | P1 (Reactive) | P2 (Band) | P3 (Greedy) | P4 (0/1 IP) | **Rank 1 Policy** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `all` | 5% | 3 | 361.4 | 526.7 | 500.7 | 508.6 | 510.0 | **P1: Reactive** |
| `all` | 5% | 5 | 361.4 | 528.1 | 510.8 | 533.8 | 532.6 | **P3: Greedy Benefit/Cost** |
| `all` | 5% | 8 | 361.4 | 529.6 | 524.3 | 565.0 | 565.1 | **P4: Integer Programming** |
| `all` | 10% | 3 | 361.4 | 643.9 | 660.3 | 615.0 | 616.1 | **P2: Worst-First by Band** |
| `all` | 10% | 5 | 361.4 | 643.9 | 660.7 | 656.0 | 656.8 | **P2: Worst-First by Band** |
| `all` | 10% | 8 | 361.4 | 643.9 | 661.4 | 733.8 | 736.7 | **P4: Integer Programming** |
| `all` | 20% | 3 | 361.4 | 653.5 | 754.8 | 780.6 | 784.8 | **P4: Integer Programming** |
| `all` | 20% | 5 | 361.4 | 653.5 | 755.3 | 837.6 | 837.7 | **P4: Integer Programming** |
| `all` | 20% | 8 | 361.4 | 653.5 | 756.6 | 870.8 | 869.9 | **P3: Greedy Benefit/Cost** |
| `cracks` | 5% | 3 | 545.2 | 595.1 | 664.4 | 641.2 | 642.8 | **P2: Worst-First by Band** |
| `cracks` | 5% | 5 | 545.2 | 595.1 | 747.3 | 693.0 | 690.3 | **P2: Worst-First by Band** |
| `cracks` | 5% | 8 | 545.2 | 595.1 | 771.2 | 748.7 | 748.5 | **P2: Worst-First by Band** |
| `cracks` | 10% | 3 | 545.2 | 618.2 | 770.3 | 718.8 | 718.0 | **P2: Worst-First by Band** |
| `cracks` | 10% | 5 | 545.2 | 618.2 | 774.8 | 810.0 | 810.6 | **P4: Integer Programming** |
| `cracks` | 10% | 8 | 545.2 | 618.2 | 775.2 | 892.2 | 892.5 | **P4: Integer Programming** |
| `cracks` | 20% | 3 | 545.2 | 625.4 | 775.2 | 867.2 | 862.8 | **P3: Greedy Benefit/Cost** |
| `cracks` | 20% | 5 | 545.2 | 625.4 | 775.2 | 907.2 | 906.0 | **P3: Greedy Benefit/Cost** |
| `cracks` | 20% | 8 | 545.2 | 625.4 | 775.2 | 913.0 | 911.8 | **P3: Greedy Benefit/Cost** |

### 7.2 Analysis of Ranking Transitions
Across the grid, no single policy ranks first in all regimes. The ranking transitions systematically:
1. **Extremely Low Budget + Low Cost Ratio (5% Budget, Ratio 3, `all`)**:
   **P1 (Reactive)** ranks first (526.7 CY). Under severe fiscal restriction, halting acute failure on failing segments ($<15$ and $<40$) is necessary to stop rapid deterioration.
2. **Moderate Budget or Low Distress Backlog (`cracks only` 5% and 10% Ratio 3; `all` 10% Ratios 3–5)**:
   **P2 (Worst-First by Band)** ranks first. By allocating reconstruction to failed segments and rehabilitation to poor segments while preserving fair segments, P2 maintains network balance without suffering from the 1-year myopia of P3/P4.
3. **High Cost Ratio or High Budget (10% Ratio 8, 20% Ratios 3–8)**:
   **P4 (Exact 0/1 IP)** and **P3 (Greedy Benefit/Cost)** dominate (up to 870.8 CY on `all` and 913.0 CY on `cracks`). When the cost ratio is high ($c_{\text{rehab}} = 8$), the budget allows large volumes of cheap preventive maintenance ($c = 1$). Maximizing benefit-cost leverage preserves large areas at high PCI.

---

## 8. Visualizations and Diagnostic Charts

### 8.1 Network Condition Trajectories over Time (10-Year Horizon)
![Condition Trajectories](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_condition_trajectories.png)
*Figure 1: Mean area-weighted True PCI across 10 years for policies P0 through P4 (10% budget, cost ratio 5, tile pool `all`), with 95% Monte Carlo confidence bands. P0 collapses monotonically to ~16 PCI; P2, P3, and P4 stabilize the network above 68–72 PCI.*

### 8.2 Cost-Effectiveness Chart
![Cost Effectiveness](file:///c:/Users/Ved/Desktop/DLbasedRoadCracks/road-pci-intelligence/docs/figures/simulation_cost_effectiveness.png)
*Figure 2: Condition-Years Gained relative to P0 versus total cumulative expenditure over 10 years across policies.*

