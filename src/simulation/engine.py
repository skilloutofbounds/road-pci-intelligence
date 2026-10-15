"""
Pavement Maintenance Simulation Engine (Step 5A / 5B).
Implements stochastic pavement deterioration, multi-category treatments,
budget-constrained resource allocation policies (P0-P4), Monte Carlo benchmarking,
noise-scaled information quality studies, and dual sensitivity analysis grids.

DISCLAIMER: All deterioration rates, treatment costs, and benefits are ILLUSTRATIVE
assumptions and do not reflect real municipality accounting or empirical fiscal savings.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
import yaml

from .network import RoadSegmentNetwork, generate_synthetic_network


def load_simulation_config(config_path: str = "configs/sim_params.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_deterioration(
    current_pci: float,
    rate_r: float,
    inflection_pci: float = 70.0,
) -> float:
    """
    Computes annual deterioration loss:
      dPCI = r * f(PCI), where f(PCI) = 1 + (70 - PCI)/70 for PCI < 70, else 1.
    Floors PCI at 0.0.
    """
    if current_pci < inflection_pci:
        f_pci = 1.0 + (inflection_pci - current_pci) / inflection_pci
    else:
        f_pci = 1.0
    loss = rate_r * f_pci
    return float(max(0.0, current_pci - loss))


def calculate_treatment_outcome(
    current_pci: float,
    treatment_type: str,
    rng: np.random.RandomState,
    rehab_outcome_min: float = 85.0,
    rehab_outcome_max: float = 95.0,
    preventive_mode: str = "cliff",
) -> float:
    """
    Computes updated PCI following treatment.
    Ensures a treatment NEVER reduces PCI (non-decreasing condition).
    Supports 'cliff' (eligible if PCI >= 55) and 'smooth' effectiveness.
    """
    if treatment_type == "none":
        return current_pci
    elif treatment_type == "preventive":
        if preventive_mode == "cliff":
            if current_pci >= 55.0:
                return float(max(current_pci, min(95.0, current_pci + 15.0)))
            return current_pci
        elif preventive_mode == "smooth":
            gain_factor = float(np.clip((current_pci - 40.0) / 30.0, 0.0, 1.0))
            return float(max(current_pci, min(95.0, current_pci + 15.0 * gain_factor)))
        else:
            raise ValueError(f"Unknown preventive_mode: {preventive_mode}")
    elif treatment_type == "rehabilitation":
        outcome = float(rng.uniform(rehab_outcome_min, rehab_outcome_max))
        return float(max(current_pci, outcome))
    elif treatment_type == "reconstruction":
        return float(max(current_pci, 100.0))
    else:
        raise ValueError(f"Unknown treatment_type: {treatment_type}")


def get_expected_gain(
    est_pci: float,
    treatment_type: str,
    rehab_expected_pci: float = 90.0,
    preventive_mode: str = "cliff",
) -> float:
    """
    Computes expected PCI gain used by policy planning:
    gain = E[PCI_new | est_pci] - est_pci.
    """
    if treatment_type == "none":
        return 0.0
    elif treatment_type == "preventive":
        if preventive_mode == "cliff":
            if est_pci >= 55.0:
                new_pci = min(95.0, est_pci + 15.0)
                return float(max(0.0, new_pci - est_pci))
            return 0.0
        elif preventive_mode == "smooth":
            gain_factor = float(np.clip((est_pci - 40.0) / 30.0, 0.0, 1.0))
            new_pci = min(95.0, est_pci + 15.0 * gain_factor)
            return float(max(0.0, new_pci - est_pci))
        else:
            raise ValueError(f"Unknown preventive_mode: {preventive_mode}")
    elif treatment_type == "rehabilitation":
        return float(max(0.0, rehab_expected_pci - est_pci))
    elif treatment_type == "reconstruction":
        return float(max(0.0, 100.0 - est_pci))
    else:
        raise ValueError(f"Unknown treatment_type: {treatment_type}")


class SimulationEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_simulation_config()
        self.cfg = config
        self.median_r = self.cfg["deterioration"]["median_rate"]
        self.sigma_r = self.cfg["deterioration"]["sigma"]
        self.inflection_pci = self.cfg["deterioration"]["inflection_pci"]
        self.horizon_years = self.cfg["budget"]["horizon_years"]

    def run_year_step(
        self,
        current_true_pci: np.ndarray,
        current_est_pci: np.ndarray,
        areas: np.ndarray,
        traffic_weights: np.ndarray,
        annual_budget: float,
        policy_name: str,
        rehab_cost_per_m2: float,
        rng: np.random.RandomState,
        use_oracle: bool = False,
        preventive_mode: str = "cliff",
    ) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
        """
        Executes one year of decisions, treatments, deterioration, and inspection updates.
        Returns: (next_true_pci, next_est_pci, total_spent, assigned_treatments)
        """
        n = len(current_true_pci)
        decision_pci = current_true_pci if use_oracle else current_est_pci

        # Treatment costs per m2
        cost_preventive = float(self.cfg["treatments"]["preventive"]["cost_per_m2"])
        cost_rehab = float(rehab_cost_per_m2)
        cost_reconstruction = float(self.cfg["treatments"]["reconstruction"]["cost_per_m2"])

        treatment_costs_per_m2 = {
            "none": 0.0,
            "preventive": cost_preventive,
            "rehabilitation": cost_rehab,
            "reconstruction": cost_reconstruction,
        }

        # Policy selection
        assigned_treatments = ["none"] * n

        if policy_name == "P0_do_nothing":
            pass

        elif policy_name == "P1_reactive":
            # Intervene only on segments with decision_pci < 40 (worst first)
            eligible = [i for i in range(n) if decision_pci[i] < 40.0]
            eligible.sort(key=lambda i: decision_pci[i])  # worst first
            rem_budget = annual_budget
            for i in eligible:
                t = "reconstruction" if decision_pci[i] < 15.0 else "rehabilitation"
                c = treatment_costs_per_m2[t] * areas[i]
                if c <= rem_budget:
                    assigned_treatments[i] = t
                    rem_budget -= c

        elif policy_name == "P2_worst_first_by_band":
            # Sort all worst first, treatment assigned by band
            all_indices = list(range(n))
            all_indices.sort(key=lambda i: decision_pci[i])
            rem_budget = annual_budget
            for i in all_indices:
                p = decision_pci[i]
                if p <= 25.0:
                    t = "reconstruction"
                elif p <= 55.0:
                    t = "rehabilitation"
                elif p <= 70.0:
                    if preventive_mode == "cliff":
                        t = "preventive" if p >= 55.0 else "none"
                    else:
                        t = "preventive" if p > 40.0 else "none"
                else:
                    t = "none"

                if t != "none":
                    c = treatment_costs_per_m2[t] * areas[i]
                    if c <= rem_budget:
                        assigned_treatments[i] = t
                        rem_budget -= c

        elif policy_name == "P3_greedy_benefit_cost":
            # Benefit = expected PCI gain * area * traffic weight
            candidates = []
            for i in range(n):
                p = decision_pci[i]
                a = areas[i]
                w = traffic_weights[i]
                for t in ["preventive", "rehabilitation", "reconstruction"]:
                    gain = get_expected_gain(p, t, preventive_mode=preventive_mode)
                    if gain > 0.0:
                        benefit = gain * a * w
                        cost = treatment_costs_per_m2[t] * a
                        ratio = benefit / cost
                        candidates.append((i, t, benefit, cost, ratio))

            # Rank by ratio descending
            candidates.sort(key=lambda x: x[4], reverse=True)
            rem_budget = annual_budget
            treated = set()
            for i, t, b, c, r in candidates:
                if i not in treated and c <= rem_budget:
                    assigned_treatments[i] = t
                    treated.add(i)
                    rem_budget -= c

        elif policy_name == "P4_integer_programming":
            # Exact 0/1 IP via scipy.optimize.milp with sparse CSC constraints
            treatments = ["none", "preventive", "rehabilitation", "reconstruction"]
            c_obj = np.zeros(4 * n)
            costs_row = np.zeros(4 * n)
            A_one_per_seg = np.zeros((n, 4 * n))

            for i in range(n):
                p = decision_pci[i]
                a = areas[i]
                w = traffic_weights[i]
                for k, t in enumerate(treatments):
                    var_idx = 4 * i + k
                    A_one_per_seg[i, var_idx] = 1.0
                    cost = treatment_costs_per_m2[t] * a
                    costs_row[var_idx] = cost
                    if t == "none":
                        gain = 0.0
                    else:
                        gain = get_expected_gain(p, t, preventive_mode=preventive_mode)
                    benefit = gain * a * w
                    c_obj[var_idx] = -benefit  # minimize negative benefit

            A_mat = csc_matrix(np.vstack([A_one_per_seg, costs_row]))
            b_l = np.hstack([np.ones(n), 0.0])
            b_u = np.hstack([np.ones(n), annual_budget])

            constraints = LinearConstraint(A_mat, b_l, b_u)
            integrality = np.ones(4 * n)
            bounds = Bounds(0, 1)

            res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                x_sol = np.round(res.x).astype(int)
                for i in range(n):
                    chosen_k = int(np.argmax(x_sol[4 * i : 4 * i + 4]))
                    assigned_treatments[i] = treatments[chosen_k]
            else:
                # Fallback to P3 greedy if solver fails
                return self.run_year_step(
                    current_true_pci, current_est_pci, areas, traffic_weights,
                    annual_budget, "P3_greedy_benefit_cost", rehab_cost_per_m2, rng,
                    use_oracle, preventive_mode
                )
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Execute treatments on TRUE condition and compute REAL spend
        total_spent = 0.0
        post_treatment_true = np.zeros(n)
        for i in range(n):
            t = assigned_treatments[i]
            cost = treatment_costs_per_m2[t] * areas[i]
            total_spent += cost
            post_treatment_true[i] = calculate_treatment_outcome(
                current_true_pci[i], t, rng, preventive_mode=preventive_mode
            )

        # Deterioration step across the year
        mu_r = np.log(self.median_r)
        r_draws = rng.lognormal(mean=mu_r, sigma=self.sigma_r, size=n)

        next_true_pci = np.zeros(n)
        for i in range(n):
            next_true_pci[i] = calculate_deterioration(
                post_treatment_true[i], r_draws[i], inflection_pci=self.inflection_pci
            )

        # Next year inspection observation: true PCI plus segment error
        error_offset = current_est_pci - current_true_pci
        next_est_pci = np.clip(next_true_pci + error_offset, 0.0, 100.0)

        return next_true_pci, next_est_pci, float(total_spent), assigned_treatments


def run_policy_simulation(
    network: RoadSegmentNetwork,
    policy_name: str,
    budget_pct: float = 10.0,
    rehab_cost_ratio: float = 5.0,
    horizon_years: int = 10,
    seed: int = 42,
    use_oracle: bool = False,
    budget_mode: str = "scales_with_rehab",
    preventive_mode: str = "cliff",
    noise_lambda: float = 1.0,
    engine: Optional[SimulationEngine] = None,
) -> Dict[str, Any]:
    """
    Executes a 10-year policy simulation on a road network.
    Tracks both area-weighted and traffic-weighted condition-years, real spending,
    and yearly budget utilization.
    """
    if engine is None:
        engine = SimulationEngine()

    rng = np.random.RandomState(seed)
    df = network.df.copy()
    n = len(df)
    areas = df["area_m2"].values
    total_area = float(np.sum(areas))
    traffic_weights = df["traffic_weight"].values

    current_true = df["true_pci"].values.astype(float).copy()
    if use_oracle or noise_lambda == 0.0:
        current_est = current_true.copy()
    else:
        base_est = df["estimated_pci"].values.astype(float).copy()
        current_est = np.clip(current_true + noise_lambda * (base_est - current_true), 0.0, 100.0)

    # Calculate annual budget based on budget_mode
    if budget_mode == "scales_with_rehab":
        annual_budget = float(rehab_cost_ratio * total_area * (budget_pct / 100.0))
    elif budget_mode == "fixed_absolute":
        annual_budget = float(5.0 * total_area * (budget_pct / 100.0))
    else:
        raise ValueError(f"Unknown budget_mode: {budget_mode}")

    yearly_area_weighted_true_pci = []
    yearly_traffic_weighted_true_pci = []
    yearly_spent = []
    yearly_budget_fraction = []
    yearly_segments_treated = []
    yearly_reconstructions = []

    for year in range(1, horizon_years + 1):
        next_true, next_est, spent, treatments = engine.run_year_step(
            current_true_pci=current_true,
            current_est_pci=current_est,
            areas=areas,
            traffic_weights=traffic_weights,
            annual_budget=annual_budget,
            policy_name=policy_name,
            rehab_cost_per_m2=rehab_cost_ratio,
            rng=rng,
            use_oracle=(use_oracle or noise_lambda == 0.0),
            preventive_mode=preventive_mode,
        )

        aw_true = float(np.average(next_true, weights=areas))
        tw_true = float(np.average(next_true, weights=areas * traffic_weights))

        yearly_area_weighted_true_pci.append(aw_true)
        yearly_traffic_weighted_true_pci.append(tw_true)
        yearly_spent.append(spent)
        yearly_budget_fraction.append(float(spent / annual_budget) if annual_budget > 0 else 0.0)

        n_treated = sum(1 for t in treatments if t != "none")
        n_reconstructed = sum(1 for t in treatments if t == "reconstruction")
        yearly_segments_treated.append(n_treated)
        yearly_reconstructions.append(n_reconstructed)

        current_true = next_true
        current_est = next_est

    # Metrics
    condition_years_aw = float(np.sum(yearly_area_weighted_true_pci))
    condition_years_tw = float(np.sum(yearly_traffic_weighted_true_pci))
    final_network_pci_aw = float(yearly_area_weighted_true_pci[-1])
    final_network_pci_tw = float(yearly_traffic_weighted_true_pci[-1])
    area_below_40 = float(np.sum(areas[current_true < 40.0]))
    share_area_poor = float((area_below_40 / total_area) * 100.0)
    total_spent = float(np.sum(yearly_spent))
    mean_budget_utilization = float(np.mean(yearly_budget_fraction))
    total_segments_treated = int(np.sum(yearly_segments_treated))
    total_reconstructions = int(np.sum(yearly_reconstructions))

    return {
        "policy": policy_name,
        "oracle": use_oracle,
        "budget_pct": budget_pct,
        "rehab_cost_ratio": rehab_cost_ratio,
        "budget_mode": budget_mode,
        "preventive_mode": preventive_mode,
        "noise_lambda": noise_lambda,
        "seed": seed,
        "annual_budget": annual_budget,
        "condition_years_aw": condition_years_aw,
        "condition_years_tw": condition_years_tw,
        "condition_years": condition_years_aw,  # backwards compatibility alias
        "final_network_pci_aw": final_network_pci_aw,
        "final_network_pci_tw": final_network_pci_tw,
        "final_network_pci": final_network_pci_aw,  # backwards compatibility alias
        "share_area_poor_at_10": share_area_poor,
        "total_spent": total_spent,
        "mean_budget_utilization": mean_budget_utilization,
        "total_segments_treated": total_segments_treated,
        "total_reconstructions": total_reconstructions,
        "yearly_trajectory_aw": yearly_area_weighted_true_pci,
        "yearly_trajectory_tw": yearly_traffic_weighted_true_pci,
        "yearly_trajectory": yearly_area_weighted_true_pci,  # backwards compatibility alias
        "yearly_spent": yearly_spent,
        "yearly_budget_fraction": yearly_budget_fraction,
    }


def _simulate_single_run_worker(args: Tuple) -> List[Dict[str, Any]]:
    (
        run_idx, base_seed, network_df, tile_pool_name, policies,
        budget_pct, rehab_cost_ratio, use_oracle, budget_mode,
        preventive_mode, noise_lambda
    ) = args

    net = RoadSegmentNetwork(network_df, tile_pool_name=tile_pool_name)
    engine = SimulationEngine()
    seed = base_seed + run_idx

    p0_res = run_policy_simulation(
        net, "P0_do_nothing", budget_pct=budget_pct, rehab_cost_ratio=rehab_cost_ratio,
        seed=seed, use_oracle=use_oracle, budget_mode=budget_mode,
        preventive_mode=preventive_mode, noise_lambda=noise_lambda, engine=engine
    )
    p0_cy_aw = p0_res["condition_years_aw"]
    p0_cy_tw = p0_res["condition_years_tw"]

    records = []
    for pol in policies:
        if pol == "P0_do_nothing":
            res = p0_res
        else:
            res = run_policy_simulation(
                net, pol, budget_pct=budget_pct, rehab_cost_ratio=rehab_cost_ratio,
                seed=seed, use_oracle=use_oracle, budget_mode=budget_mode,
                preventive_mode=preventive_mode, noise_lambda=noise_lambda, engine=engine
            )

        spent = res["total_spent"]
        cy_gained_aw = res["condition_years_aw"] - p0_cy_aw
        cy_gained_tw = res["condition_years_tw"] - p0_cy_tw

        # Efficiency: condition-years gained per 1,000 cost units
        eff_aw_per_1k = float(cy_gained_aw / (spent / 1000.0)) if spent > 0 else 0.0
        eff_tw_per_1k = float(cy_gained_tw / (spent / 1000.0)) if spent > 0 else 0.0

        records.append({
            "run": run_idx,
            "seed": seed,
            "policy": pol,
            "oracle": use_oracle,
            "budget_pct": budget_pct,
            "rehab_cost_ratio": rehab_cost_ratio,
            "budget_mode": budget_mode,
            "preventive_mode": preventive_mode,
            "noise_lambda": noise_lambda,
            "annual_budget": res["annual_budget"],
            "condition_years_aw": res["condition_years_aw"],
            "condition_years_tw": res["condition_years_tw"],
            "condition_years": res["condition_years_aw"],
            "final_pci_aw": res["final_network_pci_aw"],
            "final_pci_tw": res["final_network_pci_tw"],
            "final_network_pci": res["final_network_pci_aw"],
            "share_area_poor": res["share_area_poor_at_10"],
            "total_spent": spent,
            "mean_budget_utilization": res["mean_budget_utilization"],
            "cy_gained_aw": cy_gained_aw,
            "cy_gained_tw": cy_gained_tw,
            "cy_gained_vs_p0": cy_gained_aw,
            "efficiency_aw_per_1k": eff_aw_per_1k,
            "efficiency_tw_per_1k": eff_tw_per_1k,
            "efficiency": eff_aw_per_1k,
            "total_segments_treated": res["total_segments_treated"],
            "total_reconstructions": res["total_reconstructions"],
        })
    return records


def run_monte_carlo(
    network: RoadSegmentNetwork,
    policies: List[str] = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"],
    budget_pct: float = 10.0,
    rehab_cost_ratio: float = 5.0,
    n_runs: int = 100,
    base_seed: int = 1000,
    use_oracle: bool = False,
    budget_mode: str = "scales_with_rehab",
    preventive_mode: str = "cliff",
    noise_lambda: float = 1.0,
    max_workers: int = 14,
) -> pd.DataFrame:
    """
    Executes N Monte Carlo runs per policy using synchronized seeds for paired comparisons.
    Parallelized across CPU workers via ProcessPoolExecutor.
    """
    tasks = [
        (
            idx, base_seed, network.df, network.tile_pool_name, policies,
            budget_pct, rehab_cost_ratio, use_oracle, budget_mode,
            preventive_mode, noise_lambda
        )
        for idx in range(n_runs)
    ]

    all_records = []
    if max_workers > 1 and n_runs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for recs in executor.map(_simulate_single_run_worker, tasks):
                all_records.extend(recs)
    else:
        for task in tasks:
            all_records.extend(_simulate_single_run_worker(task))

    return pd.DataFrame(all_records)


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(arr)
    boot = [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_paired_diff_ci(arr_a: np.ndarray, arr_b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    diffs = np.array(arr_a) - np.array(arr_b)
    n = len(diffs)
    boot = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_sensitivity_grid_scenario(
    tile_scores_path: str = "results/tile_scores.csv",
    budget_mode: str = "scales_with_rehab",
    n_runs: int = 100,
    base_seed: int = 3000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs an 18-cell sensitivity grid for a specific budget_mode:
    budgets in {5, 10, 20}%, ratios in {3, 5, 8}, tile_pools in {'all', 'cracks'}.
    100 runs per cell with synchronized seeds. Identifies nominal winner and co-leaders.
    """
    policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]
    budgets = [5, 10, 20]
    ratios = [3.0, 5.0, 8.0]
    tile_pools = ["all", "cracks"]

    grid_summary_records = []
    all_runs_records = []

    cell_idx = 0
    total_cells = len(tile_pools) * len(budgets) * len(ratios)

    for pool in tile_pools:
        net = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool=pool, seed=42)
        for b in budgets:
            for r in ratios:
                cell_idx += 1
                print(f"  [{cell_idx}/{total_cells}] [{budget_mode}] pool={pool}, budget={b}%, ratio={int(r)} (N={n_runs})...", flush=True)
                mc_df = run_monte_carlo(
                    network=net,
                    policies=policies,
                    budget_pct=float(b),
                    rehab_cost_ratio=float(r),
                    n_runs=n_runs,
                    base_seed=base_seed,
                    use_oracle=False,
                    budget_mode=budget_mode,
                )
                mc_df["tile_pool"] = pool
                all_runs_records.append(mc_df)

                # Identify best policy and co-leaders based on area-weighted condition-years
                means_aw = mc_df.groupby("policy")["condition_years_aw"].mean().to_dict()
                means_tw = mc_df.groupby("policy")["condition_years_tw"].mean().to_dict()
                best_pol = max(means_aw.items(), key=lambda x: x[1])[0]

                # Paired bootstrap CI vs best
                best_vals_aw = mc_df[mc_df["policy"] == best_pol]["condition_years_aw"].values
                co_leaders_aw = []
                for pol in policies:
                    pol_vals_aw = mc_df[mc_df["policy"] == pol]["condition_years_aw"].values
                    # Paired diff: best - pol
                    m_diff, l_diff, h_diff = bootstrap_paired_diff_ci(best_vals_aw, pol_vals_aw)
                    # If CI contains 0 or pol == best_pol
                    if pol == best_pol or (l_diff <= 0.0 <= h_diff):
                        co_leaders_aw.append(pol)

                # Same for traffic-weighted
                best_pol_tw = max(means_tw.items(), key=lambda x: x[1])[0]
                best_vals_tw = mc_df[mc_df["policy"] == best_pol_tw]["condition_years_tw"].values
                co_leaders_tw = []
                for pol in policies:
                    pol_vals_tw = mc_df[mc_df["policy"] == pol]["condition_years_tw"].values
                    m_diff, l_diff, h_diff = bootstrap_paired_diff_ci(best_vals_tw, pol_vals_tw)
                    if pol == best_pol_tw or (l_diff <= 0.0 <= h_diff):
                        co_leaders_tw.append(pol)

                grid_summary_records.append({
                    "budget_mode": budget_mode,
                    "tile_pool": pool,
                    "budget_pct": b,
                    "rehab_cost_ratio": int(r),
                    "P0_aw": means_aw.get("P0_do_nothing", 0.0),
                    "P1_aw": means_aw.get("P1_reactive", 0.0),
                    "P2_aw": means_aw.get("P2_worst_first_by_band", 0.0),
                    "P3_aw": means_aw.get("P3_greedy_benefit_cost", 0.0),
                    "P4_aw": means_aw.get("P4_integer_programming", 0.0),
                    "P0_tw": means_tw.get("P0_do_nothing", 0.0),
                    "P1_tw": means_tw.get("P1_reactive", 0.0),
                    "P2_tw": means_tw.get("P2_worst_first_by_band", 0.0),
                    "P3_tw": means_tw.get("P3_greedy_benefit_cost", 0.0),
                    "P4_tw": means_tw.get("P4_integer_programming", 0.0),
                    "best_policy_aw": best_pol,
                    "co_leaders_aw": ",".join(co_leaders_aw),
                    "best_policy_tw": best_pol_tw,
                    "co_leaders_tw": ",".join(co_leaders_tw),
                })

    summary_df = pd.DataFrame(grid_summary_records)
    all_runs_df = pd.concat(all_runs_records, ignore_index=True)
    return summary_df, all_runs_df


def run_sensitivity_grid(
    tile_scores_path: str = "results/tile_scores.csv",
    n_runs: int = 100,
    base_seed: int = 3000,
    budget_mode: str = "scales_with_rehab",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for backwards compatibility and single-grid execution."""
    return run_sensitivity_grid_scenario(
        tile_scores_path=tile_scores_path,
        budget_mode=budget_mode,
        n_runs=n_runs,
        base_seed=base_seed,
    )

