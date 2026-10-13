"""
Master Experiment Runner for Step 5A: Maintenance Simulation.
Executes:
1. 200-run Monte Carlo on base scenario (10% budget, rehab ratio 5, tile pool 'all').
2. Information-Quality study (Oracle vs. Estimated decision-making).
3. Full 3x3x2 Sensitivity Grid (Budget {5, 10, 20}% x Ratio {3, 5, 8} x Pool {'all', 'cracks'}).
4. Plots condition trajectories, cost-effectiveness curves, and sensitivity heatmaps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation.network import generate_synthetic_network
from src.simulation.engine import (
    SimulationEngine,
    run_policy_simulation,
    run_monte_carlo,
    run_sensitivity_grid,
)


def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(arr)
    boot = [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_paired_diff_ci(arr_a: np.ndarray, arr_b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.RandomState(seed)
    diffs = np.array(arr_a) - np.array(arr_b)
    n = len(diffs)
    boot = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    tile_scores_path = "results/tile_scores.csv"
    if not Path(tile_scores_path).exists():
        print(f"Error: {tile_scores_path} does not exist yet. Please generate tile_scores.csv first.")
        return

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_res_dir = Path("results")
    out_res_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path("docs/figures")
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 5A: BUDGET-CONSTRAINED MAINTENANCE SIMULATION")
    print("=" * 80)

    # 1. Generate Base Network
    print("\n1. Generating synthetic network (200 segments, tile_pool='all', seed=42)...")
    base_net = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool="all", n_segments=200, seed=42)
    print(f"  * Total segments: {base_net.n_segments}")
    print(f"  * Total network area: {base_net.total_area_m2:,.1f} m2")
    print(f"  * Mean initial Estimated sPCI: {base_net.estimated_network_pci:.2f}")
    print(f"  * Mean initial True sPCI:      {base_net.true_network_pci:.2f}")

    # 2. Base Monte Carlo (N = 200 runs, Estimated decisions)
    policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]
    est_csv_path = out_res_dir / "sim_monte_carlo_base_est.csv"
    if est_csv_path.exists() and len(pd.read_csv(est_csv_path)) == 200 * len(policies):
        print(f"\n2. Loading cached Base Monte Carlo results from {est_csv_path}...")
        mc_est_df = pd.read_csv(est_csv_path)
    else:
        print("\n2. Running Base Monte Carlo (200 runs, Estimated decisions, Budget=10%, Ratio=5)...")
        mc_est_df = run_monte_carlo(
            network=base_net,
            policies=policies,
            budget_pct=10.0,
            rehab_cost_ratio=5.0,
            n_runs=200,
            base_seed=1000,
            use_oracle=False,
        )
        mc_est_df.to_csv(est_csv_path, index=False)
        print(f"  * Saved {est_csv_path}")

    # 3. Base Monte Carlo (N = 200 runs, Oracle decisions)
    oracle_csv_path = out_res_dir / "sim_monte_carlo_base_oracle.csv"
    if oracle_csv_path.exists() and len(pd.read_csv(oracle_csv_path)) == 200 * len(policies):
        print(f"\n3. Loading cached Oracle Monte Carlo results from {oracle_csv_path}...")
        mc_oracle_df = pd.read_csv(oracle_csv_path)
    else:
        print("\n3. Running Oracle Monte Carlo (200 runs, True condition decisions)...")
        mc_oracle_df = run_monte_carlo(
            network=base_net,
            policies=policies,
            budget_pct=10.0,
            rehab_cost_ratio=5.0,
            n_runs=200,
            base_seed=1000,
            use_oracle=True,
        )
        mc_oracle_df.to_csv(oracle_csv_path, index=False)
        print(f"  * Saved {oracle_csv_path}")

    # Print Base Results Table
    print("\n" + "=" * 80)
    print("BASE SCENARIO RESULTS (Budget 10%, Rehab Ratio 5, Tile Pool: All, N = 200 Runs):")
    print("=" * 80)
    print(f"{'Policy':<24} | {'Condition-Years (95% CI)':<26} | {'Final PCI (95% CI)':<22} | {'Area < 40 %':<12} | {'Efficiency (CY/cost)':<20}")
    print("-" * 110)

    for pol in policies:
        sub = mc_est_df[mc_est_df["policy"] == pol]
        cy_m, cy_l, cy_h = bootstrap_mean_ci(sub["condition_years"].values)
        fp_m, fp_l, fp_h = bootstrap_mean_ci(sub["final_network_pci"].values)
        poor_m, _, _ = bootstrap_mean_ci(sub["share_area_poor"].values)
        eff_m, eff_l, eff_h = bootstrap_mean_ci(sub["efficiency"].values)
        print(f"{pol:<24} | {cy_m:6.2f} [{cy_l:6.2f}, {cy_h:6.2f}] | {fp_m:5.2f} [{fp_l:5.2f}, {fp_h:5.2f}] | {poor_m:10.2f}% | {eff_m:6.4f} [{eff_l:6.4f}, {eff_h:6.4f}]")

    # Paired differences vs P0 and vs P1
    print("\n" + "=" * 80)
    print("PAIRED DIFFERENCES IN CONDITION-YEARS (Same Seeds across Runs):")
    print("=" * 80)
    p0_cy = mc_est_df[mc_est_df["policy"] == "P0_do_nothing"]["condition_years"].values
    p1_cy = mc_est_df[mc_est_df["policy"] == "P1_reactive"]["condition_years"].values
    for pol in ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
        pol_cy = mc_est_df[mc_est_df["policy"] == pol]["condition_years"].values
        diff_p0_m, diff_p0_l, diff_p0_h = bootstrap_paired_diff_ci(pol_cy, p0_cy)
        print(f"  * {pol} vs P0: {diff_p0_m:+6.2f} CY [95% CI: {diff_p0_l:+6.2f}, {diff_p0_h:+6.2f}]")
        if pol != "P1_reactive":
            diff_p1_m, diff_p1_l, diff_p1_h = bootstrap_paired_diff_ci(pol_cy, p1_cy)
            print(f"    -> vs P1 (Reactive): {diff_p1_m:+6.2f} CY [95% CI: {diff_p1_l:+6.2f}, {diff_p1_h:+6.2f}]")

    # Information-quality study (Regret)
    print("\n" + "=" * 80)
    print("INFORMATION QUALITY STUDY: REGRET (Oracle True PCI - Estimated sPCI Decisions):")
    print("=" * 80)
    print(f"{'Policy':<24} | {'Oracle CY':<10} | {'Estimated CY':<12} | {'Regret (CY Loss)':<18} | {'95% CI':<20}")
    print("-" * 90)
    for pol in policies:
        if pol == "P0_do_nothing":
            continue
        sub_est = mc_est_df[mc_est_df["policy"] == pol]["condition_years"].values
        sub_orc = mc_oracle_df[mc_oracle_df["policy"] == pol]["condition_years"].values
        reg_m, reg_l, reg_h = bootstrap_paired_diff_ci(sub_orc, sub_est)
        print(f"{pol:<24} | {np.mean(sub_orc):10.2f} | {np.mean(sub_est):12.2f} | {reg_m:+16.2f} | [{reg_l:+6.2f}, {reg_h:+6.2f}]")

    # 4. Sensitivity Grid (3 x 3 x 2 = 18 cells, 50 runs each)
    print("\n4. Running Full Sensitivity Grid (18 scenarios x 50 runs each)...")
    summary_grid_df, all_grid_runs_df = run_sensitivity_grid(
        tile_scores_path=tile_scores_path,
        n_runs=50,
        base_seed=2000,
    )
    summary_grid_df.to_csv(out_res_dir / "sim_sensitivity_grid.csv", index=False)
    print(f"  * Saved {out_res_dir / 'sim_sensitivity_grid.csv'}")

    print("\n" + "=" * 110)
    print("FULL SENSITIVITY GRID SUMMARY (Mean Condition-Years per Policy and Rank 1 Policy):")
    print("=" * 110)
    print(f"{'Pool':<8} | {'Budget %':<9} | {'Ratio':<6} | {'P0':<7} | {'P1':<7} | {'P2':<7} | {'P3':<7} | {'P4':<7} | {'Rank 1 Policy':<22}")
    print("-" * 110)
    for _, r in summary_grid_df.iterrows():
        print(f"{r['tile_pool']:<8} | {r['budget_pct']:9.0f} | {r['rehab_cost_ratio']:6.0f} | {r['P0']:7.1f} | {r['P1']:7.1f} | {r['P2']:7.1f} | {r['P3']:7.1f} | {r['P4']:7.1f} | {r['rank_1_policy']:<22}")
    print("=" * 110)

    # 5. Generate Figures
    print("\n5. Generating Visualization Figures...")
    # Figure 1: Trajectory over time with 95% bands
    fig, ax = plt.subplots(figsize=(10, 6))
    engine = SimulationEngine()
    colors = {"P0_do_nothing": "#7f7f7f", "P1_reactive": "#d62728", "P2_worst_first_by_band": "#ff7f0e", "P3_greedy_benefit_cost": "#1f77b4", "P4_integer_programming": "#2ca02c"}
    labels = {"P0_do_nothing": "P0: Do Nothing", "P1_reactive": "P1: Reactive (<40 PCI)", "P2_worst_first_by_band": "P2: Worst-First by Band", "P3_greedy_benefit_cost": "P3: Greedy Benefit/Cost", "P4_integer_programming": "P4: Integer Programming (0/1 IP)"}

    # Extract yearly trajectories across 200 runs
    for pol in policies:
        pol_trajs = []
        for seed in range(1000, 1050):  # 50 runs for trajectory bands
            res = run_policy_simulation(base_net, pol, budget_pct=10.0, rehab_cost_ratio=5.0, horizon_years=10, seed=seed, engine=engine)
            pol_trajs.append(res["yearly_trajectory"])
        pol_trajs = np.array(pol_trajs)  # shape (50, 10)
        years = np.arange(1, 11)
        mean_traj = np.mean(pol_trajs, axis=0)
        p5_traj = np.percentile(pol_trajs, 2.5, axis=0)
        p95_traj = np.percentile(pol_trajs, 97.5, axis=0)
        ax.plot(years, mean_traj, label=labels[pol], color=colors[pol], linewidth=2.2)
        ax.fill_between(years, p5_traj, p95_traj, color=colors[pol], alpha=0.15)

    ax.set_title("Network Condition Trajectory Over 10-Year Horizon (Budget 10%, 95% MC Bands)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Simulation Year", fontsize=11)
    ax.set_ylabel("Area-Weighted True Network PCI", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 11))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    traj_path = out_fig_dir / "simulation_condition_trajectories.png"
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"  * Saved {traj_path}")

    # Figure 2: Cost-Effectiveness Chart (Condition-Years Gained vs Total Spent)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pol in policies:
        sub = mc_est_df[mc_est_df["policy"] == pol]
        spent_mean = sub["total_spent"].mean()
        cy_gain_mean = sub["cy_gained_vs_p0"].mean()
        ax.scatter(spent_mean, cy_gain_mean, s=150, color=colors[pol], label=labels[pol], edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(pol.split("_")[0], (spent_mean, cy_gain_mean), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=11)

    ax.set_title("Cost-Effectiveness: Condition-Years Gained vs Total Spending", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Expenditure over 10 Years (Illustrative Cost Units)", fontsize=11)
    ax.set_ylabel("Condition-Years Gained Relative to P0 (CY)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    cost_eff_path = out_fig_dir / "simulation_cost_effectiveness.png"
    plt.savefig(cost_eff_path, dpi=200)
    plt.close()
    print(f"  * Saved {cost_eff_path}")

    print("\n[OK] Step 5A Master Simulation successfully completed!")


if __name__ == "__main__":
    main()
