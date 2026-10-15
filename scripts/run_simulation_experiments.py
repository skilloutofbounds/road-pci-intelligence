"""
Master Experiment Runner for Step 5B: Pavement Maintenance Simulation.
Executes:
1. Base Scenario (200 runs): Corrected real spend, budget utilization, both AW and TW metrics,
   efficiency in CY/1,000 cost units (3 sig digits), and P1 at 20% budget.
2. Treatment Model Sensitivity: Smooth preventive effectiveness vs. standard cliff mode.
3. Information Quality Study:
   a. Oracle vs. Estimated for both 'cracks only' and 'all' pools.
   b. Noise scaling lambda in {0, 0.5, 1, 1.5, 2} for both pools.
   c. P1 physical intervention counts: mean treated segments, reconstructions, and spend.
4. Dual Sensitivity Grids (100 runs per cell):
   - Grid 1: Budget scales with rehab cost (18 cells).
   - Grid 2: Fixed absolute budget (18 cells).
   - Co-leaders identified via paired bootstrap difference 95% CIs.
   - Win and co-win tallies for both grids.
5. Figure generation and saving into docs/figures/:
   - Condition trajectories with 95% MC bands.
   - Cost-effectiveness with corrected spend.
   - Sensitivity heatmaps (scaling and fixed budget).
   - Regret vs. lambda curves.
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
    run_sensitivity_grid_scenario,
    bootstrap_ci,
    bootstrap_paired_diff_ci,
)


def fmt_3sig(val: float) -> str:
    if abs(val) < 1e-12:
        return "0.00"
    return f"{val:.3g}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tile_scores_path = "results/tile_scores.csv"
    if not Path(tile_scores_path).exists():
        print(f"Error: {tile_scores_path} does not exist yet.")
        return

    out_res_dir = Path("results")
    out_res_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path("docs/figures")
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("STEP 5B: REFINED BUDGET-CONSTRAINED MAINTENANCE SIMULATION")
    print("=" * 100)

    # 1. Generate Synthetic Networks
    print("\n1. Initializing synthetic road networks (200 segments, seed=42)...")
    base_net_all = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool="all", n_segments=200, seed=42)
    base_net_cracks = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool="cracks", n_segments=200, seed=42)
    print(f"  * Total network area: {base_net_all.total_area_m2:,.1f} m2 across {base_net_all.n_segments} segments")
    print(f"  * Pool 'all':    initial Est sPCI = {base_net_all.estimated_network_pci:.2f}, True sPCI = {base_net_all.true_network_pci:.2f}")
    print(f"  * Pool 'cracks': initial Est sPCI = {base_net_cracks.estimated_network_pci:.2f}, True sPCI = {base_net_cracks.true_network_pci:.2f}")

    # =========================================================================
    # 2. BASE SCENARIO (200 runs, Budget=10%, Ratio=5.0, Pool='all')
    # =========================================================================
    print("\n2. Executing Base Scenario (N = 200 runs, Estimated decisions, Pool='all')...", flush=True)
    policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]
    
    est_path = out_res_dir / "sim_monte_carlo_base_est.csv"
    if est_path.exists() and len(pd.read_csv(est_path)) == 200 * len(policies):
        print(f"  * Loading cached base estimated results from {est_path}...", flush=True)
        mc_base_est = pd.read_csv(est_path)
    else:
        mc_base_est = run_monte_carlo(
            network=base_net_all,
            policies=policies,
            budget_pct=10.0,
            rehab_cost_ratio=5.0,
            n_runs=200,
            base_seed=1000,
            use_oracle=False,
        )
        mc_base_est.to_csv(est_path, index=False)
        print(f"  * Saved {est_path}", flush=True)

    # Base Oracle run (200 runs)
    orc_path = out_res_dir / "sim_monte_carlo_base_oracle.csv"
    if orc_path.exists() and len(pd.read_csv(orc_path)) == 200 * len(policies):
        print(f"  * Loading cached base oracle results from {orc_path}...", flush=True)
        mc_base_oracle = pd.read_csv(orc_path)
    else:
        print("  * Executing Base Scenario (N = 200 runs, Oracle decisions)...", flush=True)
        mc_base_oracle = run_monte_carlo(
            network=base_net_all,
            policies=policies,
            budget_pct=10.0,
            rehab_cost_ratio=5.0,
            n_runs=200,
            base_seed=1000,
            use_oracle=True,
        )
        mc_base_oracle.to_csv(orc_path, index=False)
        print(f"  * Saved {orc_path}", flush=True)

    # P1 at 20% budget (200 runs) to report spend and budget utilization
    p1_b20_path = out_res_dir / "sim_p1_b20.csv"
    if p1_b20_path.exists() and len(pd.read_csv(p1_b20_path)) == 200:
        print(f"  * Loading cached P1 at 20% budget from {p1_b20_path}...", flush=True)
        mc_p1_b20 = pd.read_csv(p1_b20_path)
    else:
        print("  * Evaluating P1 at 20% budget (N = 200 runs)...", flush=True)
        mc_p1_b20 = run_monte_carlo(
            network=base_net_all,
            policies=["P1_reactive"],
            budget_pct=20.0,
            rehab_cost_ratio=5.0,
            n_runs=200,
            base_seed=1000,
            use_oracle=False,
        )
        mc_p1_b20.to_csv(p1_b20_path, index=False)

    # Print Base Scenario Results Table
    print("\n" + "=" * 125)
    print("BASE SCENARIO RESULTS (Budget 10%, Rehab Ratio 5.0, Pool: All, N = 200 Runs):")
    print("=" * 125)
    print(f"{'Policy':<24} | {'Condition-Years (AW) [95% CI]':<30} | {'Condition-Years (TW) [95% CI]':<30} | {'Real Spend [95% CI]':<26} | {'Util %':<8} | {'Eff AW (CY/1k)':<15} | {'Eff TW (CY/1k)':<15}")
    print("-" * 155)

    for pol in policies:
        sub = mc_base_est[mc_base_est["policy"] == pol]
        cy_aw_m, cy_aw_l, cy_aw_h = bootstrap_ci(sub["condition_years_aw"].values)
        cy_tw_m, cy_tw_l, cy_tw_h = bootstrap_ci(sub["condition_years_tw"].values)
        sp_m, sp_l, sp_h = bootstrap_ci(sub["total_spent"].values)
        ut_m, _, _ = bootstrap_ci(sub["mean_budget_utilization"].values * 100.0)
        eff_aw_m, eff_aw_l, eff_aw_h = bootstrap_ci(sub["efficiency_aw_per_1k"].values)
        eff_tw_m, eff_tw_l, eff_tw_h = bootstrap_ci(sub["efficiency_tw_per_1k"].values)

        eff_aw_str = f"{fmt_3sig(eff_aw_m)} [{fmt_3sig(eff_aw_l)}, {fmt_3sig(eff_aw_h)}]"
        eff_tw_str = f"{fmt_3sig(eff_tw_m)} [{fmt_3sig(eff_tw_l)}, {fmt_3sig(eff_tw_h)}]"
        print(f"{pol:<24} | {cy_aw_m:6.2f} [{cy_aw_l:6.2f}, {cy_aw_h:6.2f}]         | {cy_tw_m:6.2f} [{cy_tw_l:6.2f}, {cy_tw_h:6.2f}]         | {sp_m:9,.0f} [{sp_l:9,.0f}, {sp_h:9,.0f}] | {ut_m:6.2f}% | {eff_aw_str:<15} | {eff_tw_str:<15}")

    # P1 at 20% budget summary
    p1_b20_spent_m, p1_b20_spent_l, p1_b20_spent_h = bootstrap_ci(mc_p1_b20["total_spent"].values)
    p1_b20_ut_m, _, _ = bootstrap_ci(mc_p1_b20["mean_budget_utilization"].values * 100.0)
    p1_b20_cy_aw_m, _, _ = bootstrap_ci(mc_p1_b20["condition_years_aw"].values)
    p1_b20_cy_tw_m, _, _ = bootstrap_ci(mc_p1_b20["condition_years_tw"].values)
    print(f"\n[P1 at 20% Budget Check (10-yr Budget = {5.0 * base_net_all.total_area_m2 * 0.20 * 10:,.0f})]:")
    print(f"  * Real Spend: {p1_b20_spent_m:,.0f} [95% CI: {p1_b20_spent_l:,.0f}, {p1_b20_spent_h:,.0f}]")
    print(f"  * Budget Utilization: {p1_b20_ut_m:.2f}% (P1 exhausts eligible segments with PCI < 40 and leaves substantial budget unspent)")
    print(f"  * Condition-Years: AW = {p1_b20_cy_aw_m:.2f}, TW = {p1_b20_cy_tw_m:.2f}")

    # Paired differences vs P0 and vs P1 (Both AW and TW)
    print("\n" + "=" * 100)
    print("PAIRED DIFFERENCES (Identical Simulation Seeds Across Runs):")
    print("=" * 100)
    p0_aw = mc_base_est[mc_base_est["policy"] == "P0_do_nothing"]["condition_years_aw"].values
    p0_tw = mc_base_est[mc_base_est["policy"] == "P0_do_nothing"]["condition_years_tw"].values
    p1_aw = mc_base_est[mc_base_est["policy"] == "P1_reactive"]["condition_years_aw"].values
    p1_tw = mc_base_est[mc_base_est["policy"] == "P1_reactive"]["condition_years_tw"].values

    for pol in ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
        pol_aw = mc_base_est[mc_base_est["policy"] == pol]["condition_years_aw"].values
        pol_tw = mc_base_est[mc_base_est["policy"] == pol]["condition_years_tw"].values

        d_p0_aw_m, d_p0_aw_l, d_p0_aw_h = bootstrap_paired_diff_ci(pol_aw, p0_aw)
        d_p0_tw_m, d_p0_tw_l, d_p0_tw_h = bootstrap_paired_diff_ci(pol_tw, p0_tw)
        print(f"  * {pol} vs P0:")
        print(f"      Area-Weighted:    {d_p0_aw_m:+6.2f} CY [95% CI: {d_p0_aw_l:+6.2f}, {d_p0_aw_h:+6.2f}]")
        print(f"      Traffic-Weighted: {d_p0_tw_m:+6.2f} CY [95% CI: {d_p0_tw_l:+6.2f}, {d_p0_tw_h:+6.2f}]")

        if pol != "P1_reactive":
            d_p1_aw_m, d_p1_aw_l, d_p1_aw_h = bootstrap_paired_diff_ci(pol_aw, p1_aw)
            d_p1_tw_m, d_p1_tw_l, d_p1_tw_h = bootstrap_paired_diff_ci(pol_tw, p1_tw)
            print(f"      -> vs P1 (Reactive): AW = {d_p1_aw_m:+6.2f} [{d_p1_aw_l:+6.2f}, {d_p1_aw_h:+6.2f}] | TW = {d_p1_tw_m:+6.2f} [{d_p1_tw_l:+6.2f}, {d_p1_tw_h:+6.2f}]")

    # =========================================================================
    # 3. TREATMENT-MODEL SENSITIVITY: SMOOTH PREVENTIVE EFFECTIVENESS
    # =========================================================================
    print("\n" + "=" * 100)
    print("3. TREATMENT-MODEL SENSITIVITY: SMOOTH PREVENTIVE EFFECTIVENESS")
    print("=" * 100)
    print("Evaluating smooth effectiveness: gain factor = clip((TRUE PCI - 40) / 30, 0, 1) on both pools (100 runs)...")
    
    mc_smooth_all_est = run_monte_carlo(base_net_all, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, use_oracle=False, preventive_mode="smooth")
    mc_smooth_all_orc = run_monte_carlo(base_net_all, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, use_oracle=True, preventive_mode="smooth")
    mc_smooth_crk_est = run_monte_carlo(base_net_cracks, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, use_oracle=False, preventive_mode="smooth")
    mc_smooth_crk_orc = run_monte_carlo(base_net_cracks, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, use_oracle=True, preventive_mode="smooth")

    print(f"\n[Smooth Preventive Results - Pool 'all' (Budget 10%, Ratio 5.0)]:")
    print(f"{'Policy':<24} | {'Est CY (AW)':<12} | {'Orc CY (AW)':<12} | {'Regret AW [95% CI]':<26} | {'Rank (Est)':<10}")
    print("-" * 90)
    means_smooth_all = mc_smooth_all_est.groupby("policy")["condition_years_aw"].mean().to_dict()
    ranked_smooth_all = sorted(means_smooth_all.items(), key=lambda x: x[1], reverse=True)
    rank_map_smooth_all = {p: i+1 for i, (p, _) in enumerate(ranked_smooth_all)}
    for pol in policies:
        est_vals = mc_smooth_all_est[mc_smooth_all_est["policy"] == pol]["condition_years_aw"].values
        orc_vals = mc_smooth_all_orc[mc_smooth_all_orc["policy"] == pol]["condition_years_aw"].values
        reg_m, reg_l, reg_h = bootstrap_paired_diff_ci(orc_vals, est_vals)
        print(f"{pol:<24} | {np.mean(est_vals):12.2f} | {np.mean(orc_vals):12.2f} | {reg_m:+6.2f} [{reg_l:+6.2f}, {reg_h:+6.2f}] | Rank {rank_map_smooth_all[pol]}")

    print(f"\n[Smooth Preventive Results - Pool 'cracks' (Budget 10%, Ratio 5.0)]:")
    print(f"{'Policy':<24} | {'Est CY (AW)':<12} | {'Orc CY (AW)':<12} | {'Regret AW [95% CI]':<26} | {'Rank (Est)':<10}")
    print("-" * 90)
    means_smooth_crk = mc_smooth_crk_est.groupby("policy")["condition_years_aw"].mean().to_dict()
    ranked_smooth_crk = sorted(means_smooth_crk.items(), key=lambda x: x[1], reverse=True)
    rank_map_smooth_crk = {p: i+1 for i, (p, _) in enumerate(ranked_smooth_crk)}
    for pol in policies:
        est_vals = mc_smooth_crk_est[mc_smooth_crk_est["policy"] == pol]["condition_years_aw"].values
        orc_vals = mc_smooth_crk_orc[mc_smooth_crk_orc["policy"] == pol]["condition_years_aw"].values
        reg_m, reg_l, reg_h = bootstrap_paired_diff_ci(orc_vals, est_vals)
        print(f"{pol:<24} | {np.mean(est_vals):12.2f} | {np.mean(orc_vals):12.2f} | {reg_m:+6.2f} [{reg_l:+6.2f}, {reg_h:+6.2f}] | Rank {rank_map_smooth_crk[pol]}")

    # =========================================================================
    # 4. INFORMATION QUALITY STUDY (REDONE)
    # =========================================================================
    print("\n" + "=" * 100)
    print("4. INFORMATION QUALITY STUDY (REDONE)")
    print("=" * 100)

    # 4a. Oracle vs Estimated on Cracks-Only vs All
    print("\n--- 4a. Oracle vs. Estimated across Tile Pools (Budget 10%, Ratio 5.0, 200 runs) ---")
    mc_crk_est_200 = run_monte_carlo(base_net_cracks, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=200, base_seed=1000, use_oracle=False)
    mc_crk_orc_200 = run_monte_carlo(base_net_cracks, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=200, base_seed=1000, use_oracle=True)

    print("\n[Pool 'cracks only' - Cracks ground truth contains full crack channels]:")
    print(f"{'Policy':<24} | {'Oracle AW':<10} | {'Est AW':<10} | {'Regret AW [95% CI]':<24} | {'Regret TW [95% CI]':<24}")
    print("-" * 100)
    for pol in policies:
        if pol == "P0_do_nothing":
            continue
        e_aw = mc_crk_est_200[mc_crk_est_200["policy"] == pol]["condition_years_aw"].values
        o_aw = mc_crk_orc_200[mc_crk_orc_200["policy"] == pol]["condition_years_aw"].values
        e_tw = mc_crk_est_200[mc_crk_est_200["policy"] == pol]["condition_years_tw"].values
        o_tw = mc_crk_orc_200[mc_crk_orc_200["policy"] == pol]["condition_years_tw"].values

        reg_aw_m, reg_aw_l, reg_aw_h = bootstrap_paired_diff_ci(o_aw, e_aw)
        reg_tw_m, reg_tw_l, reg_tw_h = bootstrap_paired_diff_ci(o_tw, e_tw)
        print(f"{pol:<24} | {np.mean(o_aw):10.2f} | {np.mean(e_aw):10.2f} | {reg_aw_m:+6.2f} [{reg_aw_l:+6.2f}, {reg_aw_h:+6.2f}] | {reg_tw_m:+6.2f} [{reg_tw_l:+6.2f}, {reg_tw_h:+6.2f}]")

    print("\n[Pool 'all' - NOTE: Ground truth for pothole tiles has no crack channel]:")
    print(f"{'Policy':<24} | {'Oracle AW':<10} | {'Est AW':<10} | {'Regret AW [95% CI]':<24} | {'Regret TW [95% CI]':<24}")
    print("-" * 100)
    for pol in policies:
        if pol == "P0_do_nothing":
            continue
        e_aw = mc_base_est[mc_base_est["policy"] == pol]["condition_years_aw"].values
        o_aw = mc_base_oracle[mc_base_oracle["policy"] == pol]["condition_years_aw"].values
        e_tw = mc_base_est[mc_base_est["policy"] == pol]["condition_years_tw"].values
        o_tw = mc_base_oracle[mc_base_oracle["policy"] == pol]["condition_years_tw"].values

        reg_aw_m, reg_aw_l, reg_aw_h = bootstrap_paired_diff_ci(o_aw, e_aw)
        reg_tw_m, reg_tw_l, reg_tw_h = bootstrap_paired_diff_ci(o_tw, e_tw)
        print(f"{pol:<24} | {np.mean(o_aw):10.2f} | {np.mean(e_aw):10.2f} | {reg_aw_m:+6.2f} [{reg_aw_l:+6.2f}, {reg_aw_h:+6.2f}] | {reg_tw_m:+6.2f} [{reg_tw_l:+6.2f}, {reg_tw_h:+6.2f}]")

    # 4b. Noise Scaling: lambda in {0, 0.5, 1, 1.5, 2}
    print("\n--- 4b. Noise Scaling Analysis: estimate = true + lambda * (predicted - true) ---")
    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]
    noise_records = []

    for pool_name, net_obj in [("all", base_net_all), ("cracks", base_net_cracks)]:
        # Oracle baseline for lambda=0
        mc_lam0 = run_monte_carlo(net_obj, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, noise_lambda=0.0)
        for lam in lambdas:
            if lam == 0.0:
                mc_lam = mc_lam0
            else:
                mc_lam = run_monte_carlo(net_obj, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=1000, noise_lambda=lam)
            
            for pol in ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
                o_aw = mc_lam0[mc_lam0["policy"] == pol]["condition_years_aw"].values
                l_aw = mc_lam[mc_lam["policy"] == pol]["condition_years_aw"].values
                o_tw = mc_lam0[mc_lam0["policy"] == pol]["condition_years_tw"].values
                l_tw = mc_lam[mc_lam["policy"] == pol]["condition_years_tw"].values

                reg_aw_m, reg_aw_l, reg_aw_h = bootstrap_paired_diff_ci(o_aw, l_aw)
                reg_tw_m, reg_tw_l, reg_tw_h = bootstrap_paired_diff_ci(o_tw, l_tw)

                noise_records.append({
                    "tile_pool": pool_name,
                    "lambda": lam,
                    "policy": pol,
                    "regret_aw_mean": reg_aw_m,
                    "regret_aw_low": reg_aw_l,
                    "regret_aw_high": reg_aw_h,
                    "regret_tw_mean": reg_tw_m,
                    "regret_tw_low": reg_tw_l,
                    "regret_tw_high": reg_tw_h,
                })

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(out_res_dir / "sim_regret_vs_lambda.csv", index=False)
    print(f"  * Saved {out_res_dir / 'sim_regret_vs_lambda.csv'}")

    print("\n[Regret vs. Noise Lambda Summary Table]:")
    print(f"{'Pool':<8} | {'Lambda':<6} | {'Policy':<24} | {'Regret AW [95% CI]':<26} | {'Regret TW [95% CI]':<26}")
    print("-" * 95)
    for _, r in noise_df.iterrows():
        reg_aw_str = f"{r['regret_aw_mean']:+6.2f} [{r['regret_aw_low']:+6.2f}, {r['regret_aw_high']:+6.2f}]"
        reg_tw_str = f"{r['regret_tw_mean']:+6.2f} [{r['regret_tw_low']:+6.2f}, {r['regret_tw_high']:+6.2f}]"
        print(f"{r['tile_pool']:<8} | {r['lambda']:<6.1f} | {r['policy']:<24} | {reg_aw_str:<26} | {reg_tw_str:<26}")

    # 4c. P1 Detailed Accounting: Treated Segments, Reconstructions, and Spend (Oracle vs Estimated)
    print("\n--- 4c. P1 Physical Intervention Accounting (Mean over 10 Years, Numbers Only) ---")
    p1_est = mc_base_est[mc_base_est["policy"] == "P1_reactive"]
    p1_orc = mc_base_oracle[mc_base_oracle["policy"] == "P1_reactive"]

    print(f"{'Metric':<35} | {'Oracle':<15} | {'Estimated':<15}")
    print("-" * 70)
    print(f"{'Mean Total Segments Treated (10-yr)':<35} | {p1_orc['total_segments_treated'].mean():15.1f} | {p1_est['total_segments_treated'].mean():15.1f}")
    print(f"{'Mean Total Reconstructions (10-yr)':<35} | {p1_orc['total_reconstructions'].mean():15.1f} | {p1_est['total_reconstructions'].mean():15.1f}")
    print(f"{'Mean Total Spend (10-yr)':<35} | {p1_orc['total_spent'].mean():15,.1f} | {p1_est['total_spent'].mean():15,.1f}")

    # =========================================================================
    # 5. DUAL SENSITIVITY GRIDS (100 runs per cell)
    # =========================================================================
    print("\n" + "=" * 100)
    print("5. DUAL SENSITIVITY GRIDS (100 runs per cell)")
    print("=" * 100)

    print("\n5a. Running Grid 1: 'Budget Scales with Rehab Cost' (18 cells x 100 runs)...")
    grid_scale_summary, grid_scale_runs = run_sensitivity_grid_scenario(
        tile_scores_path=tile_scores_path,
        budget_mode="scales_with_rehab",
        n_runs=100,
        base_seed=3000,
    )
    grid_scale_summary.to_csv(out_res_dir / "sim_sensitivity_grid_scaling.csv", index=False)
    print(f"  * Saved {out_res_dir / 'sim_sensitivity_grid_scaling.csv'}")

    print("\n5b. Running Grid 2: 'Fixed Absolute Budget' (18 cells x 100 runs)...")
    grid_fixed_summary, grid_fixed_runs = run_sensitivity_grid_scenario(
        tile_scores_path=tile_scores_path,
        budget_mode="fixed_absolute",
        n_runs=100,
        base_seed=3000,
    )
    grid_fixed_summary.to_csv(out_res_dir / "sim_sensitivity_grid_fixed.csv", index=False)
    print(f"  * Saved {out_res_dir / 'sim_sensitivity_grid_fixed.csv'}")

    # Print Grid 1 Table
    print("\n" + "=" * 135)
    print("GRID 1: BUDGET SCALES WITH REHAB COST (Mean Area-Weighted CY, Winner & Co-Leaders Marked):")
    print("=" * 135)
    print(f"{'Pool':<8} | {'Budget %':<9} | {'Ratio':<6} | {'P0':<7} | {'P1':<7} | {'P2':<7} | {'P3':<7} | {'P4':<7} | {'Winner (AW)':<22} | {'Co-Leaders (95% CI Contains 0)':<30}")
    print("-" * 135)
    for _, r in grid_scale_summary.iterrows():
        print(f"{r['tile_pool']:<8} | {r['budget_pct']:9.0f} | {r['rehab_cost_ratio']:6.0f} | {r['P0_aw']:7.1f} | {r['P1_aw']:7.1f} | {r['P2_aw']:7.1f} | {r['P3_aw']:7.1f} | {r['P4_aw']:7.1f} | {r['best_policy_aw']:<22} | {r['co_leaders_aw']:<30}")

    # Print Grid 2 Table
    print("\n" + "=" * 135)
    print("GRID 2: FIXED ABSOLUTE BUDGET (Mean Area-Weighted CY, Winner & Co-Leaders Marked):")
    print("=" * 135)
    print(f"{'Pool':<8} | {'Budget %':<9} | {'Ratio':<6} | {'P0':<7} | {'P1':<7} | {'P2':<7} | {'P3':<7} | {'P4':<7} | {'Winner (AW)':<22} | {'Co-Leaders (95% CI Contains 0)':<30}")
    print("-" * 135)
    for _, r in grid_fixed_summary.iterrows():
        print(f"{r['tile_pool']:<8} | {r['budget_pct']:9.0f} | {r['rehab_cost_ratio']:6.0f} | {r['P0_aw']:7.1f} | {r['P1_aw']:7.1f} | {r['P2_aw']:7.1f} | {r['P3_aw']:7.1f} | {r['P4_aw']:7.1f} | {r['best_policy_aw']:<22} | {r['co_leaders_aw']:<30}")

    # Tally of won and co-won cells
    print("\n" + "=" * 80)
    print("POLICY VICTORY AND CO-LEADER TALLIES (18 Cells per Grid):")
    print("=" * 80)
    print(f"{'Policy':<24} | {'Grid 1 Won (Nominal)':<20} | {'Grid 1 Co-Won':<15} | {'Grid 2 Won (Nominal)':<20} | {'Grid 2 Co-Won':<15}")
    print("-" * 100)

    for pol in policies:
        g1_won = sum(grid_scale_summary["best_policy_aw"] == pol)
        g1_cowon = sum(grid_scale_summary["co_leaders_aw"].apply(lambda s: pol in s.split(",")))
        g2_won = sum(grid_fixed_summary["best_policy_aw"] == pol)
        g2_cowon = sum(grid_fixed_summary["co_leaders_aw"].apply(lambda s: pol in s.split(",")))
        print(f"{pol:<24} | {g1_won:<20} | {g1_cowon:<15} | {g2_won:<20} | {g2_cowon:<15}")

    # =========================================================================
    # 6. GENERATE AND SAVE VISUALIZATION FIGURES
    # =========================================================================
    print("\n6. Generating and rendering visualization figures in docs/figures/...")
    engine = SimulationEngine()
    colors = {
        "P0_do_nothing": "#7f7f7f",
        "P1_reactive": "#d62728",
        "P2_worst_first_by_band": "#ff7f0e",
        "P3_greedy_benefit_cost": "#1f77b4",
        "P4_integer_programming": "#2ca02c",
    }
    labels = {
        "P0_do_nothing": "P0: Do Nothing",
        "P1_reactive": "P1: Reactive (<40 PCI)",
        "P2_worst_first_by_band": "P2: Worst-First by Band",
        "P3_greedy_benefit_cost": "P3: Greedy Benefit/Cost",
        "P4_integer_programming": "P4: 0/1 Integer Programming",
    }

    # Figure 1: Condition Trajectories over Time with 95% MC Bands (Both AW and TW)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for pol in policies:
        sub = mc_base_est[mc_base_est["policy"] == pol]
        # Extract trajectories across seeds 1000..1049 (50 runs)
        trajs_aw = []
        trajs_tw = []
        for s in range(1000, 1050):
            res = run_policy_simulation(base_net_all, pol, budget_pct=10.0, rehab_cost_ratio=5.0, seed=s, engine=engine)
            trajs_aw.append(res["yearly_trajectory_aw"])
            trajs_tw.append(res["yearly_trajectory_tw"])

        trajs_aw = np.array(trajs_aw)
        trajs_tw = np.array(trajs_tw)
        years = np.arange(1, 11)

        # Plot AW
        m_aw = np.mean(trajs_aw, axis=0)
        p5_aw = np.percentile(trajs_aw, 2.5, axis=0)
        p95_aw = np.percentile(trajs_aw, 97.5, axis=0)
        ax1.plot(years, m_aw, color=colors[pol], label=labels[pol], linewidth=2.0)
        ax1.fill_between(years, p5_aw, p95_aw, color=colors[pol], alpha=0.15)

        # Plot TW
        m_tw = np.mean(trajs_tw, axis=0)
        p5_tw = np.percentile(trajs_tw, 2.5, axis=0)
        p95_tw = np.percentile(trajs_tw, 97.5, axis=0)
        ax2.plot(years, m_tw, color=colors[pol], label=labels[pol], linewidth=2.0)
        ax2.fill_between(years, p5_tw, p95_tw, color=colors[pol], alpha=0.15)

    ax1.set_title("Area-Weighted Network Condition (95% MC Bands)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("True PCI (Area-Weighted)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8.5)

    ax2.set_title("Traffic-Weighted Network Condition (95% MC Bands)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("True PCI (Traffic-Weighted)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=8.5)

    plt.suptitle("10-Year Network Condition Trajectories (Budget 10%, Rehab Ratio 5.0, Pool 'all')", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    traj_path = out_fig_dir / "simulation_condition_trajectories.png"
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"  * Saved {traj_path}")

    # Figure 2: Cost-Effectiveness (CY Gained vs. Real Spend)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pol in policies:
        sub = mc_base_est[mc_base_est["policy"] == pol]
        sp_m = sub["total_spent"].mean()
        gain_m = sub["cy_gained_aw"].mean()
        ax.scatter(sp_m, gain_m, s=150, color=colors[pol], label=labels[pol], edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(pol.split("_")[0], (sp_m, gain_m), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=11)

    ax.set_title("Cost-Effectiveness: Condition-Years Gained vs. Real Spend", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Real Expenditure Over 10 Years (Illustrative Cost Units)")
    ax.set_ylabel("Condition-Years Gained Relative to P0 (Area-Weighted)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    cost_eff_path = out_fig_dir / "simulation_cost_effectiveness.png"
    plt.savefig(cost_eff_path, dpi=150)
    plt.close()
    print(f"  * Saved {cost_eff_path}")

    # Figure 3 & 4: Sensitivity Heatmaps (Scaling Budget & Fixed Budget)
    def plot_grid_heatmap(grid_df: pd.DataFrame, title: str, out_path: Path):
        fig, (ax_all, ax_crk) = plt.subplots(1, 2, figsize=(14, 5))
        pol_code_map = {"P0_do_nothing": 0, "P1_reactive": 1, "P2_worst_first_by_band": 2, "P3_greedy_benefit_cost": 3, "P4_integer_programming": 4}
        cmap = plt.cm.get_cmap("tab10", 5)

        for ax, pool_name in [(ax_all, "all"), (ax_crk, "cracks")]:
            sub = grid_df[grid_df["tile_pool"] == pool_name]
            pivot = sub.pivot(index="budget_pct", columns="rehab_cost_ratio", values="best_policy_aw")
            num_matrix = np.zeros(pivot.shape)
            ratios = sorted(sub["rehab_cost_ratio"].unique())
            budgets = sorted(sub["budget_pct"].unique(), reverse=True)

            for i, b in enumerate(budgets):
                for j, r in enumerate(ratios):
                    pol = pivot.loc[b, r]
                    num_matrix[i, j] = pol_code_map[pol]

            im = ax.imshow(num_matrix, cmap="Accent", vmin=0, vmax=4)
            ax.set_xticks(range(len(ratios)))
            ax.set_xticklabels([f"Ratio {int(r)}" for r in ratios])
            ax.set_yticks(range(len(budgets)))
            ax.set_yticklabels([f"{b}% Budget" for b in budgets])
            ax.set_title(f"Tile Pool: {pool_name.upper()}", fontsize=11, fontweight="bold")

            # Annotate winner and co-leaders
            for i, b in enumerate(budgets):
                for j, r in enumerate(ratios):
                    row = sub[(sub["budget_pct"] == b) & (sub["rehab_cost_ratio"] == r)].iloc[0]
                    win = row["best_policy_aw"].split("_")[0]
                    co = row["co_leaders_aw"].split(",")
                    co_short = [c.split("_")[0] for c in co if c != row["best_policy_aw"]]
                    if co_short:
                        text = f"{win}*\n(+{','.join(co_short)})"
                    else:
                        text = f"{win}"
                    ax.text(j, i, text, ha="center", va="center", fontweight="bold", fontsize=9.5, color="black")

        plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  * Saved {out_path}")

    plot_grid_heatmap(
        grid_scale_summary,
        "Sensitivity Grid 1: Budget Scales with Rehab Cost (Winner & Co-Leaders Marked *)",
        out_fig_dir / "simulation_sensitivity_grid_scaling.png"
    )

    plot_grid_heatmap(
        grid_fixed_summary,
        "Sensitivity Grid 2: Fixed Absolute Budget (Winner & Co-Leaders Marked *)",
        out_fig_dir / "simulation_sensitivity_grid_fixed.png"
    )

    # Figure 5: Regret vs. Lambda Plot
    fig, (ax_r_all, ax_r_crk) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, p_name in [(ax_r_all, "all"), (ax_r_crk, "cracks")]:
        sub_p = noise_df[noise_df["tile_pool"] == p_name]
        for pol in ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
            sub_pol = sub_p[sub_p["policy"] == pol]
            ax.plot(sub_pol["lambda"], sub_pol["regret_aw_mean"], marker="o", color=colors[pol], label=labels[pol], linewidth=2.0)
            ax.fill_between(sub_pol["lambda"], sub_pol["regret_aw_low"], sub_pol["regret_aw_high"], color=colors[pol], alpha=0.15)

        ax.set_title(f"Tile Pool: {p_name.upper()} (Area-Weighted Regret)", fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Noise Scaling Parameter $\lambda$ ($\lambda=0$ Oracle, $\lambda=1$ Model)")
        ax.set_ylabel("Regret: Oracle CY - Estimated CY")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle(r"Information Regret vs. Inspection Noise Level $\lambda$", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    reg_lambda_path = out_fig_dir / "simulation_regret_vs_lambda.png"
    plt.savefig(reg_lambda_path, dpi=150)
    plt.close()
    print(f"  * Saved {reg_lambda_path}")

    print("\n[OK] Step 5B Master Simulation experiments completed successfully!")


if __name__ == "__main__":
    main()
