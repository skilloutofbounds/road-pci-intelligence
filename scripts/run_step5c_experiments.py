"""
Master Experiment Runner for Step 5C: Closing Simulation Questions.
Executes:
1. Error structure of the estimates (per tile and segment means k=6) for all pools and sources.
2. Base Scenario (200 runs, both pools 'all' and 'cracks', AW and TW) with P2t and paired differences vs P2, P3, P4.
3. Bias vs noise decomposition (100 runs each, policies P0-P4 and P2t, both pools, complete regret tables with CIs).
4. Grid 2: Fixed absolute budget (18 cells, 100 runs, co-leaders including P2t).
5. Generation of publication figures (regret vs bias, regret vs lambda, updated Grid 2).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation.network import generate_synthetic_network, RoadSegmentNetwork
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


def compute_segment_error_stats(df_tiles: pd.DataFrame, k: int = 6, n_segments: int = 200, seed: int = 42):
    rng = np.random.RandomState(seed)
    tile_indices = np.arange(len(df_tiles))
    diffs = []
    for _ in range(n_segments):
        idx = rng.choice(tile_indices, size=k, replace=True)
        sub = df_tiles.iloc[idx]
        seg_est = sub["predicted_spci"].mean()
        seg_gt = sub["gt_spci"].mean()
        diffs.append(seg_est - seg_gt)
    arr = np.array(diffs)
    return len(arr), float(np.mean(arr)), float(np.mean(np.abs(arr))), float(np.std(arr, ddof=1))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tile_scores_path = "results/tile_scores.csv"
    if not Path(tile_scores_path).exists():
        print(f"Error: {tile_scores_path} does not exist.")
        return

    out_res_dir = Path("results")
    out_res_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = Path("docs/figures")
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("STEP 5C: SIMULATION ANALYSIS EXPANSION & RIGOROUS ERROR DECOMPOSITION")
    print("=" * 110)

    # =========================================================================
    # ITEM 2: ERROR STRUCTURE OF THE ESTIMATES
    # =========================================================================
    print("\n" + "=" * 110)
    print("ITEM 2: ERROR STRUCTURE OF ESTIMATES (Per-Tile and Segment Means k=6)")
    print("=" * 110)
    tiles_df = pd.read_csv(tile_scores_path)

    pools_and_sources = [
        ("pool: all", tiles_df),
        ("pool: cracks", tiles_df[tiles_df["source"].isin(["crack500", "road_crack"])]),
        ("source: crack500", tiles_df[tiles_df["source"] == "crack500"]),
        ("source: road_crack", tiles_df[tiles_df["source"] == "road_crack"]),
        ("source: pothole", tiles_df[tiles_df["source"] == "pothole"]),
    ]

    print("\n[Per-Tile Error Metrics (Predicted sPCI - GT sPCI)]:")
    print(f"{'Group':<22} | {'N':>5} | {'Mean Signed Error':>18} | {'MAE':>10} | {'Std Dev':>10}")
    print("-" * 75)
    pool_means = {}
    for name, grp in pools_and_sources:
        diff = grp["predicted_spci"] - grp["gt_spci"]
        n_tile = len(grp)
        m_signed = float(diff.mean())
        mae = float(diff.abs().mean())
        std_dev = float(diff.std(ddof=1))
        print(f"{name:<22} | {n_tile:5d} | {m_signed:+18.4f} | {mae:10.4f} | {std_dev:10.4f}")
        if name in ["pool: all", "pool: cracks"]:
            pool_key = "all" if "all" in name else "cracks"
            pool_means[pool_key] = m_signed

    print("\n[Segment Means Error Metrics (k = 6 Tiles per Segment, N = 200 Segments)]:")
    print(f"{'Group':<22} | {'N Segments':>10} | {'Mean Signed Error':>18} | {'MAE':>10} | {'Std Dev (Emp)':>14} | {'Std Dev (CLT Th)':>16}")
    print("-" * 105)
    for name, grp in pools_and_sources:
        n_seg, m_seg, mae_seg, std_seg = compute_segment_error_stats(grp, k=6, n_segments=200, seed=42)
        diff_tile = grp["predicted_spci"] - grp["gt_spci"]
        clt_th = float(diff_tile.std(ddof=1) / np.sqrt(6))
        print(f"{name:<22} | {n_seg:10d} | {m_seg:+18.4f} | {mae_seg:10.4f} | {std_seg:14.4f} | {clt_th:16.4f}")

    # =========================================================================
    # ITEM 1: BASE SCENARIO WITH P2t (Both Pools, AW and TW, 200 runs)
    # =========================================================================
    print("\n" + "=" * 110)
    print("ITEM 1: BASE SCENARIO WITH P2t (Budget 10%, Rehab Ratio 5.0, N = 200 Runs)")
    print("=" * 110)

    policies = [
        "P0_do_nothing", "P1_reactive", "P2_worst_first_by_band",
        "P2t_traffic_aware_worst_first", "P3_greedy_benefit_cost", "P4_integer_programming"
    ]

    base_results = {}
    for pool in ["all", "cracks"]:
        base_cache_file = out_res_dir / f"sim_base_with_p2t_{pool}.csv"
        if base_cache_file.exists() and len(pd.read_csv(base_cache_file)) == 200 * len(policies):
            print(f"\n  * Loading cached base scenario for pool '{pool}' from {base_cache_file}...", flush=True)
            mc_base = pd.read_csv(base_cache_file)
        else:
            net = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool=pool, n_segments=200, seed=42)
            print(f"\n--- Running Base Scenario on Pool '{pool}' (200 runs, synchronized seeds) ---", flush=True)
            mc_base = run_monte_carlo(
                network=net,
                policies=policies,
                budget_pct=10.0,
                rehab_cost_ratio=5.0,
                n_runs=200,
                base_seed=1000,
                use_oracle=False,
            )
            mc_base.to_csv(base_cache_file, index=False)
            print(f"  * Saved {base_cache_file}", flush=True)
        base_results[pool] = mc_base

        print(f"\n[Base Table - Pool '{pool}']:")
        print(f"{'Policy':<28} | {'AW CY [95% CI]':<26} | {'TW CY [95% CI]':<26} | {'Real Spend [95% CI]':<26} | {'Util %':<8} | {'Eff AW (CY/1k)':<15} | {'Eff TW (CY/1k)':<15}")
        print("-" * 155)
        for pol in policies:
            sub = mc_base[mc_base["policy"] == pol]
            cy_aw_m, cy_aw_l, cy_aw_h = bootstrap_ci(sub["condition_years_aw"].values)
            cy_tw_m, cy_tw_l, cy_tw_h = bootstrap_ci(sub["condition_years_tw"].values)
            sp_m, sp_l, sp_h = bootstrap_ci(sub["total_spent"].values)
            ut_m, _, _ = bootstrap_ci(sub["mean_budget_utilization"].values * 100.0)
            eff_aw_m, eff_aw_l, eff_aw_h = bootstrap_ci(sub["efficiency_aw_per_1k"].values)
            eff_tw_m, eff_tw_l, eff_tw_h = bootstrap_ci(sub["efficiency_tw_per_1k"].values)

            eff_aw_str = f"{fmt_3sig(eff_aw_m)} [{fmt_3sig(eff_aw_l)}, {fmt_3sig(eff_aw_h)}]"
            eff_tw_str = f"{fmt_3sig(eff_tw_m)} [{fmt_3sig(eff_tw_l)}, {fmt_3sig(eff_tw_h)}]"
            print(f"{pol:<28} | {cy_aw_m:6.2f} [{cy_aw_l:6.2f}, {cy_aw_h:6.2f}]   | {cy_tw_m:6.2f} [{cy_tw_l:6.2f}, {cy_tw_h:6.2f}]   | {sp_m:9,.0f} [{sp_l:9,.0f}, {sp_h:9,.0f}] | {ut_m:6.2f}% | {eff_aw_str:<15} | {eff_tw_str:<15}")

        # Paired differences of P2t vs P2, P3, P4
        p2t_sub = mc_base[mc_base["policy"] == "P2t_traffic_aware_worst_first"]
        p2t_aw = p2t_sub["condition_years_aw"].values
        p2t_tw = p2t_sub["condition_years_tw"].values

        print(f"\n[Paired Differences: P2t minus Competitor on Pool '{pool}' (200 Runs)]:")
        print(f"{'Comparison':<28} | {'AW Difference [95% CI]':<32} | {'TW Difference [95% CI]':<32}")
        print("-" * 100)
        for comp_pol in ["P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
            comp_sub = mc_base[mc_base["policy"] == comp_pol]
            comp_aw = comp_sub["condition_years_aw"].values
            comp_tw = comp_sub["condition_years_tw"].values

            d_aw_m, d_aw_l, d_aw_h = bootstrap_paired_diff_ci(p2t_aw, comp_aw)
            d_tw_m, d_tw_l, d_tw_h = bootstrap_paired_diff_ci(p2t_tw, comp_tw)
            print(f"P2t - {comp_pol:<22} | {d_aw_m:+6.2f} [{d_aw_l:+6.2f}, {d_aw_h:+6.2f}]             | {d_tw_m:+6.2f} [{d_tw_l:+6.2f}, {d_tw_h:+6.2f}]")

    # =========================================================================
    # ITEM 3: BIAS VS NOISE DECOMPOSITION (100 runs each)
    # =========================================================================
    print("\n" + "=" * 110)
    print("ITEM 3: BIAS VS NOISE DECOMPOSITION (100 Runs per Setting, Synchronized Seeds)")
    print("=" * 110)

    bias_csv = out_res_dir / "sim_bias_decomposition.csv"
    noise_csv = out_res_dir / "sim_noise_decomposition.csv"

    if bias_csv.exists() and noise_csv.exists() and len(pd.read_csv(bias_csv)) == 60 and len(pd.read_csv(noise_csv)) == 60:
        print("  * Loading pre-computed complete Bias and Noise decompositions from CSV...", flush=True)
        df_bias = pd.read_csv(bias_csv)
        df_noise = pd.read_csv(noise_csv)
    else:
        biases = [-10.0, -5.0, 0.0, 5.0, 10.0]
        lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]

        bias_decomp_records = []
        noise_decomp_records = []

        for pool in ["all", "cracks"]:
            net = generate_synthetic_network(tile_scores_path=tile_scores_path, tile_pool=pool, n_segments=200, seed=42)
            mean_err = pool_means[pool]
            print(f"\n--- Running Decompositions for Pool '{pool}' (Pool Mean Signed Error = {mean_err:+.4f}) ---", flush=True)

            # 3a. Bias-Only
            mc_b0 = run_monte_carlo(net, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=2000, bias_b=0.0)

            for b in biases:
                if b == 0.0:
                    mc_b = mc_b0
                else:
                    mc_b = run_monte_carlo(net, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=2000, bias_b=b)

                for pol in policies:
                    orc_aw = mc_b0[mc_b0["policy"] == pol]["condition_years_aw"].values
                    orc_tw = mc_b0[mc_b0["policy"] == pol]["condition_years_tw"].values
                    cur_aw = mc_b[mc_b["policy"] == pol]["condition_years_aw"].values
                    cur_tw = mc_b[mc_b["policy"] == pol]["condition_years_tw"].values

                    # Regret = Oracle - Setting
                    reg_aw_m, reg_aw_l, reg_aw_h = bootstrap_paired_diff_ci(orc_aw, cur_aw)
                    reg_tw_m, reg_tw_l, reg_tw_h = bootstrap_paired_diff_ci(orc_tw, cur_tw)

                    cy_aw_m, _, _ = bootstrap_ci(cur_aw)
                    cy_tw_m, _, _ = bootstrap_ci(cur_tw)

                    bias_decomp_records.append({
                        "tile_pool": pool,
                        "bias_b": b,
                        "policy": pol,
                        "condition_years_aw": cy_aw_m,
                        "condition_years_tw": cy_tw_m,
                        "regret_aw_mean": reg_aw_m,
                        "regret_aw_low": reg_aw_l,
                        "regret_aw_high": reg_aw_h,
                        "regret_tw_mean": reg_tw_m,
                        "regret_tw_low": reg_tw_l,
                        "regret_tw_high": reg_tw_h,
                    })

            # 3b. Noise-Only
            mc_lam0 = run_monte_carlo(net, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=2000, noise_lambda=0.0, pool_mean_error=mean_err)

            for lam in lambdas:
                if lam == 0.0:
                    mc_lam = mc_lam0
                else:
                    mc_lam = run_monte_carlo(net, policies, budget_pct=10.0, rehab_cost_ratio=5.0, n_runs=100, base_seed=2000, noise_lambda=lam, pool_mean_error=mean_err)

                for pol in policies:
                    orc_aw = mc_lam0[mc_lam0["policy"] == pol]["condition_years_aw"].values
                    orc_tw = mc_lam0[mc_lam0["policy"] == pol]["condition_years_tw"].values
                    cur_aw = mc_lam[mc_lam["policy"] == pol]["condition_years_aw"].values
                    cur_tw = mc_lam[mc_lam["policy"] == pol]["condition_years_tw"].values

                    # Regret = Oracle - Setting
                    reg_aw_m, reg_aw_l, reg_aw_h = bootstrap_paired_diff_ci(orc_aw, cur_aw)
                    reg_tw_m, reg_tw_l, reg_tw_h = bootstrap_paired_diff_ci(orc_tw, cur_tw)

                    cy_aw_m, _, _ = bootstrap_ci(cur_aw)
                    cy_tw_m, _, _ = bootstrap_ci(cur_tw)

                    noise_decomp_records.append({
                        "tile_pool": pool,
                        "noise_lambda": lam,
                        "policy": pol,
                        "condition_years_aw": cy_aw_m,
                        "condition_years_tw": cy_tw_m,
                        "regret_aw_mean": reg_aw_m,
                        "regret_aw_low": reg_aw_l,
                        "regret_aw_high": reg_aw_h,
                        "regret_tw_mean": reg_tw_m,
                        "regret_tw_low": reg_tw_l,
                        "regret_tw_high": reg_tw_h,
                    })

        df_bias = pd.DataFrame(bias_decomp_records)
        df_noise = pd.DataFrame(noise_decomp_records)
        df_bias.to_csv(out_res_dir / "sim_bias_decomposition.csv", index=False)
        df_noise.to_csv(out_res_dir / "sim_noise_decomposition.csv", index=False)
        print(f"  * Saved {out_res_dir / 'sim_bias_decomposition.csv'}")
        print(f"  * Saved {out_res_dir / 'sim_noise_decomposition.csv'}")

    # Print COMPLETE Bias Decomposition Tables
    print("\n" + "=" * 115)
    print("COMPLETE TABLE 3a: BIAS-ONLY DECOMPOSITION (Regret vs Oracle b = 0)")
    print("=" * 115)
    print(f"{'Pool':<8} | {'Bias b':<7} | {'Policy':<28} | {'Regret AW [95% CI]':<28} | {'Regret TW [95% CI]':<28} | {'Mean AW CY':<10} | {'Mean TW CY':<10}")
    print("-" * 135)
    for _, r in df_bias.iterrows():
        reg_aw_str = f"{r['regret_aw_mean']:+6.2f} [{r['regret_aw_low']:+6.2f}, {r['regret_aw_high']:+6.2f}]"
        reg_tw_str = f"{r['regret_tw_mean']:+6.2f} [{r['regret_tw_low']:+6.2f}, {r['regret_tw_high']:+6.2f}]"
        print(f"{r['tile_pool']:<8} | {r['bias_b']:+6.1f}  | {r['policy']:<28} | {reg_aw_str:<28} | {reg_tw_str:<28} | {r['condition_years_aw']:10.2f} | {r['condition_years_tw']:10.2f}")

    # Print COMPLETE Noise Decomposition Tables
    print("\n" + "=" * 115)
    print("COMPLETE TABLE 3b: NOISE-ONLY DECOMPOSITION (Regret vs Oracle lambda = 0)")
    print("=" * 115)
    print(f"{'Pool':<8} | {'Lambda':<7} | {'Policy':<28} | {'Regret AW [95% CI]':<28} | {'Regret TW [95% CI]':<28} | {'Mean AW CY':<10} | {'Mean TW CY':<10}")
    print("-" * 135)
    for _, r in df_noise.iterrows():
        reg_aw_str = f"{r['regret_aw_mean']:+6.2f} [{r['regret_aw_low']:+6.2f}, {r['regret_aw_high']:+6.2f}]"
        reg_tw_str = f"{r['regret_tw_mean']:+6.2f} [{r['regret_tw_low']:+6.2f}, {r['regret_tw_high']:+6.2f}]"
        print(f"{r['tile_pool']:<8} | {r['noise_lambda']:6.1f}  | {r['policy']:<28} | {reg_aw_str:<28} | {reg_tw_str:<28} | {r['condition_years_aw']:10.2f} | {r['condition_years_tw']:10.2f}")

    # =========================================================================
    # ITEM 1 (PART B): GRID 2 SENSITIVITY GRID (18 cells, 100 runs, fixed absolute budget)
    # =========================================================================
    print("\n" + "=" * 110)
    print("GRID 2: FIXED ABSOLUTE BUDGET SENSITIVITY GRID (18 Cells x 100 Runs, Including P2t)")
    print("=" * 110)

    grid_fixed_summary, grid_fixed_runs = run_sensitivity_grid_scenario(
        tile_scores_path=tile_scores_path,
        budget_mode="fixed_absolute",
        n_runs=100,
        base_seed=3000,
    )
    grid_fixed_summary.to_csv(out_res_dir / "sim_sensitivity_grid_fixed_step5c.csv", index=False)
    print(f"  * Saved {out_res_dir / 'sim_sensitivity_grid_fixed_step5c.csv'}")

    print("\n[Grid 2 Summary Table (18 Cells)]:")
    print(f"{'Pool':<8} | {'Budget':<7} | {'Ratio':<6} | {'Nominal Winner AW':<24} | {'Co-Leaders (AW)':<32} | {'Nominal Winner TW':<24} | {'Co-Leaders (TW)':<32}")
    print("-" * 150)
    for _, r in grid_fixed_summary.iterrows():
        print(f"{r['tile_pool']:<8} | {r['budget_pct']:4.0f}%  | {r['rehab_cost_ratio']:4d}   | {r['best_policy_aw']:<24} | {r['co_leaders_aw']:<32} | {r['best_policy_tw']:<24} | {r['co_leaders_tw']:<32}")

    # Victory tallies
    print("\n[Grid 2 Victory Tallies across 18 Cells]:")
    print(f"{'Policy':<28} | {'Nominal Wins (AW)':>18} | {'Co-Wins (AW)':>14} | {'Nominal Wins (TW)':>18} | {'Co-Wins (TW)':>14}")
    print("-" * 105)
    for pol in policies:
        nom_aw = sum(grid_fixed_summary["best_policy_aw"] == pol)
        co_aw = sum(grid_fixed_summary["co_leaders_aw"].apply(lambda s: pol in s.split(",")))
        nom_tw = sum(grid_fixed_summary["best_policy_tw"] == pol)
        co_tw = sum(grid_fixed_summary["co_leaders_tw"].apply(lambda s: pol in s.split(",")))
        print(f"{pol:<28} | {nom_aw:18d} | {co_aw:14d} | {nom_tw:18d} | {co_tw:14d}")

    # =========================================================================
    # ITEM 5: FIGURES GENERATION
    # =========================================================================
    print("\n" + "=" * 110)
    print("ITEM 5: GENERATING PUBLICATION FIGURES (docs/figures/)")
    print("=" * 110)

    # 1. Regret vs Bias Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    palette = {
        "P0_do_nothing": "#95a5a6",
        "P1_reactive": "#e74c3c",
        "P2_worst_first_by_band": "#e67e22",
        "P2t_traffic_aware_worst_first": "#9b59b6",
        "P3_greedy_benefit_cost": "#3498db",
        "P4_integer_programming": "#2ecc71",
    }
    labels_map = {
        "P0_do_nothing": "P0 (Do Nothing)",
        "P1_reactive": "P1 (Reactive)",
        "P2_worst_first_by_band": "P2 (Worst-First)",
        "P2t_traffic_aware_worst_first": "P2t (Traffic-Aware Worst-First)",
        "P3_greedy_benefit_cost": "P3 (Greedy Benefit/Cost)",
        "P4_integer_programming": "P4 (Exact 0/1 IP)",
    }

    # Top row: Pool All (AW and TW)
    for p_idx, pool in enumerate(["all", "cracks"]):
        sub_bias = df_bias[df_bias["tile_pool"] == pool]
        ax_aw = axes[p_idx, 0]
        ax_tw = axes[p_idx, 1]

        for pol in policies:
            if pol == "P0_do_nothing":
                continue
            pol_data = sub_bias[sub_bias["policy"] == pol].sort_values("bias_b")
            bx = pol_data["bias_b"].values
            # AW
            ax_aw.plot(bx, pol_data["regret_aw_mean"], marker="o", color=palette[pol], label=labels_map[pol], lw=2)
            ax_aw.fill_between(bx, pol_data["regret_aw_low"], pol_data["regret_aw_high"], color=palette[pol], alpha=0.15)
            # TW
            ax_tw.plot(bx, pol_data["regret_tw_mean"], marker="s", color=palette[pol], label=labels_map[pol], lw=2)
            ax_tw.fill_between(bx, pol_data["regret_tw_low"], pol_data["regret_tw_high"], color=palette[pol], alpha=0.15)

        ax_aw.axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        ax_tw.axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        ax_aw.set_title(f"Pool '{pool}' - Area-Weighted Regret vs Bias b", fontsize=11, fontweight="bold")
        ax_tw.set_title(f"Pool '{pool}' - Traffic-Weighted Regret vs Bias b", fontsize=11, fontweight="bold")
        ax_aw.set_ylabel("Regret (Oracle CY - Setting CY)", fontsize=10)
        ax_aw.grid(True, linestyle=":", alpha=0.6)
        ax_tw.grid(True, linestyle=":", alpha=0.6)
        if p_idx == 1:
            ax_aw.set_xlabel("Bias b (sPCI points: estimate = true + b)", fontsize=10)
            ax_tw.set_xlabel("Bias b (sPCI points: estimate = true + b)", fontsize=10)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    bias_fig_path = out_fig_dir / "simulation_regret_vs_bias.png"
    plt.savefig(bias_fig_path, dpi=300)
    plt.close()
    print(f"  * Generated and saved: {bias_fig_path}")

    # 2. Regret vs Lambda Figure (Noise decomposition)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for p_idx, pool in enumerate(["all", "cracks"]):
        sub_noise = df_noise[df_noise["tile_pool"] == pool]
        ax_aw = axes[p_idx, 0]
        ax_tw = axes[p_idx, 1]

        for pol in policies:
            if pol == "P0_do_nothing":
                continue
            pol_data = sub_noise[sub_noise["policy"] == pol].sort_values("noise_lambda")
            lx = pol_data["noise_lambda"].values
            # AW
            ax_aw.plot(lx, pol_data["regret_aw_mean"], marker="o", color=palette[pol], label=labels_map[pol], lw=2)
            ax_aw.fill_between(lx, pol_data["regret_aw_low"], pol_data["regret_aw_high"], color=palette[pol], alpha=0.15)
            # TW
            ax_tw.plot(lx, pol_data["regret_tw_mean"], marker="s", color=palette[pol], label=labels_map[pol], lw=2)
            ax_tw.fill_between(lx, pol_data["regret_tw_low"], pol_data["regret_tw_high"], color=palette[pol], alpha=0.15)

        ax_aw.axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        ax_tw.axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        ax_aw.set_title(f"Pool '{pool}' - Area-Weighted Regret vs Noise Scale λ", fontsize=11, fontweight="bold")
        ax_tw.set_title(f"Pool '{pool}' - Traffic-Weighted Regret vs Noise Scale λ", fontsize=11, fontweight="bold")
        ax_aw.set_ylabel("Regret (Oracle CY - Setting CY)", fontsize=10)
        ax_aw.grid(True, linestyle=":", alpha=0.6)
        ax_tw.grid(True, linestyle=":", alpha=0.6)
        if p_idx == 1:
            ax_aw.set_xlabel("Noise Scale λ (estimate = true + λ * zero-mean noise)", fontsize=10)
            ax_tw.set_xlabel("Noise Scale λ (estimate = true + λ * zero-mean noise)", fontsize=10)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    noise_fig_path = out_fig_dir / "simulation_regret_vs_lambda.png"
    plt.savefig(noise_fig_path, dpi=300)
    plt.close()
    print(f"  * Generated and saved: {noise_fig_path}")

    # 3. Grid 2 Heatmap Figure (Updated with P2t)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for p_idx, pool in enumerate(["all", "cracks"]):
        sub_grid = grid_fixed_summary[grid_fixed_summary["tile_pool"] == pool]
        matrix_tw = np.zeros((3, 3))
        text_annot = []

        budgets = [5, 10, 20]
        ratios = [3, 5, 8]
        pol_code = {
            "P0_do_nothing": 0, "P1_reactive": 1, "P2_worst_first_by_band": 2,
            "P2t_traffic_aware_worst_first": 2.5, "P3_greedy_benefit_cost": 3, "P4_integer_programming": 4
        }
        short_names = {
            "P0_do_nothing": "P0", "P1_reactive": "P1", "P2_worst_first_by_band": "P2",
            "P2t_traffic_aware_worst_first": "P2t", "P3_greedy_benefit_cost": "P3", "P4_integer_programming": "P4"
        }

        for r_i, b in enumerate(budgets):
            row_annot = []
            for c_i, r in enumerate(ratios):
                row_match = sub_grid[(sub_grid["budget_pct"] == b) & (sub_grid["rehab_cost_ratio"] == r)].iloc[0]
                best = row_match["best_policy_tw"]
                co = row_match["co_leaders_tw"].split(",")
                matrix_tw[r_i, c_i] = pol_code[best]
                co_short = "/".join([short_names[p] for p in co])
                row_annot.append(f"Win: {short_names[best]}\n({co_short})")
            text_annot.append(row_annot)

        ax = axes[p_idx]
        cmap = plt.cm.get_cmap("viridis", 5)
        im = ax.imshow(matrix_tw, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels([f"r = {r}" for r in ratios])
        ax.set_yticklabels([f"B = {b}%" for b in budgets])
        ax.set_xlabel("Rehabilitation Cost Ratio (c_rehab)")
        ax.set_ylabel("Annual Budget (% of Ref Rehab)")
        ax.set_title(f"Grid 2 (Fixed Budget): Pool '{pool}' - TW Winners & Co-Leaders", fontsize=11, fontweight="bold")

        for r_i in range(3):
            for c_i in range(3):
                ax.text(c_i, r_i, text_annot[r_i][c_i], ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    grid_fig_path = out_fig_dir / "simulation_sensitivity_grid_fixed.png"
    plt.savefig(grid_fig_path, dpi=300)
    plt.close()
    print(f"  * Generated and saved: {grid_fig_path}")

    # List exact figure files and byte sizes
    print("\n[docs/figures/ Directory Listing]:")
    fig_files = sorted(out_fig_dir.glob("*.png"))
    for f in fig_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    print(f"Total figure files in docs/figures/: {len(fig_files)}")

    print("\nStep 5C Master Experiment Run Completed Successfully.")


if __name__ == "__main__":
    main()
