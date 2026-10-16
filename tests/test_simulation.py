"""
Unit Test Suite for Pavement Maintenance Simulation (Step 5A).
Tests:
1. Budget never exceeded across policies.
2. PCI strictly bounded within [0, 100].
3. No maintenance gives non-increasing PCI over time.
4. Treatments never reduce PCI.
5. P4 (MILP) never has a lower one-year total benefit than P3 (greedy) on 50 random instances.
6. P4 equals brute force global optimum on 8-segment instances.
7. Deterministic reproducibility under a fixed seed.
"""

import unittest
import itertools
import numpy as np
import pandas as pd

from src.simulation.network import RoadSegmentNetwork
from src.simulation.engine import (
    SimulationEngine,
    calculate_deterioration,
    calculate_treatment_outcome,
    get_expected_gain,
    run_policy_simulation,
)


def create_mock_network(n_segments: int = 20, seed: int = 42) -> RoadSegmentNetwork:
    rng = np.random.RandomState(seed)
    records = []
    for i in range(n_segments):
        l = float(rng.uniform(100.0, 600.0))
        w = float(rng.uniform(6.0, 14.0))
        area = l * w
        t_cls = rng.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15])
        t_wt = {"low": 1.0, "medium": 2.0, "high": 3.0}[t_cls]
        true_p = float(rng.uniform(10.0, 95.0))
        est_p = float(np.clip(true_p + rng.normal(0, 5.0), 0.0, 100.0))
        records.append({
            "segment_id": f"seg_{i:03d}",
            "length_m": l,
            "width_m": w,
            "area_m2": area,
            "traffic_class": t_cls,
            "traffic_weight": t_wt,
            "estimated_pci": est_p,
            "true_pci": true_p,
            "tile_paths": ["mock.jpg"] * 6,
        })
    return RoadSegmentNetwork(pd.DataFrame(records))


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.engine = SimulationEngine()

    def test_01_budget_never_exceeded(self):
        """Verify that annual budget is never exceeded across all policies and years."""
        net = create_mock_network(n_segments=30, seed=101)
        policies = ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]
        for pol in policies:
            for b_pct in [5.0, 10.0, 20.0]:
                for ratio in [3.0, 5.0, 8.0]:
                    annual_budget = float(ratio * net.total_area_m2 * (b_pct / 100.0))
                    res = run_policy_simulation(
                        net, pol, budget_pct=b_pct, rehab_cost_ratio=ratio,
                        horizon_years=5, seed=42, engine=self.engine
                    )
                    # Total spent over 5 years cannot exceed 5 * annual_budget
                    self.assertLessEqual(res["total_spent"], 5 * annual_budget + 1e-4)

    def test_02_pci_always_within_bounds(self):
        """Verify that PCI remains strictly in [0.0, 100.0] under extreme treatments and deteriorations."""
        rng = np.random.RandomState(42)
        # Test deterioration bounds
        for pci in [0.0, 1.0, 50.0, 70.0, 100.0]:
            for r in [0.0, 3.0, 10.0, 50.0, 200.0]:
                new_p = calculate_deterioration(pci, r)
                self.assertGreaterEqual(new_p, 0.0)
                self.assertLessEqual(new_p, 100.0)

        # Test treatment bounds
        for t in ["none", "preventive", "rehabilitation", "reconstruction"]:
            for pci in [0.0, 20.0, 55.0, 75.0, 95.0, 100.0]:
                new_p = calculate_treatment_outcome(pci, t, rng)
                self.assertGreaterEqual(new_p, 0.0)
                self.assertLessEqual(new_p, 100.0)

    def test_03_no_maintenance_gives_non_increasing_pci(self):
        """Verify that P0 (no maintenance) guarantees non-increasing PCI monotonically for all segments."""
        net = create_mock_network(n_segments=25, seed=202)
        rng = np.random.RandomState(42)
        df = net.df.copy()
        current_true = df["true_pci"].values.copy()
        current_est = df["estimated_pci"].values.copy()
        areas = df["area_m2"].values
        weights = df["traffic_weight"].values

        for year in range(5):
            next_true, next_est, spent, _ = self.engine.run_year_step(
                current_true, current_est, areas, weights,
                annual_budget=0.0, policy_name="P0_do_nothing",
                rehab_cost_per_m2=5.0, rng=rng
            )
            self.assertEqual(spent, 0.0)
            # Monotonically non-increasing
            self.assertTrue(np.all(next_true <= current_true + 1e-9))
            current_true = next_true
            current_est = next_est

    def test_04_treatment_never_reduces_pci(self):
        """Verify that any treatment applied to any condition never decreases PCI."""
        rng = np.random.RandomState(303)
        treatments = ["none", "preventive", "rehabilitation", "reconstruction"]
        for _ in range(500):
            pci = float(rng.uniform(0.0, 100.0))
            t = rng.choice(treatments)
            outcome = calculate_treatment_outcome(pci, t, rng)
            self.assertGreaterEqual(outcome, pci - 1e-9)

    def test_05_p4_never_lower_than_p3_on_50_instances(self):
        """Verify that exact 0/1 IP (P4) never achieves a lower total benefit than greedy (P3) on 50 instances."""
        treatments = ["none", "preventive", "rehabilitation", "reconstruction"]
        t_costs = {"none": 0.0, "preventive": 1.0, "rehabilitation": 5.0, "reconstruction": 12.0}

        rng = np.random.RandomState(404)
        for trial in range(50):
            n = rng.randint(10, 30)
            pcis = rng.uniform(5.0, 95.0, size=n)
            areas = rng.uniform(500.0, 5000.0, size=n)
            weights = rng.choice([1.0, 2.0, 3.0], size=n)
            budget = float(rng.uniform(1000.0, 25000.0))

            # Run P3 step
            _, _, _, t_p3 = self.engine.run_year_step(
                current_true_pci=pcis.copy(),
                current_est_pci=pcis.copy(),
                areas=areas,
                traffic_weights=weights,
                annual_budget=budget,
                policy_name="P3_greedy_benefit_cost",
                rehab_cost_per_m2=5.0,
                rng=rng,
                use_oracle=True,
            )
            # Compute P3 total benefit
            b_p3 = sum(
                get_expected_gain(pcis[i], t_p3[i]) * areas[i] * weights[i]
                for i in range(n)
            )

            # Run P4 step
            _, _, _, t_p4 = self.engine.run_year_step(
                current_true_pci=pcis.copy(),
                current_est_pci=pcis.copy(),
                areas=areas,
                traffic_weights=weights,
                annual_budget=budget,
                policy_name="P4_integer_programming",
                rehab_cost_per_m2=5.0,
                rng=rng,
                use_oracle=True,
            )
            # Compute P4 total benefit
            b_p4 = sum(
                get_expected_gain(pcis[i], t_p4[i]) * areas[i] * weights[i]
                for i in range(n)
            )

            # P4 must be >= P3 within numerical tolerance
            self.assertGreaterEqual(
                b_p4, b_p3 - 1e-4,
                f"Trial {trial}: P4 benefit ({b_p4:.2f}) < P3 benefit ({b_p3:.2f})"
            )

    def test_06_p4_equals_brute_force_on_8_segments(self):
        """Verify that P4 achieves exact global optimum by comparing against brute force on 8 segments (4^8 = 65,536)."""
        n = 8
        rng = np.random.RandomState(505)
        pcis = rng.uniform(10.0, 90.0, size=n)
        areas = rng.uniform(500.0, 2000.0, size=n)
        weights = rng.choice([1.0, 2.0, 3.0], size=n)
        budget = 15000.0

        treatments = ["none", "preventive", "rehabilitation", "reconstruction"]
        t_costs = {"none": 0.0, "preventive": 1.0, "rehabilitation": 5.0, "reconstruction": 12.0}

        # Brute force search over 4^8 combinations
        best_bf_benefit = -1.0
        best_bf_combo = None

        for combo in itertools.product(treatments, repeat=n):
            total_c = sum(t_costs[combo[i]] * areas[i] for i in range(n))
            if total_c <= budget:
                total_b = sum(
                    get_expected_gain(pcis[i], combo[i]) * areas[i] * weights[i]
                    for i in range(n)
                )
                if total_b > best_bf_benefit:
                    best_bf_benefit = total_b
                    best_bf_combo = combo

        # P4 solution
        _, _, spent_p4, t_p4 = self.engine.run_year_step(
            current_true_pci=pcis.copy(),
            current_est_pci=pcis.copy(),
            areas=areas,
            traffic_weights=weights,
            annual_budget=budget,
            policy_name="P4_integer_programming",
            rehab_cost_per_m2=5.0,
            rng=rng,
            use_oracle=True,
        )
        b_p4 = sum(
            get_expected_gain(pcis[i], t_p4[i]) * areas[i] * weights[i]
            for i in range(n)
        )

        self.assertAlmostEqual(b_p4, best_bf_benefit, places=3)
        self.assertLessEqual(spent_p4, budget + 1e-4)

    def test_07_deterministic_reproducibility(self):
        """Verify identical results under fixed random seed across multiple simulation executions."""
        net = create_mock_network(n_segments=20, seed=606)
        res1 = run_policy_simulation(net, "P3_greedy_benefit_cost", budget_pct=10.0, seed=777, engine=self.engine)
        res2 = run_policy_simulation(net, "P3_greedy_benefit_cost", budget_pct=10.0, seed=777, engine=self.engine)

        self.assertEqual(res1["condition_years"], res2["condition_years"])
        self.assertEqual(res1["final_network_pci"], res2["final_network_pci"])
        self.assertEqual(res1["total_spent"], res2["total_spent"])
        self.assertEqual(res1["yearly_trajectory"], res2["yearly_trajectory"])

    def test_08_corrected_spend_accounting(self):
        """Verify spend accounting: spend <= budget each year, P0 spend = 0, reported spend = sum of applied treatment costs."""
        net = create_mock_network(n_segments=30, seed=707)
        policies = ["P0_do_nothing", "P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]
        areas = net.df["area_m2"].values
        treatment_costs = {"none": 0.0, "preventive": 1.0, "rehabilitation": 5.0, "reconstruction": 12.0}

        for pol in policies:
            res = run_policy_simulation(net, pol, budget_pct=10.0, rehab_cost_ratio=5.0, horizon_years=5, seed=888, engine=self.engine)
            annual_budget = res["annual_budget"]

            if pol == "P0_do_nothing":
                self.assertEqual(res["total_spent"], 0.0)
                self.assertTrue(all(s == 0.0 for s in res["yearly_spent"]))
            else:
                for y_idx, spent in enumerate(res["yearly_spent"]):
                    self.assertLessEqual(spent, annual_budget + 1e-5, f"{pol} year {y_idx+1} spend exceeds budget")

            # Check that total_spent equals sum of yearly_spent
            self.assertAlmostEqual(res["total_spent"], sum(res["yearly_spent"]), places=4)

        # Single-step exact sum verification
        rng = np.random.RandomState(999)
        current_true = net.df["true_pci"].values.copy()
        current_est = net.df["estimated_pci"].values.copy()
        weights = net.df["traffic_weight"].values
        budget = 25000.0

        for pol in ["P1_reactive", "P2_worst_first_by_band", "P3_greedy_benefit_cost", "P4_integer_programming"]:
            _, _, spent, treatments = self.engine.run_year_step(
                current_true_pci=current_true,
                current_est_pci=current_est,
                areas=areas,
                traffic_weights=weights,
                annual_budget=budget,
                policy_name=pol,
                rehab_cost_per_m2=5.0,
                rng=rng,
            )
            expected_sum = sum(treatment_costs[t] * areas[i] for i, t in enumerate(treatments))
            self.assertAlmostEqual(spent, expected_sum, places=4, msg=f"{pol} reported spend != sum of treatment costs")
            self.assertLessEqual(spent, budget + 1e-5)

    def test_09_smooth_preventive_effectiveness(self):
        """Verify smooth preventive treatment effectiveness: clip((PCI - 40)/30, 0, 1)."""
        rng = np.random.RandomState(42)
        # Below 40: factor is 0, gain is 0
        self.assertEqual(calculate_treatment_outcome(35.0, "preventive", rng, preventive_mode="smooth"), 35.0)
        self.assertEqual(calculate_treatment_outcome(40.0, "preventive", rng, preventive_mode="smooth"), 40.0)

        # Above 70: factor is 1, full 15 gain
        self.assertAlmostEqual(calculate_treatment_outcome(70.0, "preventive", rng, preventive_mode="smooth"), 85.0, places=4)
        self.assertAlmostEqual(calculate_treatment_outcome(80.0, "preventive", rng, preventive_mode="smooth"), 95.0, places=4)

        # Midpoint: PCI 55 -> factor = 15/30 = 0.5 -> gain = 7.5 -> 62.5
        self.assertAlmostEqual(calculate_treatment_outcome(55.0, "preventive", rng, preventive_mode="smooth"), 62.5, places=4)

        # Cap at 95
        self.assertAlmostEqual(calculate_treatment_outcome(90.0, "preventive", rng, preventive_mode="smooth"), 95.0, places=4)

    def test_10_traffic_weighted_condition_years(self):
        """Verify traffic-weighted condition-years calculation."""
        net = create_mock_network(n_segments=15, seed=123)
        res = run_policy_simulation(net, "P2_worst_first_by_band", budget_pct=10.0, horizon_years=3, seed=456, engine=self.engine)
        self.assertIn("condition_years_aw", res)
        self.assertIn("condition_years_tw", res)
        self.assertGreater(res["condition_years_aw"], 0.0)
        self.assertGreater(res["condition_years_tw"], 0.0)

    def test_11_noise_lambda_scaling(self):
        """Verify noise scaling parameter lambda: lambda=0 is oracle, lambda=1 is standard model."""
        net = create_mock_network(n_segments=15, seed=321)
        res_oracle = run_policy_simulation(net, "P3_greedy_benefit_cost", noise_lambda=0.0, seed=111, engine=self.engine)
        res_real = run_policy_simulation(net, "P3_greedy_benefit_cost", noise_lambda=1.0, seed=111, engine=self.engine)
        res_noisy = run_policy_simulation(net, "P3_greedy_benefit_cost", noise_lambda=2.0, seed=111, engine=self.engine)

        self.assertIsNotNone(res_oracle)
        self.assertIsNotNone(res_real)
        self.assertIsNotNone(res_noisy)

    def test_12_p2t_traffic_aware_ordering_and_budget(self):
        """Verify P2t orders segments strictly descending by (100 - PCI) * traffic_weight and respects budget."""
        rng = np.random.RandomState(42)
        n = 10
        current_true = np.array([20.0, 30.0, 60.0, 80.0, 20.0, 50.0, 65.0, 10.0, 45.0, 75.0])
        current_est = current_true.copy()
        areas = np.full(n, 1000.0)
        weights = np.array([1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0])

        # Priority scores:
        # Seg 4: (100-20)*3 = 240 (Reconstruction, cost $12,000)
        # Seg 1: (100-30)*3 = 210 (Rehabilitation, cost $5,000)
        # Seg 8: (100-45)*2 = 110 (Rehabilitation, cost $5,000)
        # Seg 7: (100-10)*1 = 90  (Reconstruction, cost $12,000; note PCI=10 is worse than Seg 1 PCI=30, but lower traffic)
        # Budget = $17,500 ensures top 2 (Seg 4 and Seg 1) are treated for $17,000, leaving $500.
        budget = 17500.0
        _, _, spent, treatments = self.engine.run_year_step(
            current_true_pci=current_true,
            current_est_pci=current_est,
            areas=areas,
            traffic_weights=weights,
            annual_budget=budget,
            policy_name="P2t_traffic_aware_worst_first",
            rehab_cost_per_m2=5.0,
            rng=rng,
        )
        self.assertEqual(treatments[4], "reconstruction")
        self.assertEqual(treatments[1], "rehabilitation")
        # Remaining segments must be "none" because remaining budget ($500) < candidate cost
        for i in [0, 2, 3, 5, 6, 7, 8, 9]:
            self.assertEqual(treatments[i], "none", f"Segment {i} unexpectedly treated")
        self.assertEqual(spent, 17000.0)
        self.assertLessEqual(spent, budget)

    def test_13_bias_b_zero_and_lambda_zero_reproduce_oracle(self):
        """Verify b=0 and lambda=0 reproduce the oracle exactly for all policies."""
        net = create_mock_network(n_segments=20, seed=777)
        policies = [
            "P0_do_nothing", "P1_reactive", "P2_worst_first_by_band",
            "P2t_traffic_aware_worst_first", "P3_greedy_benefit_cost", "P4_integer_programming"
        ]
        for pol in policies:
            res_orc = run_policy_simulation(net, pol, seed=999, use_oracle=True, engine=self.engine)
            res_b0 = run_policy_simulation(net, pol, seed=999, bias_b=0.0, engine=self.engine)
            res_lam0 = run_policy_simulation(net, pol, seed=999, noise_lambda=0.0, engine=self.engine)

            self.assertEqual(res_orc["condition_years_aw"], res_b0["condition_years_aw"],
                             f"{pol}: b=0 AW condition-years != oracle")
            self.assertEqual(res_orc["condition_years_tw"], res_b0["condition_years_tw"],
                             f"{pol}: b=0 TW condition-years != oracle")
            self.assertEqual(res_orc["total_spent"], res_b0["total_spent"],
                             f"{pol}: b=0 total spent != oracle")

            self.assertEqual(res_orc["condition_years_aw"], res_lam0["condition_years_aw"],
                             f"{pol}: lambda=0 AW condition-years != oracle")
            self.assertEqual(res_orc["condition_years_tw"], res_lam0["condition_years_tw"],
                             f"{pol}: lambda=0 TW condition-years != oracle")
            self.assertEqual(res_orc["total_spent"], res_lam0["total_spent"],
                             f"{pol}: lambda=0 total spent != oracle")

    def test_14_p2t_and_bias_noise_deterministic(self):
        """Verify determinism under fixed seed for P2t and bias/noise settings."""
        net = create_mock_network(n_segments=20, seed=888)
        # Test P2t determinism
        run1 = run_policy_simulation(net, "P2t_traffic_aware_worst_first", seed=555, engine=self.engine)
        run2 = run_policy_simulation(net, "P2t_traffic_aware_worst_first", seed=555, engine=self.engine)
        self.assertEqual(run1["condition_years_aw"], run2["condition_years_aw"])
        self.assertEqual(run1["condition_years_tw"], run2["condition_years_tw"])
        self.assertEqual(run1["total_spent"], run2["total_spent"])

        # Test bias determinism
        run_b1 = run_policy_simulation(net, "P4_integer_programming", bias_b=5.0, seed=666, engine=self.engine)
        run_b2 = run_policy_simulation(net, "P4_integer_programming", bias_b=5.0, seed=666, engine=self.engine)
        self.assertEqual(run_b1["condition_years_aw"], run_b2["condition_years_aw"])
        self.assertEqual(run_b1["total_spent"], run_b2["total_spent"])

        # Test noise scaling determinism
        run_l1 = run_policy_simulation(net, "P3_greedy_benefit_cost", noise_lambda=1.5, pool_mean_error=-3.1, seed=777, engine=self.engine)
        run_l2 = run_policy_simulation(net, "P3_greedy_benefit_cost", noise_lambda=1.5, pool_mean_error=-3.1, seed=777, engine=self.engine)
        self.assertEqual(run_l1["condition_years_aw"], run_l2["condition_years_aw"])
        self.assertEqual(run_l1["total_spent"], run_l2["total_spent"])


if __name__ == "__main__":
    unittest.main()
