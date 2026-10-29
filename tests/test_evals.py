"""Unit tests for the extensible evaluation and benchmarking suite."""

import unittest
from pathlib import Path
import tempfile
import json
import numpy as np

from runtime.evals.scenarios import (
    EvalScenario,
    register_scenario,
    get_scenario,
    list_scenarios,
)
from runtime.evals.metrics import (
    register_metric,
    compute_all_metrics,
    compute_improvement_pct,
    compute_safety_score,
    compute_recovery_speedup,
    StepRecord,
)
from runtime.evals.eval_harness import run_eval, EvalResult
from runtime.evals.benchmark_suite import BenchmarkSuite
from runtime.evals.report_generator import generate_html_dashboard
from runtime.sim.config import System2MasterConfig
from runtime.chaos.models import DroneChaosState, WindChaosConfig


class TestEvalsSuite(unittest.TestCase):
    """Test suite verifying extensibility, metric calculation, and benchmarking."""

    def test_preset_auto_discovery(self) -> None:
        """Verifies that presets in CHAOS_PRESETS are automatically discovered."""
        scenarios = list_scenarios()
        self.assertGreaterEqual(len(scenarios), 5)
        # Verify core IDs exist
        ids = [s.id for s in scenarios]
        self.assertIn("chipped_propeller", ids)
        self.assertIn("microburst_wind", ids)
        self.assertIn("tethered_cargo_wind", ids)
        self.assertIn("sensor_chatter", ids)
        self.assertIn("compound_stress", ids)

    def test_dynamic_severity_scaling(self) -> None:
        """Verifies that severity scaling scales physical disturbance parameters."""
        wind_sc = get_scenario("microburst_wind")
        state_mild = wind_sc.instantiate(severity=0.5)
        state_baseline = wind_sc.instantiate(severity=1.0)
        state_harsh = wind_sc.instantiate(severity=2.0)

        # Baseline steady force is 6.5N in X
        self.assertAlmostEqual(state_mild.wind.steady_force[0], 3.25, places=2)
        self.assertAlmostEqual(state_baseline.wind.steady_force[0], 6.50, places=2)
        self.assertAlmostEqual(state_harsh.wind.steady_force[0], 13.00, places=2)

    def test_custom_scenario_registration(self) -> None:
        """Verifies that new future chaos scenarios can be registered dynamically."""
        def make_thermal_draft(severity: float) -> DroneChaosState:
            st = DroneChaosState()
            st.wind.enabled = True
            st.wind.steady_force = [0.0, 0.0, 4.0 * severity]
            return st

        custom = EvalScenario(
            id="thermal_updraft_test",
            display_name="Thermal Vertical Updraft",
            description="Strong rising vertical thermal column",
            make_state=make_thermal_draft,
            tags=["thermal", "updraft"],
        )
        register_scenario(custom, aliases=["updraft"])

        retrieved = get_scenario("thermal_updraft_test")
        self.assertEqual(retrieved.display_name, "Thermal Vertical Updraft")
        by_alias = get_scenario("updraft")
        self.assertEqual(by_alias.id, "thermal_updraft_test")

        st = retrieved.instantiate(severity=1.5)
        self.assertAlmostEqual(st.wind.steady_force[2], 6.0, places=2)

    def test_metric_registry_and_custom_metric(self) -> None:
        """Verifies pluggable metric calculation and custom metric addition."""
        # Register custom metric
        register_metric(
            id="test_x_disp",
            display_name="Max X Displacement",
            unit="m",
            fn=lambda steps: float(np.max([abs(s["pos"][0]) for s in steps])),
        )

        dummy_steps: list[StepRecord] = [
            {"pos_error": 0.1, "pos": [0.05, 0.0, 1.0], "tilt_deg": 5.0, "motor_cmds": [3.25]*4, "altitude": 1.0, "t": 0.0},
            {"pos_error": 0.2, "pos": [0.15, 0.0, 1.0], "tilt_deg": 10.0, "motor_cmds": [3.5]*4, "altitude": 1.0, "t": 0.1},
        ]
        metrics = compute_all_metrics(dummy_steps)
        self.assertIn("mean_pos_error", metrics)
        self.assertIn("test_x_disp", metrics)
        self.assertAlmostEqual(metrics["test_x_disp"], 0.15, places=2)

    def test_run_eval_single_execution(self) -> None:
        """Tests that run_eval executes both s1_only and s1_s2 cleanly."""
        sc = get_scenario("microburst_wind")
        res_s1 = run_eval(sc, mode="s1_only", severity=1.0, duration_s=1.0, seed=42)
        res_s2 = run_eval(sc, mode="s1_s2", severity=1.0, duration_s=1.0, seed=42)

        self.assertFalse(res_s1.crashed)
        self.assertFalse(res_s2.crashed)
        self.assertIn("mean_pos_error", res_s1.metrics)
        self.assertIn("mean_pos_error", res_s2.metrics)
        # S1+S2 should have significantly lower position error than S1 alone
        self.assertLess(res_s2.metrics["mean_pos_error"], res_s1.metrics["mean_pos_error"])

    def test_benchmark_suite_comparison(self) -> None:
        """Tests multi-seed scenario comparison in BenchmarkSuite."""
        suite = BenchmarkSuite()
        comp = suite.compare_scenario("microburst_wind", severity=1.0, seeds=[42], duration_s=1.0)
        self.assertIn("s1_s2", comp.winner)
        self.assertGreater(comp.pos_improvement_pct, 50.0)

    def test_grid_search_execution(self) -> None:
        """Tests parameter sweep grid search optimization."""
        suite = BenchmarkSuite()
        grid = {
            "trigger.pos_error_threshold": [0.20, 0.30],
            "patch.ramp_duration_s": [0.0, 0.20],
        }
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            res = suite.run_grid_search(
                scenario_id="microburst_wind",
                param_grid=grid,
                severity=1.0,
                duration_s=1.0,
                output_best_json=tmp.name,
                verbose=False,
            )
            self.assertEqual(res["trials_evaluated"], 4)
            self.assertIsNotNone(res["best_params"])
            self.assertTrue(Path(tmp.name).exists())

    def test_html_dashboard_generation(self) -> None:
        """Tests that the interactive HTML dashboard is successfully built."""
        suite = BenchmarkSuite()
        report = suite.run_full_benchmark(
            scenario_ids=["microburst_wind"],
            severities=[1.0],
            seeds=[42],
            duration_s=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_html = Path(tmpdir) / "test_dash.html"
            p = generate_html_dashboard(report, output_path=out_html)
            self.assertTrue(p.exists())
            content = p.read_text(encoding="utf-8")
            self.assertIn("System-1 + System-2 Hybrid Autonomous Drone Benchmark", content)
            self.assertIn("Microburst", content)


if __name__ == "__main__":
    unittest.main()
