#!/usr/bin/env python3
"""CLI Runner for Automated System-2 Hyperparameter Optimization (Grid Search).

Systematically tests combinations of trigger thresholds, ramp-in durations,
cooldown periods, and safety envelopes to find the globally optimal configuration.

Usage:
    uv run python evals/run_grid_search.py
    uv run python evals/run_grid_search.py --scenario compound_stress --output configs/tuned_compound.json
"""

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from runtime.sim.config import System2MasterConfig
from runtime.evals import BenchmarkSuite


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated System-2 Hyperparameter Optimizer")
    parser.add_argument("--scenario", "-s", type=str, default="microburst_wind", help="Scenario to optimize for")
    parser.add_argument("--severity", "-v", type=float, default=1.0, help="Disturbance severity")
    parser.add_argument("--duration", "-d", type=float, default=4.0, help="Simulation duration per trial")
    parser.add_argument("--output", "-o", type=str, default="configs/best_config.json", help="Path to save best config")
    args = parser.parse_args()

    suite = BenchmarkSuite()

    # Pre-configured high-impact search space
    search_grid = {
        "trigger.pos_error_threshold": [0.15, 0.25],
        "trigger.force_residual_threshold": [1.8, 2.5],
        "patch.ramp_duration_s": [0.0, 0.20, 0.35],
        "worker.cooldown_s": [1.0, 1.8],
    }

    print(f"🎯 Starting Automated Hyperparameter Sweep on [{args.scenario}]...")
    res = suite.run_grid_search(
        scenario_id=args.scenario,
        param_grid=search_grid,
        severity=args.severity,
        duration_s=args.duration,
        output_best_json=args.output,
        verbose=True,
    )

    print("\n" + "═" * 70)
    print("🏆 GRID SEARCH OPTIMIZATION COMPLETE")
    print(f"• Optimal Improvement: {res['best_score']:+.2f}%")
    print(f"• Optimal Parameters:  {res['best_params']}")
    print(f"• Output Saved To:     {args.output}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
