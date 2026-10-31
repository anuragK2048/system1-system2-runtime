#!/usr/bin/env python3
"""CLI Runner for Hero Breaking-Point Analysis.

Sweeps disturbance severity to locate the exact threshold where System-1
(baseline L1 adaptive) crashes while System-1 + System-2 survives.

Usage:
    uv run python evals/run_breaking_point.py
    uv run python evals/run_breaking_point.py --scenario microburst_wind --max-sev 3.0
"""

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from runtime.sim.config import System2MasterConfig
from runtime.evals import (
    BenchmarkSuite,
    list_scenarios,
    print_breaking_point,
    generate_html_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the Hero Breaking-Point where S1 crashes and S2 survives")
    parser.add_argument("--scenario", "-s", type=str, default="microburst_wind", help="Scenario ID to test")
    parser.add_argument("--min-sev", type=float, default=0.5, help="Minimum severity scale (default: 0.5)")
    parser.add_argument("--max-sev", type=float, default=3.0, help="Maximum severity scale (default: 3.0)")
    parser.add_argument("--step", type=float, default=0.25, help="Step size for severity ramp (default: 0.25)")
    parser.add_argument("--duration", "-d", type=float, default=4.0, help="Simulation duration per step (default: 4.0s)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Custom System2MasterConfig JSON path")
    args = parser.parse_args()

    s2_cfg = None
    if args.config:
        s2_cfg = System2MasterConfig.load_json(args.config)

    suite = BenchmarkSuite(s2_cfg=s2_cfg)
    print(f"🔬 Searching for breaking point on '{args.scenario}' from {args.min_sev}x to {args.max_sev}x...")

    bp = suite.run_breaking_point_search(
        scenario_id=args.scenario,
        severity_min=args.min_sev,
        severity_max=args.max_sev,
        step_size=args.step,
        duration_s=args.duration,
        s2_cfg=s2_cfg,
        verbose=False,
    )

    print_breaking_point(bp)


if __name__ == "__main__":
    main()
