#!/usr/bin/env python3
"""CLI Runner for Multi-Scenario Fault-Tolerance Benchmark.

Usage:
    uv run python evals/run_benchmark.py
    uv run python evals/run_benchmark.py --scenario microburst_wind --severity 1.5
    uv run python evals/run_benchmark.py --live-llm
"""

import sys
import argparse
from pathlib import Path

# Add src to pythonpath if run as standalone script
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from runtime.sim.config import System2MasterConfig
from runtime.evals import (
    BenchmarkSuite,
    list_scenarios,
    print_comparison_table,
    generate_html_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="System-1 vs System-2 Autonomous Drone Benchmark Suite")
    parser.add_argument("--name", "-n", type=str, default=None, help="Descriptive name for this benchmark run (e.g. baseline, tuned_v1)")
    parser.add_argument("--scenario", "-s", type=str, default=None, help="Scenario ID (default: all scenarios)")
    parser.add_argument("--severity", "-v", type=float, default=None, help="Specific severity scale (e.g. 1.0, 1.5)")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Simulation duration in seconds (default: 5.0)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom System2MasterConfig JSON")
    parser.add_argument("--live-llm", action="store_true", help="Query live LLM API instead of deterministic patcher")
    parser.add_argument("--output-html", type=str, default=None, help="HTML dashboard output path")
    parser.add_argument("--output-json", type=str, default=None, help="JSON report output path")
    args = parser.parse_args()

    s2_cfg = None
    config_name = "default"
    if args.config:
        s2_cfg = System2MasterConfig.load_json(args.config)
        config_name = Path(args.config).stem
        print(f"⚙️ Loaded custom System-2 config from: {args.config}")

    run_name = args.name or config_name
    output_json = args.output_json or f"results/runs/{run_name}.json"
    output_html = args.output_html or f"results/runs/{run_name}.html"

    suite = BenchmarkSuite(s2_cfg=s2_cfg)
    scenario_ids = [args.scenario] if args.scenario else None
    severities = [args.severity] if args.severity else [0.5, 1.0, 1.5, 2.0]

    print(f"🚀 Initiating Benchmark Run: [{run_name}]...")
    report = suite.run_full_benchmark(
        scenario_ids=scenario_ids,
        severities=severities,
        seeds=[42],
        duration_s=args.duration,
        s2_cfg=s2_cfg,
        use_live_llm=args.live_llm,
        run_name=run_name,
        config_path=args.config,
        verbose=True,
    )

    print_comparison_table(report)

    # Save run-specific artifacts
    report.save_json(output_json)
    html_path = generate_html_dashboard(report, output_path=output_html)

    # Also update latest results/summary/
    report.save_json("results/summary/benchmark_report.json")
    generate_html_dashboard(report, output_path="results/summary/benchmark_dashboard.html")

    print(f"💾 Run saved to: {output_json}")
    print(f"🌐 Run dashboard generated at: {html_path}")
    print(f"📌 Latest summary updated at: results/summary/benchmark_dashboard.html")


if __name__ == "__main__":
    main()
