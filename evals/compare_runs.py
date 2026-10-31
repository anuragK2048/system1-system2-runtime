#!/usr/bin/env python3
"""CLI Tool to Compare Evaluation Runs or Configurations Side-by-Side.

Usage:
    # 1. Compare two existing run files:
    uv run python evals/compare_runs.py results/runs/default.json results/runs/tuned.json

    # 2. Or benchmark and compare two config files on the fly:
    uv run python evals/compare_runs.py --config-a configs/system2_default.json --config-b configs/system2_tuned_template.json
"""

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from runtime.sim.config import System2MasterConfig
from runtime.evals.benchmark_suite import BenchmarkSuite, BenchmarkReport
from runtime.evals.comparator import compare_reports, print_comparison, generate_comparison_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side Configuration & Run Comparator")
    parser.add_argument("run_files", nargs="*", help="Paths to two JSON run reports to compare")
    parser.add_argument("--config-a", type=str, default=None, help="Path to Configuration A JSON")
    parser.add_argument("--config-b", type=str, default=None, help="Path to Configuration B JSON")
    parser.add_argument("--duration", "-d", type=float, default=3.0, help="Simulation duration when running configs on the fly")
    parser.add_argument("--scenario", "-s", type=str, default=None, help="Optional scenario filter")
    parser.add_argument("--output-html", type=str, default="results/summary/config_comparison.html", help="HTML dashboard output path")
    args = parser.parse_args()

    report_a: BenchmarkReport
    report_b: BenchmarkReport
    name_a: str = "Config A"
    name_b: str = "Config B"

    # Mode 1: Compare two existing JSON files
    if len(args.run_files) == 2:
        p_a = Path(args.run_files[0])
        p_b = Path(args.run_files[1])
        name_a = p_a.stem
        name_b = p_b.stem
        print(f"📖 Loading run records: [{name_a}] and [{name_b}]...")
        report_a = BenchmarkReport.load_json(p_a)
        report_b = BenchmarkReport.load_json(p_b)

    # Mode 2: Run two configurations on the fly
    elif args.config_a and args.config_b:
        p_a = Path(args.config_a)
        p_b = Path(args.config_b)
        name_a = p_a.stem
        name_b = p_b.stem

        cfg_a = System2MasterConfig.load_json(p_a)
        cfg_b = System2MasterConfig.load_json(p_b)

        suite_a = BenchmarkSuite(s2_cfg=cfg_a)
        suite_b = BenchmarkSuite(s2_cfg=cfg_b)

        sc_ids = [args.scenario] if args.scenario else None
        severities = [1.0, 1.5]

        print(f"🚀 Running benchmark sweep for [{name_a}]...")
        report_a = suite_a.run_full_benchmark(
            scenario_ids=sc_ids,
            severities=severities,
            duration_s=args.duration,
            run_name=name_a,
            config_path=str(p_a),
            verbose=False,
        )

        print(f"🚀 Running benchmark sweep for [{name_b}]...")
        report_b = suite_b.run_full_benchmark(
            scenario_ids=sc_ids,
            severities=severities,
            duration_s=args.duration,
            run_name=name_b,
            config_path=str(p_b),
            verbose=False,
        )

    else:
        print("❌ Error: Provide either two run JSON files OR --config-a and --config-b.")
        print("\nExamples:")
        print("  uv run python evals/compare_runs.py results/runs/default.json results/runs/tuned.json")
        print("  uv run python evals/compare_runs.py --config-a configs/system2_default.json --config-b configs/system2_tuned_template.json")
        sys.exit(1)

    # Perform comparison
    comp_data = compare_reports(report_a, report_b, name_a=name_a, name_b=name_b)
    print_comparison(comp_data)

    # Generate interactive HTML dashboard
    html_p = generate_comparison_html(comp_data, output_path=args.output_html)
    print(f"🌐 Interactive comparison dashboard generated at: {html_p}")


if __name__ == "__main__":
    main()

