"""Unified Multimodal VLM Evaluation & Benchmarking CLI.

Provides a unified interface across all flight reasoner benchmark suites:
  1. 'single' (Default): Single-Drone Visual Navigation & Target Tracking (SD-01 to SD-07)
     - Pad Alpha, Pedestal Delta, Pad Bravo, Pad Charlie, Pylon Gate, Wind Gust, Degraded Motor.
  2. 'canonical': Dual-Drone Disturbance & Corridor Evasion (TC-01 to TC-06)
     - Headwind compensation, motor trim, gyro chatter damping, lateral obstacle bypass.
  3. 'new': Out-of-Sample Validation Suite (TC-07 to TC-12)
     - Crosswinds, touchdown, payload oscillation, motor failure under forward flight.

Usage:
    # 1. Ground Truth verification on Single-Drone Navigation Suite:
    uv run python vlm/evaluate.py --suite single

    # 2. Benchmark Google Gemini on Single-Drone Suite:
    uv run python vlm/evaluate.py --suite single --backend gemini

    # 3. Benchmark Local Ollama Qwen2.5-VL:
    uv run python vlm/evaluate.py --suite single --backend ollama --model qwen2.5vl:3b

    # 4. Benchmark Canonical Dual-Drone Disturbance Suite:
    uv run python vlm/evaluate.py --suite canonical --backend gemini
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm.evaluate_target_navigation import run_benchmark as run_single_evaluator
from vlm.evaluate_model import run_evaluator as run_dual_evaluator


SUITE_MAP = {
    "single": {
        "desc": "Single-Drone Target Navigation & Waypoint Guidance (SD-01..14)",
        "file": "vlm/test_cases_single.json",
        "runner": "single",
    },
    "canonical": {
        "desc": "Dual-Drone Canonical Disturbance & Corridor Evasion (TC-01..06)",
        "file": "vlm/test_cases.json",
        "runner": "dual",
    },
    "new": {
        "desc": "Dual-Drone Out-of-Sample Generalization Suite (TC-07..12)",
        "file": "vlm/test_cases_new.json",
        "runner": "dual",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Multimodal VLM Evaluation & Benchmarking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="single",
        choices=["single", "canonical", "new"],
        help="Benchmark test suite: 'single' (SD-01..07, default), 'canonical' (TC-01..06), or 'new' (TC-07..12)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "gemini", "gemini-live", "live", "ollama", "mlx", "cloud"],
        help="Inference backend to benchmark (default: ground_truth, supports 'gemini-live' over WebSockets)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. 'gemini-3.5-flash-lite', 'qwen2.5vl:3b')",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Optional custom JSON test file path (overrides --suite file)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostic field diffs for failed checks",
    )

    args = parser.parse_args()
    suite_info = SUITE_MAP[args.suite]
    test_file = args.test_file or suite_info["file"]

    print("=" * 76)
    print("🚁 [MULTIMODAL VLM UNIFIED BENCHMARK]")
    print(f"   Suite:   {args.suite.upper()} - {suite_info['desc']}")
    print(f"   File:    {test_file}")
    print(f"   Backend: {args.backend.upper()}" + (f" ({args.model})" if args.model else ""))
    print("=" * 76)

    if suite_info["runner"] == "single":
        run_single_evaluator(
            backend=args.backend,
            model=args.model,
            test_file=Path(test_file),
            verbose=args.verbose,
        )
    else:
        run_dual_evaluator(
            backend=args.backend,
            model=args.model,
            test_file=test_file,
        )


if __name__ == "__main__":
    main()
