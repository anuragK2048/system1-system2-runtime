"""Evaluation, Benchmarking, and Parameter Tuning Suite for System-1 and System-2."""

from runtime.evals.scenarios import (
    EvalScenario,
    register_scenario,
    get_scenario,
    list_scenarios,
)
from runtime.evals.metrics import (
    MetricDefinition,
    StepRecord,
    register_metric,
    compute_all_metrics,
    get_metric_definition,
    list_metrics,
    compute_improvement_pct,
    compute_safety_score,
    compute_recovery_speedup,
)
from runtime.evals.eval_harness import (
    EvalResult,
    run_eval,
)
from runtime.evals.benchmark_suite import (
    BenchmarkComparison,
    BenchmarkReport,
    BreakingPointResult,
    BenchmarkSuite,
)
from runtime.evals.report_generator import (
    print_comparison_table,
    print_breaking_point,
    generate_html_dashboard,
)
from runtime.evals.comparator import (
    diff_configs,
    compare_reports,
    print_comparison,
    generate_comparison_html,
)

__all__ = [
    "EvalScenario",
    "register_scenario",
    "get_scenario",
    "list_scenarios",
    "MetricDefinition",
    "StepRecord",
    "register_metric",
    "compute_all_metrics",
    "get_metric_definition",
    "list_metrics",
    "compute_improvement_pct",
    "compute_safety_score",
    "compute_recovery_speedup",
    "EvalResult",
    "run_eval",
    "BenchmarkComparison",
    "BenchmarkReport",
    "BreakingPointResult",
    "BenchmarkSuite",
    "print_comparison_table",
    "print_breaking_point",
    "generate_html_dashboard",
    "diff_configs",
    "compare_reports",
    "print_comparison",
    "generate_comparison_html",
]
