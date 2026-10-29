"""Automated Multi-Scenario Benchmark Suite, Grid Search, and Hero Breaking-Point Finder.

Orchestrates multi-seed quantitative evaluations across all registered chaos scenarios,
computes statistical confidence metrics, conducts automated parameter sweeps (grid search),
and discovers the 'hero breaking point' where System-1 crashes but System-1+2 survives.
"""

import os
import sys
import time
import copy
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

from runtime.sim.config import System2MasterConfig
from runtime.evals.scenarios import EvalScenario, get_scenario, list_scenarios
from runtime.evals.eval_harness import run_eval, EvalResult
from runtime.evals.metrics import (
    compute_improvement_pct,
    compute_recovery_speedup,
    compute_safety_score,
)


@dataclass
class BenchmarkComparison:
    """A/B comparison between System 1 (Baseline) and System 1 + System 2 (Enhanced)."""

    scenario_id: str
    scenario_name: str
    severity: float
    s1_metrics: Dict[str, float]
    s2_metrics: Dict[str, float]
    pos_improvement_pct: float
    rms_improvement_pct: float
    recovery_speedup: float
    safety_score_s1: float
    safety_score_s2: float
    s1_crashed: bool
    s2_crashed: bool
    s2_calls_mean: float
    winner: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "severity": self.severity,
            "s1_metrics": self.s1_metrics,
            "s2_metrics": self.s2_metrics,
            "pos_improvement_pct": self.pos_improvement_pct,
            "rms_improvement_pct": self.rms_improvement_pct,
            "recovery_speedup": self.recovery_speedup,
            "safety_score_s1": self.safety_score_s1,
            "safety_score_s2": self.safety_score_s2,
            "s1_crashed": self.s1_crashed,
            "s2_crashed": self.s2_crashed,
            "s2_calls_mean": self.s2_calls_mean,
            "winner": self.winner,
        }


@dataclass
class BenchmarkReport:
    """Consolidated report spanning multiple scenarios, severities, and statistical seeds."""

    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    overall_mean_improvement_pct: float = 0.0
    overall_win_rate_pct: float = 0.0
    s1_total_crashes: int = 0
    s2_total_crashes: int = 0
    hero_scenarios_count: int = 0  # Where S1 crashed but S2 survived
    run_name: str = "default"
    config_path: Optional[str] = None
    config_dict: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config_path": self.config_path,
            "config_dict": self.config_dict,
            "timestamp": self.timestamp,
            "overall_mean_improvement_pct": self.overall_mean_improvement_pct,
            "overall_win_rate_pct": self.overall_win_rate_pct,
            "s1_total_crashes": self.s1_total_crashes,
            "s2_total_crashes": self.s2_total_crashes,
            "hero_scenarios_count": self.hero_scenarios_count,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        comps = []
        for c in data.get("comparisons", []):
            comps.append(
                BenchmarkComparison(
                    scenario_id=c["scenario_id"],
                    scenario_name=c["scenario_name"],
                    severity=float(c["severity"]),
                    s1_metrics=c["s1_metrics"],
                    s2_metrics=c["s2_metrics"],
                    pos_improvement_pct=float(c["pos_improvement_pct"]),
                    rms_improvement_pct=float(c["rms_improvement_pct"]),
                    recovery_speedup=float(c["recovery_speedup"]),
                    safety_score_s1=float(c["safety_score_s1"]),
                    safety_score_s2=float(c["safety_score_s2"]),
                    s1_crashed=bool(c["s1_crashed"]),
                    s2_crashed=bool(c["s2_crashed"]),
                    s2_calls_mean=float(c.get("s2_calls_mean", 0.0)),
                    winner=c["winner"],
                )
            )
        return cls(
            comparisons=comps,
            overall_mean_improvement_pct=float(data.get("overall_mean_improvement_pct", 0.0)),
            overall_win_rate_pct=float(data.get("overall_win_rate_pct", 0.0)),
            s1_total_crashes=int(data.get("s1_total_crashes", 0)),
            s2_total_crashes=int(data.get("s2_total_crashes", 0)),
            hero_scenarios_count=int(data.get("hero_scenarios_count", 0)),
            run_name=str(data.get("run_name", "default")),
            config_path=data.get("config_path"),
            config_dict=data.get("config_dict"),
            timestamp=float(data.get("timestamp", time.time())),
        )

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "BenchmarkReport":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class BreakingPointResult:
    """Discovered failure thresholds comparing S1-only against S1+S2."""

    scenario_id: str
    scenario_name: str
    s1_crash_severity: Optional[float]  # Lowest severity causing S1 to crash
    s2_crash_severity: Optional[float]  # Lowest severity causing S2 to crash
    survivability_gap: float            # s2_crash_severity - s1_crash_severity
    hero_range: Tuple[float, float]     # Severity range where S1 crashes but S2 flies
    evaluations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "s1_crash_severity": self.s1_crash_severity,
            "s2_crash_severity": self.s2_crash_severity,
            "survivability_gap": self.survivability_gap,
            "hero_range": list(self.hero_range),
            "evaluations": self.evaluations,
        }


# ==============================================================================
# BENCHMARK SUITE CLASS
# ==============================================================================
class BenchmarkSuite:
    """High-level testbed orchestrator for quantitative evaluation and tuning."""

    def __init__(self, s2_cfg: Optional[System2MasterConfig] = None) -> None:
        self.default_s2_cfg = s2_cfg or System2MasterConfig()

    def compare_scenario(
        self,
        scenario_id: str,
        severity: float = 1.0,
        seeds: Optional[List[int]] = None,
        duration_s: Optional[float] = None,
        s2_cfg: Optional[System2MasterConfig] = None,
        use_live_llm: bool = False,
        verbose: bool = False,
    ) -> BenchmarkComparison:
        """Runs side-by-side multi-seed statistical evaluation of a single scenario."""
        sc = get_scenario(scenario_id)
        active_seeds = seeds or [42, 123, 456]
        effective_s2 = s2_cfg or self.default_s2_cfg

        s1_results: List[EvalResult] = []
        s2_results: List[EvalResult] = []

        for seed in active_seeds:
            # 1. Run Baseline (S1 only)
            r_s1 = run_eval(
                scenario=sc,
                mode="s1_only",
                severity=severity,
                duration_s=duration_s,
                seed=seed,
                s2_cfg=effective_s2,
                record_timeseries=False,
                verbose=verbose,
            )
            s1_results.append(r_s1)

            # 2. Run Enhanced (S1 + S2)
            r_s2 = run_eval(
                scenario=sc,
                mode="s1_s2",
                severity=severity,
                duration_s=duration_s,
                seed=seed,
                s2_cfg=effective_s2,
                use_live_llm=use_live_llm,
                record_timeseries=False,
                verbose=verbose,
            )
            s2_results.append(r_s2)

        # Average metrics across seeds
        def _avg_metrics(results: List[EvalResult]) -> Dict[str, float]:
            avg: Dict[str, float] = {}
            if not results:
                return avg
            metric_keys = list(results[0].metrics.keys())
            for k in metric_keys:
                vals = [float(r.metrics.get(k, 0.0)) for r in results]
                avg[k] = float(np.mean(vals))
            return avg

        s1_avg = _avg_metrics(s1_results)
        s2_avg = _avg_metrics(s2_results)

        s1_crashed = any(r.crashed for r in s1_results)
        s2_crashed = any(r.crashed for r in s2_results)

        pos_imp = compute_improvement_pct(s1_avg.get("mean_pos_error", 1.0), s2_avg.get("mean_pos_error", 1.0))
        rms_imp = compute_improvement_pct(s1_avg.get("rms_pos_error", 1.0), s2_avg.get("rms_pos_error", 1.0))
        rec_speedup = compute_recovery_speedup(
            s1_avg.get("recovery_time", 10.0), s2_avg.get("recovery_time", 10.0)
        )

        safety_s1 = float(np.mean([r.safety_score for r in s1_results]))
        safety_s2 = float(np.mean([r.safety_score for r in s2_results]))
        s2_calls_avg = float(np.mean([float(r.metrics.get("s2_calls", 0.0)) for r in s2_results]))

        # Winner determination logic
        if s1_crashed and not s2_crashed:
            winner = "s1_s2 (SURVIVAL)"
        elif s2_crashed and not s1_crashed:
            winner = "s1_only"
        elif pos_imp > 5.0 or safety_s2 > safety_s1 + 5.0:
            winner = "s1_s2"
        elif pos_imp < -5.0:
            winner = "s1_only"
        else:
            winner = "tie"

        return BenchmarkComparison(
            scenario_id=sc.id,
            scenario_name=sc.display_name,
            severity=severity,
            s1_metrics=s1_avg,
            s2_metrics=s2_avg,
            pos_improvement_pct=round(pos_imp, 2),
            rms_improvement_pct=round(rms_imp, 2),
            recovery_speedup=round(rec_speedup, 2),
            safety_score_s1=round(safety_s1, 1),
            safety_score_s2=round(safety_s2, 1),
            s1_crashed=s1_crashed,
            s2_crashed=s2_crashed,
            s2_calls_mean=round(s2_calls_avg, 1),
            winner=winner,
        )

    def run_full_benchmark(
        self,
        scenario_ids: Optional[List[str]] = None,
        severities: Optional[List[float]] = None,
        seeds: Optional[List[int]] = None,
        duration_s: float = 6.0,
        s2_cfg: Optional[System2MasterConfig] = None,
        use_live_llm: bool = False,
        run_name: str = "default",
        config_path: Optional[str] = None,
        verbose: bool = False,
    ) -> BenchmarkReport:
        """Executes full benchmark matrix across scenarios and severities."""
        # Auto-discover all registered scenarios if none specified
        if not scenario_ids:
            scenarios = list_scenarios()
        else:
            scenarios = [get_scenario(sid) for sid in scenario_ids]

        active_severities = severities or [0.5, 1.0, 1.5, 2.0]
        comparisons: List[BenchmarkComparison] = []
        effective_cfg = s2_cfg or self.default_s2_cfg

        for sc in scenarios:
            for sev in active_severities:
                if verbose:
                    print(f"📊 Benchmarking [{sc.id}] @ Severity {sev:.1f}x...")
                comp = self.compare_scenario(
                    scenario_id=sc.id,
                    severity=sev,
                    seeds=seeds,
                    duration_s=duration_s,
                    s2_cfg=effective_cfg,
                    use_live_llm=use_live_llm,
                    verbose=verbose,
                )
                comparisons.append(comp)

        # Aggregate report statistics
        imp_vals = [c.pos_improvement_pct for c in comparisons]
        mean_imp = float(np.mean(imp_vals)) if imp_vals else 0.0

        wins = sum(1 for c in comparisons if "s1_s2" in c.winner)
        win_rate = float(wins / len(comparisons) * 100.0) if comparisons else 0.0

        s1_crashes = sum(1 for c in comparisons if c.s1_crashed)
        s2_crashes = sum(1 for c in comparisons if c.s2_crashed)
        hero_count = sum(1 for c in comparisons if c.s1_crashed and not c.s2_crashed)

        return BenchmarkReport(
            comparisons=comparisons,
            overall_mean_improvement_pct=round(mean_imp, 2),
            overall_win_rate_pct=round(win_rate, 1),
            s1_total_crashes=s1_crashes,
            s2_total_crashes=s2_crashes,
            hero_scenarios_count=hero_count,
            run_name=run_name,
            config_path=config_path,
            config_dict=effective_cfg.to_dict(),
        )

    def run_breaking_point_search(
        self,
        scenario_id: str,
        severity_min: float = 0.5,
        severity_max: float = 3.5,
        step_size: float = 0.25,
        duration_s: float = 6.0,
        s2_cfg: Optional[System2MasterConfig] = None,
        use_live_llm: bool = False,
        verbose: bool = False,
    ) -> BreakingPointResult:
        """Finds the critical severity threshold where System-1 crashes but System-1+2 survives."""
        sc = get_scenario(scenario_id)
        evals_record: List[Dict[str, Any]] = []

        s1_crash_sev: Optional[float] = None
        s2_crash_sev: Optional[float] = None

        current_sev = severity_min
        while current_sev <= severity_max + 1e-4:
            comp = self.compare_scenario(
                scenario_id=sc.id,
                severity=current_sev,
                seeds=[42],
                duration_s=duration_s,
                s2_cfg=s2_cfg,
                use_live_llm=use_live_llm,
                verbose=verbose,
            )

            rec = {
                "severity": round(current_sev, 2),
                "s1_crashed": comp.s1_crashed,
                "s2_crashed": comp.s2_crashed,
                "pos_improvement_pct": comp.pos_improvement_pct,
                "s1_pos_error": comp.s1_metrics.get("mean_pos_error", 0.0),
                "s2_pos_error": comp.s2_metrics.get("mean_pos_error", 0.0),
                "s1_tilt_max": comp.s1_metrics.get("max_tilt", 0.0),
                "s2_tilt_max": comp.s2_metrics.get("max_tilt", 0.0),
            }
            evals_record.append(rec)

            if comp.s1_crashed and s1_crash_sev is None:
                s1_crash_sev = round(current_sev, 2)
            if comp.s2_crashed and s2_crash_sev is None:
                s2_crash_sev = round(current_sev, 2)

            current_sev += step_size

        gap = 0.0
        hero_low = 0.0
        hero_high = 0.0
        if s1_crash_sev is not None:
            hero_low = s1_crash_sev
            hero_high = s2_crash_sev if s2_crash_sev is not None else severity_max
            gap = round(hero_high - hero_low, 2)

        return BreakingPointResult(
            scenario_id=sc.id,
            scenario_name=sc.display_name,
            s1_crash_severity=s1_crash_sev,
            s2_crash_severity=s2_crash_sev,
            survivability_gap=gap,
            hero_range=(hero_low, hero_high),
            evaluations=evals_record,
        )

    def run_grid_search(
        self,
        scenario_id: str,
        param_grid: Dict[str, List[Any]],
        severity: float = 1.0,
        seeds: Optional[List[int]] = None,
        duration_s: float = 5.0,
        objective_metric: str = "pos_improvement_pct",
        output_best_json: Optional[Union[str, Path]] = "configs/best_config.json",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Automated hyperparameter optimizer over System-2 configuration space."""
        sc = get_scenario(scenario_id)
        active_seeds = seeds or [42]

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        if verbose:
            print(f"\n🔍 [GRID SEARCH] Testing {len(combinations)} parameter configurations on '{sc.id}'...")

        best_score = -float("inf")
        best_cfg: Optional[System2MasterConfig] = None
        best_combination: Optional[Dict[str, Any]] = None
        trials: List[Dict[str, Any]] = []

        for idx, combo in enumerate(combinations):
            combo_dict = dict(zip(param_names, combo))
            test_cfg = copy.deepcopy(self.default_s2_cfg)

            # Apply parameters to nested config dataclasses
            for path_str, val in combo_dict.items():
                parts = path_str.split(".")
                target = test_cfg
                for p in parts[:-1]:
                    target = getattr(target, p)
                setattr(target, parts[-1], val)

            comp = self.compare_scenario(
                scenario_id=sc.id,
                severity=severity,
                seeds=active_seeds,
                duration_s=duration_s,
                s2_cfg=test_cfg,
                verbose=False,
            )

            # Extract objective score
            if objective_metric == "pos_improvement_pct":
                score = comp.pos_improvement_pct
            elif objective_metric == "safety_score":
                score = comp.safety_score_s2
            elif objective_metric == "rms_improvement_pct":
                score = comp.rms_improvement_pct
            else:
                score = float(comp.s2_metrics.get(objective_metric, 0.0))

            trial_record = {
                "trial": idx + 1,
                "params": combo_dict,
                "score": round(score, 2),
                "pos_improvement_pct": comp.pos_improvement_pct,
                "rms_pos_error": comp.s2_metrics.get("rms_pos_error", 0.0),
                "max_tilt": comp.s2_metrics.get("max_tilt", 0.0),
                "crashed": comp.s2_crashed,
            }
            trials.append(trial_record)

            if verbose:
                print(f"  • Trial #{idx+1:02d} | Score: {score:+6.1f}% | Params: {combo_dict}")

            if score > best_score:
                best_score = score
                best_cfg = test_cfg
                best_combination = combo_dict

        if verbose and best_combination:
            print(f"\n🏆 [OPTIMAL CONFIG FOUND] Score: {best_score:+.2f}% | Best params: {best_combination}")

        if output_best_json and best_cfg:
            p = Path(output_best_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            best_cfg.save_json(str(p))
            if verbose:
                print(f"💾 Saved optimal configuration to: {p}")

        return {
            "best_score": round(best_score, 2),
            "best_params": best_combination,
            "best_config": best_cfg.to_dict() if best_cfg else None,
            "trials_evaluated": len(trials),
            "all_trials": trials,
        }
