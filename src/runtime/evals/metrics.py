"""Extensible Metric Registry and Quantitative Performance Calculators.

Every evaluation run collects 500 Hz per-step telemetry frames (`StepRecord`).
Metrics are registered as pure functions mapping `List[StepRecord]` -> float or bool.
Adding new metrics is a single `register_metric(...)` call that automatically takes effect
in the evaluation harness, multi-scenario benchmarks, grid searches, and reporting engines.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
import numpy as np

# StepRecord is a structured dictionary containing all physical states per 500 Hz step.
StepRecord = Dict[str, Any]
MetricFunction = Callable[[List[StepRecord]], Any]


@dataclass
class MetricDefinition:
    """Metadata describing a single evaluatable flight dynamic metric."""

    id: str
    display_name: str
    unit: str
    fn: MetricFunction
    higher_is_better: bool = False
    format_spec: str = ".3f"

    def compute(self, steps: List[StepRecord]) -> Any:
        """Evaluates the metric over a sequence of step records."""
        if not steps:
            return 0.0
        try:
            return self.fn(steps)
        except Exception:
            return 0.0

    def format_value(self, val: Any) -> str:
        """Formats the metric value according to unit and format spec."""
        if isinstance(val, bool):
            return "YES" if val else "NO"
        if isinstance(val, (int, float, np.floating, np.integer)):
            formatted = f"{float(val):{self.format_spec}}"
            return f"{formatted} {self.unit}".strip()
        return str(val)


# ==============================================================================
# GLOBAL METRIC REGISTRY
# ==============================================================================
class MetricRegistry:
    """Central registry of flight dynamic and fault-tolerance metrics."""

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricDefinition] = {}

    def register(
        self,
        id: str,
        display_name: str,
        unit: str,
        fn: MetricFunction,
        higher_is_better: bool = False,
        format_spec: str = ".3f",
    ) -> None:
        canon_id = id.lower().strip()
        self._metrics[canon_id] = MetricDefinition(
            id=canon_id,
            display_name=display_name,
            unit=unit,
            fn=fn,
            higher_is_better=higher_is_better,
            format_spec=format_spec,
        )

    def get(self, id: str) -> MetricDefinition:
        canon_id = id.lower().strip()
        if canon_id not in self._metrics:
            raise KeyError(f"Unknown metric '{id}'. Available: {list(self._metrics.keys())}")
        return self._metrics[canon_id]

    def list_all(self) -> List[MetricDefinition]:
        return list(self._metrics.values())

    def compute_all(self, steps: List[StepRecord]) -> Dict[str, Any]:
        """Runs all registered metrics over a series of steps."""
        results: Dict[str, Any] = {}
        for m_id, m_def in self._metrics.items():
            results[m_id] = m_def.compute(steps)
        return results


_GLOBAL_METRICS = MetricRegistry()


def register_metric(
    id: str,
    display_name: str,
    unit: str,
    fn: MetricFunction,
    higher_is_better: bool = False,
    format_spec: str = ".3f",
) -> None:
    """Registers a new metric into the global registry."""
    _GLOBAL_METRICS.register(
        id=id,
        display_name=display_name,
        unit=unit,
        fn=fn,
        higher_is_better=higher_is_better,
        format_spec=format_spec,
    )


def compute_all_metrics(steps: List[StepRecord]) -> Dict[str, Any]:
    """Computes all registered metrics for a run."""
    return _GLOBAL_METRICS.compute_all(steps)


def get_metric_definition(id: str) -> MetricDefinition:
    return _GLOBAL_METRICS.get(id)


def list_metrics() -> List[MetricDefinition]:
    return _GLOBAL_METRICS.list_all()


# ==============================================================================
# BUILT-IN CORE FLIGHT DYNAMICS METRICS
# ==============================================================================

def _mean_pos_error(steps: List[StepRecord]) -> float:
    errs = [float(s.get("pos_error", 0.0)) for s in steps]
    return float(np.mean(errs)) if errs else 0.0


def _rms_pos_error(steps: List[StepRecord]) -> float:
    errs = [float(s.get("pos_error", 0.0)) for s in steps]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0


def _max_pos_error(steps: List[StepRecord]) -> float:
    errs = [float(s.get("pos_error", 0.0)) for s in steps]
    return float(np.max(errs)) if errs else 0.0


def _settling_error(steps: List[StepRecord]) -> float:
    """Computes mean position error during the final 20% of the flight."""
    if not steps:
        return 0.0
    tail_count = max(1, int(len(steps) * 0.20))
    tail = steps[-tail_count:]
    return float(np.mean([s.get("pos_error", 0.0) for s in tail]))


def _mean_tilt(steps: List[StepRecord]) -> float:
    tilts = [float(s.get("tilt_deg", 0.0)) for s in steps]
    return float(np.mean(tilts)) if tilts else 0.0


def _max_tilt(steps: List[StepRecord]) -> float:
    tilts = [float(s.get("tilt_deg", 0.0)) for s in steps]
    return float(np.max(tilts)) if tilts else 0.0


def _crash_detector(steps: List[StepRecord]) -> bool:
    """Returns True if the drone ever crashed or reached an irrecoverable state."""
    for s in steps:
        if s.get("crashed", False):
            return True
        alt = s.get("altitude", 1.0)
        tilt = s.get("tilt_deg", 0.0)
        # Ground contact or extreme tumble
        if alt < 0.12 or tilt > 85.0:
            return True
    return False


def _control_effort(steps: List[StepRecord]) -> float:
    """Mean quadratic deviation of motor thrusts from nominal 3.25N hover weight."""
    if not steps:
        return 0.0
    efforts = []
    for s in steps:
        cmds = np.array(s.get("motor_cmds", [3.25, 3.25, 3.25, 3.25]), dtype=float)
        efforts.append(np.sum(np.square(cmds - 3.25)))
    return float(np.mean(efforts))


def _motor_saturation_pct(steps: List[StepRecord]) -> float:
    """Percentage of steps where any rotor operated at or near upper saturation (>=12.95N)."""
    if not steps:
        return 0.0
    sat_count = 0
    for s in steps:
        cmds = s.get("motor_cmds", [0.0, 0.0, 0.0, 0.0])
        if any(m >= 12.95 for m in cmds):
            sat_count += 1
    return float(sat_count / len(steps) * 100.0)


def _s2_call_count(steps: List[StepRecord]) -> int:
    return sum(1 for s in steps if s.get("s2_called", False))


def _time_to_detect(steps: List[StepRecord]) -> Optional[float]:
    """Sim seconds from fault onset to the first System-2 trigger event.

    Fault-onset source: the harness records ``fault_onset_s`` on every step
    at the moment the scenario fault is applied (post-warmup). The trigger
    time comes from the bridge FastTriggerEvent timestamp stamped as
    ``trigger_t``. Returns None when no trigger fired (e.g. s1_only runs).
    """
    if not steps:
        return None
    onset = steps[0].get("fault_onset_s")
    if onset is None:
        return None
    for s in steps:
        tt = s.get("trigger_t")
        if tt is not None:
            return float(tt) - float(onset)
    return None


def _time_to_patch(steps: List[StepRecord]) -> Optional[float]:
    """Sim seconds from fault onset to the first System-2 patch application.

    Fault-onset source: same harness ``fault_onset_s`` anchor as
    :func:`_time_to_detect`. The patch time is the sim time at which
    ``bridge.apply_system2_patch`` was called, stamped as
    ``patch_applied_t``. Returns None when no patch was applied.
    """
    if not steps:
        return None
    onset = steps[0].get("fault_onset_s")
    if onset is None:
        return None
    for s in steps:
        pt = s.get("patch_applied_t")
        if pt is not None:
            return float(pt) - float(onset)
    return None


def _altitude_sag(steps: List[StepRecord]) -> float:
    """Maximum downward altitude sag below target altitude."""
    if not steps:
        return 0.0
    sags = []
    for s in steps:
        target_z = float(s.get("target_z", 1.0))
        actual_z = float(s.get("altitude", target_z))
        sags.append(max(0.0, target_z - actual_z))
    return float(np.max(sags)) if sags else 0.0


def _recovery_time(steps: List[StepRecord]) -> float:
    """Calculates time from disturbance onset until error is sustained below 0.20m for >= 0.5s."""
    if not steps or len(steps) < 2:
        return 0.0

    dt = steps[1].get("t", 0.002) - steps[0].get("t", 0.0)
    if dt <= 0:
        dt = 0.002

    threshold_err = 0.20
    sustain_steps = int(0.5 / dt)  # 250 steps at 500 Hz

    consecutive_settled = 0
    t_start = steps[0].get("t", 0.0)
    total_time = steps[-1].get("t", 0.0) - t_start

    for idx, s in enumerate(steps):
        err = float(s.get("pos_error", 0.0))
        if err <= threshold_err:
            consecutive_settled += 1
            if consecutive_settled >= sustain_steps:
                recovery_idx = idx - sustain_steps
                return float(max(0.0, steps[recovery_idx].get("t", 0.0) - t_start))
        else:
            consecutive_settled = 0

    return float(total_time)


# Register all standard built-in metrics
register_metric("mean_pos_error", "Mean Position Error", "m", _mean_pos_error, higher_is_better=False)
register_metric("rms_pos_error", "RMS Position Error", "m", _rms_pos_error, higher_is_better=False)
register_metric("max_pos_error", "Max Position Error", "m", _max_pos_error, higher_is_better=False)
register_metric("settling_error", "Settling Error (Tail 20%)", "m", _settling_error, higher_is_better=False)
register_metric("recovery_time", "Recovery Time (<0.2m)", "s", _recovery_time, higher_is_better=False, format_spec=".2f")
register_metric("mean_tilt", "Mean Tilt Angle", "°", _mean_tilt, higher_is_better=False, format_spec=".1f")
register_metric("max_tilt", "Max Tilt Angle", "°", _max_tilt, higher_is_better=False, format_spec=".1f")
register_metric("altitude_sag", "Max Altitude Sag", "m", _altitude_sag, higher_is_better=False)
register_metric("crash", "Crashed / Ground Tumble", "", _crash_detector, higher_is_better=False)
register_metric("control_effort", "Excess Control Effort", "N²", _control_effort, higher_is_better=False, format_spec=".1f")
register_metric("motor_saturation_pct", "Motor Saturation Rate", "%", _motor_saturation_pct, higher_is_better=False, format_spec=".1f")
register_metric("s2_calls", "System-2 Reasoning Invocations", "", _s2_call_count, higher_is_better=False, format_spec="d")
register_metric("t_detect_s", "Time to Detect (fault onset → S2 trigger)", "s", _time_to_detect, higher_is_better=False, format_spec=".2f")
register_metric("t_patch_s", "Time to Patch (fault onset → S2 patch)", "s", _time_to_patch, higher_is_better=False, format_spec=".2f")


# ==============================================================================
# COMPARATIVE DERIVED SCORES (Baseline vs Enhanced)
# ==============================================================================

def compute_improvement_pct(baseline_val: float, enhanced_val: float) -> float:
    """Computes percentage improvement where lower is better (e.g. position error)."""
    b = float(baseline_val)
    e = float(enhanced_val)
    if b <= 1e-6:
        return 0.0
    return float((b - e) / b * 100.0)


def compute_safety_score(metrics: Dict[str, Any]) -> float:
    """Calculates a composite safety score (0 to 100).

    Penalizes crashes (instant 0), excessive tilt angles, and large position deviations.
    """
    if bool(metrics.get("crash", False)):
        return 0.0

    max_tilt = float(metrics.get("max_tilt", 0.0))
    tilt_penalty = min(1.0, max_tilt / 80.0)  # Near 80° is dangerous

    rms_err = float(metrics.get("rms_pos_error", 0.0))
    err_penalty = min(1.0, rms_err / 2.0)

    score = 100.0 * (1.0 - 0.6 * tilt_penalty - 0.4 * err_penalty)
    return float(max(0.0, min(100.0, score)))


def compute_recovery_speedup(baseline_recovery_s: float, enhanced_recovery_s: float) -> float:
    """Computes recovery time speedup factor (e.g., 2.5x faster)."""
    e = max(0.05, float(enhanced_recovery_s))
    b = max(0.05, float(baseline_recovery_s))
    return float(b / e)
