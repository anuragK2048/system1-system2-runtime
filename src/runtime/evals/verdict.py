"""Pass/fail verdict layer for evaluation runs.

Computes a per-run PASS/FAIL verdict from metric values against configurable
thresholds. Thresholds follow the repo's JSON config pattern: pass an explicit
path, or drop ``configs/verdict_default.json`` in the repo root to override
the built-in defaults below.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Built-in defaults. Rationale:
# - crash: any crash is an automatic FAIL (matches the spec's max_crash_rate 0.0).
# - max_pos_error_m 1.5: from the spec's pass_criteria (max_pos_err_m).
# - recovery_time_s 5.0: from the spec's pass_criteria (max_t_recover_s).
# - max_tilt_deg 60.0: below the harness crash rule (85 deg) with margin;
#   beyond 60 deg the drone is tumbling, not recovering.
# - t_detect_s 2.0: generous; fast triggers normally fire within ~0.5 s of
#   fault onset at 500 Hz.
# - t_patch_s 10.0: generous; deterministic patches apply <1 s after trigger,
#   live-LLM patches within a few seconds.
DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "crash": False,
    "max_pos_error_m": 1.5,
    "recovery_time_s": 5.0,
    "max_tilt_deg": 60.0,
    "t_detect_s": 2.0,
    "t_patch_s": 10.0,
}

# metric key -> (threshold key, human label, must_be_below?)
_CRITERIA = [
    ("crash", "crash", "no crash", False),
    ("max_pos_error", "max_pos_error_m", "max position error", True),
    ("recovery_time", "recovery_time_s", "recovery time", True),
    ("max_tilt", "max_tilt_deg", "max tilt", True),
    ("t_detect_s", "t_detect_s", "time to detect", True),
    ("t_patch_s", "t_patch_s", "time to patch", True),
]

_thresholds_cache: Optional[Dict[str, Any]] = None


def _repo_config_path() -> Path:
    # verdict.py lives at src/runtime/evals/verdict.py -> repo root is 4 up.
    return Path(__file__).resolve().parents[3] / "configs" / "verdict_default.json"


def load_thresholds(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads verdict thresholds.

    Precedence: explicit ``path`` > ``configs/verdict_default.json`` >
    built-in :data:`DEFAULT_THRESHOLDS`. A JSON file only needs to contain
    the keys it overrides.
    """
    global _thresholds_cache
    if path is None:
        if _thresholds_cache is not None:
            return _thresholds_cache
        cfg_path = _repo_config_path()
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(DEFAULT_THRESHOLDS)
            merged.update({k: v for k, v in data.items() if not k.startswith("_")})
            _thresholds_cache = merged
            return merged
        _thresholds_cache = dict(DEFAULT_THRESHOLDS)
        return _thresholds_cache
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update({k: v for k, v in data.items() if not k.startswith("_")})
    return merged


def compute_verdict(
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes a PASS/FAIL verdict for one evaluation run.

    Every criterion must hold for a PASS. A metric value of None means "not
    applicable" (e.g. t_detect_s on an s1_only run with no trigger) and that
    criterion is skipped rather than failed.

    Returns a dict with:
      - verdict: "pass" or "fail"
      - failed_criteria: list of {criterion, metric, value, threshold, reason}
      - thresholds: the threshold set that was applied
    """
    th = thresholds if thresholds is not None else load_thresholds()
    failed: List[Dict[str, Any]] = []

    for metric_key, th_key, label, below in _CRITERIA:
        val = metrics.get(metric_key)
        if val is None:
            continue  # not applicable -> skip, do not fail
        limit = th.get(th_key)
        if limit is None:
            continue
        if metric_key == "crash":
            ok = not bool(val)
        else:
            try:
                ok = float(val) <= float(limit)
            except (TypeError, ValueError):
                continue
        if not ok:
            if metric_key == "crash":
                reason = "crashed"
            else:
                unit = "s" if th_key.endswith("_s") else ("deg" if "tilt" in th_key else "m")
                reason = f"{label} {float(val):.2f}{unit} > {float(limit):.2f}{unit}"
            failed.append(
                {
                    "criterion": label,
                    "metric": metric_key,
                    "value": val,
                    "threshold": limit,
                    "reason": reason,
                }
            )

    return {
        "verdict": "fail" if failed else "pass",
        "failed_criteria": failed,
        "thresholds": th,
    }
