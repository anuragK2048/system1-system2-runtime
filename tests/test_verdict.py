"""Unit tests for the pass/fail verdict layer and the time-to-detect /
time-to-patch timing metrics.

These tests are mujoco-free: they exercise runtime.evals.verdict and the
metric functions in runtime.evals.metrics directly with synthetic step
records, so they run anywhere the evals package imports.
"""

import json
import tempfile
import unittest
from pathlib import Path

from runtime.evals.verdict import (
    DEFAULT_THRESHOLDS,
    compute_verdict,
    load_thresholds,
)
from runtime.evals.metrics import (
    compute_all_metrics,
    get_metric_definition,
)


def _steps(n=20, onset=2.0, trigger_at=None, patch_at=None):
    """Builds synthetic step records with timing anchors stamped."""
    steps = []
    for i in range(n):
        t = onset + i * 0.1
        steps.append(
            {
                "t": round(t, 4),
                "pos_error": 0.1,
                "fault_onset_s": round(onset, 4),
                "trigger_t": round(trigger_at, 4) if trigger_at is not None and t >= trigger_at else None,
                "patch_applied_t": round(patch_at, 4) if patch_at is not None and t >= patch_at else None,
            }
        )
    return steps


class TestTimingMetrics(unittest.TestCase):
    def test_registered(self):
        d1 = get_metric_definition("t_detect_s")
        d2 = get_metric_definition("t_patch_s")
        self.assertEqual(d1.unit, "s")
        self.assertEqual(d2.unit, "s")
        self.assertFalse(d1.higher_is_better)

    def test_detect_and_patch(self):
        m = compute_all_metrics(_steps(trigger_at=2.5, patch_at=3.0))
        self.assertAlmostEqual(m["t_detect_s"], 0.5, places=6)
        self.assertAlmostEqual(m["t_patch_s"], 1.0, places=6)

    def test_no_trigger_no_patch(self):
        m = compute_all_metrics(_steps())
        self.assertIsNone(m["t_detect_s"])
        self.assertIsNone(m["t_patch_s"])

    def test_trigger_but_no_patch(self):
        m = compute_all_metrics(_steps(trigger_at=2.2))
        self.assertAlmostEqual(m["t_detect_s"], 0.2, places=6)
        self.assertIsNone(m["t_patch_s"])

    def test_missing_onset_returns_none(self):
        steps = [{"t": 1.0, "trigger_t": 1.5}]
        m = compute_all_metrics(steps)
        self.assertIsNone(m["t_detect_s"])
        self.assertIsNone(m["t_patch_s"])

    def test_empty_steps(self):
        # Registry contract: compute() returns 0.0 for empty input (all metrics).
        m = compute_all_metrics([])
        self.assertEqual(m["t_detect_s"], 0.0)
        self.assertEqual(m["t_patch_s"], 0.0)


class TestVerdict(unittest.TestCase):
    def _good_metrics(self):
        return {
            "crash": False,
            "max_pos_error": 0.8,
            "recovery_time": 2.5,
            "max_tilt": 25.0,
            "t_detect_s": 0.4,
            "t_patch_s": 1.2,
        }

    def test_pass(self):
        v = compute_verdict(self._good_metrics())
        self.assertEqual(v["verdict"], "pass")
        self.assertEqual(v["failed_criteria"], [])

    def test_crash_fails(self):
        m = self._good_metrics()
        m["crash"] = True
        v = compute_verdict(m)
        self.assertEqual(v["verdict"], "fail")
        self.assertEqual(len(v["failed_criteria"]), 1)
        self.assertEqual(v["failed_criteria"][0]["metric"], "crash")

    def test_multiple_failures_listed(self):
        m = self._good_metrics()
        m["max_pos_error"] = 2.5
        m["max_tilt"] = 70.0
        v = compute_verdict(m)
        self.assertEqual(v["verdict"], "fail")
        reasons = [f["metric"] for f in v["failed_criteria"]]
        self.assertIn("max_pos_error", reasons)
        self.assertIn("max_tilt", reasons)

    def test_timing_thresholds(self):
        m = self._good_metrics()
        m["t_detect_s"] = 5.0
        v = compute_verdict(m)
        self.assertEqual(v["verdict"], "fail")
        self.assertEqual(v["failed_criteria"][0]["metric"], "t_detect_s")

    def test_none_metrics_skipped_not_failed(self):
        # s1_only run: no trigger, no patch -> None timings must not fail.
        m = self._good_metrics()
        m["t_detect_s"] = None
        m["t_patch_s"] = None
        v = compute_verdict(m)
        self.assertEqual(v["verdict"], "pass")

    def test_custom_thresholds(self):
        th = dict(DEFAULT_THRESHOLDS)
        th["max_pos_error_m"] = 0.5
        v = compute_verdict(self._good_metrics(), thresholds=th)
        self.assertEqual(v["verdict"], "fail")

    def test_thresholds_echoed(self):
        v = compute_verdict(self._good_metrics())
        self.assertEqual(v["thresholds"]["max_pos_error_m"], 1.5)

    def test_load_thresholds_explicit_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"max_tilt_deg": 45.0}, f)
            path = f.name
        try:
            th = load_thresholds(path)
            self.assertEqual(th["max_tilt_deg"], 45.0)
            # Untouched keys fall back to defaults.
            self.assertEqual(th["max_pos_error_m"], DEFAULT_THRESHOLDS["max_pos_error_m"])
        finally:
            Path(path).unlink()

    def test_load_thresholds_ignores_notes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"_notes": "hello", "t_detect_s": 3.0}, f)
            path = f.name
        try:
            th = load_thresholds(path)
            self.assertNotIn("_notes", th)
            self.assertEqual(th["t_detect_s"], 3.0)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
