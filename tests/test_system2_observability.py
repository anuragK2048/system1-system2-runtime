"""Unit and integration tests for System 2 latency, call timestamps, and patch timer."""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from runtime.schemas.tier2 import System2Patch, GainMultipliers
from runtime.sim.bridge import System1System2Bridge
from runtime.brain.llm import System2Brain, System2Worker
from runtime.server.state import sim_context


class TestSystem2Observability(unittest.TestCase):
    """Verifies System 2 timing, latency tracking, and patch age calculation."""

    def test_bridge_patch_applied_timer(self):
        """Verifies that System1System2Bridge records patch applied time and computes elapsed age."""
        bridge = System1System2Bridge()
        self.assertIsNone(bridge.get_patch_applied_time())
        self.assertIsNone(bridge.get_patch_age_s())

        patch = System2Patch(
            diagnosis="Motor 1 degraded test",
            feedforward_force_offset=[0.0, 0.0, 0.0],
            motor_trim_scale=[1.6, 0.95, 0.95, 1.0],
        )

        t_before = time.time()
        bridge.apply_system2_patch(patch)
        t_after = time.time()

        applied_t = bridge.get_patch_applied_time()
        self.assertIsNotNone(applied_t)
        self.assertTrue(t_before <= applied_t <= t_after)

        time.sleep(0.05)
        age = bridge.get_patch_age_s()
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 0.04)

        # Clear patch resets timer
        bridge.clear_active_patch()
        self.assertIsNone(bridge.get_patch_applied_time())
        self.assertIsNone(bridge.get_patch_age_s())

        # Apply again and reset bridge
        bridge.apply_system2_patch(patch)
        self.assertIsNotNone(bridge.get_patch_applied_time())
        bridge.reset()
        self.assertIsNone(bridge.get_patch_applied_time())
        self.assertIsNone(bridge.get_patch_age_s())

    def test_brain_timing_and_latency_tracking(self):
        """Verifies that System2Brain tracks dispatched/received timestamps and call latency."""
        brain = System2Brain(provider="groq", api_key="test_key_mock")
        self.assertFalse(brain.in_flight)

        # Mock client chat completion
        mock_completion = MagicMock()
        mock_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_choice = MagicMock()
        mock_choice.message.content = '{"diagnosis": "High wind along +X", "feedforward_force_offset": [-5.0, 0.0, 0.0], "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0}, "motor_trim_scale": [1.0, 1.0, 1.0, 1.0], "safety_envelope": {"max_tilt_deg": 20.0}}'
        mock_choice.finish_reason = "stop"
        mock_completion.choices = [mock_choice]
        brain.client.chat.completions.create = MagicMock(return_value=mock_completion)

        from runtime.schemas.telemetry import TelemetrySummary
        summary = TelemetrySummary(
            timestamp=5.0,
            window_duration_s=1.0,
            samples_count=500,
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            gyro_rms=[0.1, 0.1, 0.1],
            oscillation_detected=False,
            oscillation_note="none",
            pos_error=[-1.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            motor_commands_mean=[3.25, 3.25, 3.25, 3.25],
            motor_saturation_pct=[0.0, 0.0, 0.0, 0.0],
            unmodeled_force_disturbance=[5.0, 0.0, 0.0],
            unmodeled_moment_disturbance=[0.0, 0.0, 0.0],
        )

        t_before = time.time()
        result_patch = brain.analyze_and_patch(summary)
        t_after = time.time()

        self.assertIsNotNone(result_patch)
        self.assertFalse(brain.in_flight)
        self.assertIsNotNone(brain.last_call_dispatched_at)
        self.assertIsNotNone(brain.last_call_received_at)
        self.assertIsNotNone(brain.last_latency_s)
        self.assertTrue(t_before <= brain.last_call_dispatched_at <= t_after)
        self.assertTrue(brain.last_call_dispatched_at <= brain.last_call_received_at <= t_after + 0.1)
        self.assertGreaterEqual(brain.last_latency_s, 0.0)
        self.assertEqual(brain.call_count, 1)

    def test_worker_get_status_dict(self):
        """Verifies that System2Worker exposes complete status dict with latency and timer."""
        bridge = System1System2Bridge()
        brain = System2Brain(provider="groq", api_key="test_key_mock")
        worker = System2Worker(bridge=bridge, brain=brain, drone_name="Drone 2 Test")

        status = worker.get_status_dict()
        self.assertEqual(status["drone_name"], "Drone 2 Test")
        self.assertEqual(status["provider"], "groq")
        self.assertIn("in_flight", status)
        self.assertIn("last_latency_s", status)
        self.assertIn("last_call_dispatched_at", status)
        self.assertIn("last_call_received_at", status)
        self.assertIn("patch_age_s", status)
        self.assertIsNone(status["patch_age_s"])

        # When patch applied
        patch = System2Patch(
            diagnosis="Test patch",
            feedforward_force_offset=[0.0, 0.0, 0.0],
        )
        bridge.apply_system2_patch(patch)
        time.sleep(0.02)
        status_patched = worker.get_status_dict()
        self.assertIsNotNone(status_patched["patch_applied_time"])
        self.assertIsNotNone(status_patched["patch_age_s"])
        self.assertGreaterEqual(status_patched["patch_age_s"], 0.01)


if __name__ == "__main__":
    unittest.main()
