"""Focused tests for Louie findings fixes (F1-F8 + LLM timing).

Covers, without needing an LLM backend or the 3D viewer:
- F2: validate_patch clamps out-of-range values and rejects bad patches.
- F3: wind gust is a real timed event (latch, expiry, early cancel).
- Timing item 3: bridge trigger cooldown uses sim time, not wall time.
- Timing item 4: triggers correlate to fault onset for time-to-trigger.
"""

import unittest
import numpy as np

from runtime.schemas.tier2 import System2Patch, GainMultipliers, SafetyEnvelope, validate_patch
from runtime.sim.bridge import System1System2Bridge
from runtime.sim.chaos_injector import ChaosManager
from runtime.sim.config import ChaosConfig


def _sample_push(bridge, sim_time, pos_err_x=0.8):
    """Pushes one telemetry sample with a position-error trigger."""
    target = np.array([0.0, 0.0, 1.0])
    return bridge.push_telemetry(
        sim_time=sim_time,
        pos=np.array([pos_err_x, 0.0, 1.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        motor_ctrl=np.array([3.25, 3.25, 3.25, 3.25]),
        target_pos=target,
        sigma_f_hat=np.zeros(3),
        sigma_M_hat=np.zeros(3),
    )


class TestValidatePatch(unittest.TestCase):
    """F2: LLM patch validation and clamping."""

    def test_clamps_out_of_range_values(self):
        patch = System2Patch(
            diagnosis="Test overcompensation",
            feedforward_force_offset=[10.0, -10.0, 0.0],
            gain_multipliers=GainMultipliers(pos_kp=5.0, pos_kd=0.1, att_kp=1.0, att_kd=1.0),
            motor_trim_scale=[9.0, 0.1, 1.0, 1.0],
            safety_envelope=SafetyEnvelope(max_tilt_deg=45.0),
        )
        out = validate_patch(patch)
        self.assertEqual(out.feedforward_force_offset, [6.0, -6.0, 0.0])
        self.assertEqual(out.gain_multipliers.pos_kp, 2.0)
        self.assertEqual(out.gain_multipliers.pos_kd, 0.3)
        self.assertEqual(out.motor_trim_scale, [2.5, 0.5, 1.0, 1.0])
        self.assertEqual(out.safety_envelope.max_tilt_deg, 25.0)

    def test_rejects_missing_diagnosis(self):
        patch = System2Patch(diagnosis="  ", feedforward_force_offset=[1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            validate_patch(patch)

    def test_rejects_non_numeric_values(self):
        patch = System2Patch(diagnosis="Bad numbers", feedforward_force_offset=["a", 0.0, 0.0])
        with self.assertRaises(ValueError):
            validate_patch(patch)
        patch = System2Patch(diagnosis="Bad numbers", motor_trim_scale=[1.0, float("nan"), 1.0, 1.0])
        with self.assertRaises(ValueError):
            validate_patch(patch)

    def test_rejects_wrong_length_vectors(self):
        patch = System2Patch(diagnosis="Short trim", motor_trim_scale=[1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            validate_patch(patch)
        patch = System2Patch(diagnosis="Long ff", feedforward_force_offset=[1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            validate_patch(patch)

    def test_accepts_prompt_example_patches_unchanged(self):
        # F1 motor-failure trim, sensor-noise gains, and wind feedforward from
        # the SYSTEM_PROMPT must pass validation untouched.
        patch = System2Patch(
            diagnosis="Motor 1 degraded",
            motor_trim_scale=[2.50, 1.0, 1.0, 1.0],
            gain_multipliers=GainMultipliers(pos_kp=1.0, pos_kd=1.0, att_kp=0.40, att_kd=1.6),
            feedforward_force_offset=[-5.0, 0.0, 0.0],
            safety_envelope=SafetyEnvelope(max_tilt_deg=20.0),
        )
        out = validate_patch(patch)
        self.assertEqual(out.motor_trim_scale, [2.50, 1.0, 1.0, 1.0])
        self.assertEqual(out.gain_multipliers.att_kp, 0.40)
        self.assertEqual(out.gain_multipliers.att_kd, 1.6)
        self.assertEqual(out.feedforward_force_offset, [-5.0, 0.0, 0.0])

    def test_from_dict_validates(self):
        data = {
            "diagnosis": "Wind from +X",
            "feedforward_force_offset": [-50.0, 0.0, 0.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "safety_envelope": {"max_tilt_deg": 60.0},
        }
        patch = System2Patch.from_dict(data)
        self.assertEqual(patch.feedforward_force_offset, [-6.0, 0.0, 0.0])
        self.assertEqual(patch.safety_envelope.max_tilt_deg, 25.0)

        with self.assertRaises(ValueError):
            System2Patch.from_dict({"feedforward_force_offset": [1.0, 0.0, 0.0]})

    def test_bridge_apply_rejects_invalid_patch(self):
        bridge = System1System2Bridge()
        bad = System2Patch(diagnosis="", feedforward_force_offset=[1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            bridge.apply_system2_patch(bad)
        self.assertIsNone(bridge.get_active_patch())


class _StubData:
    """Minimal stand-in for mujoco.MjData for wind-gust tests."""

    def __init__(self, nbodies=3):
        self.xfrc_applied = np.zeros((nbodies, 6))


class TestTimedWindGust(unittest.TestCase):
    """F3: wind gust is a real timed event."""

    def test_gust_expires_after_duration(self):
        chaos = ChaosManager(cfg=ChaosConfig())
        data = _StubData()
        chaos.trigger_wind_gust(current_time=1.0)
        self.assertTrue(chaos.active_chaos["wind_gust"])

        chaos.apply_wind_gust(data, 1, 1.0)
        self.assertAlmostEqual(data.xfrc_applied[1, 0], 5.0)

        data2 = _StubData()
        chaos.apply_wind_gust(data2, 1, 1.5)  # before 1.0 + 0.8 expiry
        self.assertAlmostEqual(data2.xfrc_applied[1, 0], 5.0)
        self.assertTrue(chaos.active_chaos["wind_gust"])

        data3 = _StubData()
        chaos.apply_wind_gust(data3, 1, 1.9)  # past expiry
        self.assertAlmostEqual(data3.xfrc_applied[1, 0], 0.0)
        self.assertFalse(chaos.active_chaos["wind_gust"])

    def test_gust_latches_on_direct_flag_set(self):
        # Scripts/demo set the flag directly; end time must still latch.
        chaos = ChaosManager(cfg=ChaosConfig())
        data = _StubData()
        chaos.active_chaos["wind_gust"] = True
        chaos.apply_wind_gust(data, 1, 5.0)
        self.assertAlmostEqual(data.xfrc_applied[1, 0], 5.0)
        self.assertAlmostEqual(chaos.gust_end_time, 5.8)

    def test_hotkey_cancels_gust_early(self):
        chaos = ChaosManager(cfg=ChaosConfig())
        chaos.handle_key(57)  # '9': wind gust on
        self.assertTrue(chaos.active_chaos["wind_gust"])
        chaos.handle_key(57)  # '9' again: cancel early
        self.assertFalse(chaos.active_chaos["wind_gust"])
        self.assertEqual(chaos.gust_end_time, 0.0)


class TestBridgeTiming(unittest.TestCase):
    """Timing items 3-4: sim-time cooldown and fault correlation."""

    def test_cooldown_uses_sim_time(self):
        bridge = System1System2Bridge()
        _sample_push(bridge, 1.0)
        event = bridge.get_latest_trigger_event(current_time=1.0)
        self.assertIsNotNone(event)
        self.assertAlmostEqual(event.timestamp, 1.0)

        # 0.2 s later in sim time: inside the 0.5 s cooldown, no new event,
        # even though wall time barely advanced.
        _sample_push(bridge, 1.2)
        event = bridge.get_latest_trigger_event(current_time=1.2)
        self.assertAlmostEqual(event.timestamp, 1.0)

        # Past the cooldown: new event recorded.
        _sample_push(bridge, 1.6)
        event = bridge.get_latest_trigger_event(current_time=1.6)
        self.assertAlmostEqual(event.timestamp, 1.6)

    def test_trigger_correlates_fault_and_time_to_trigger(self):
        bridge = System1System2Bridge()
        bridge.note_fault_state({"motor_failure": True}, 2.0)
        bridge.note_fault_state({"motor_failure": True}, 2.2)
        _sample_push(bridge, 2.3)
        event = bridge.get_latest_trigger_event(current_time=2.3)
        self.assertIsNotNone(event)
        self.assertEqual(event.fault_type, "motor_failure")
        self.assertAlmostEqual(event.time_to_trigger_s, 0.3)


if __name__ == "__main__":
    unittest.main()
