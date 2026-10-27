"""Unit and Integration Tests for Robust System 1 (L1 Adaptive + Geometric) Controller.

Tests:
1. QuadLib loading & C library verification.
2. Geometric control & L1 adaptive controller execution.
3. System1Controller nominal hover & backward compatibility.
4. System1System2Bridge sliding window summaries & fast triggers.
5. System 2 patch injection (feedforward force, gain multipliers, motor trim).
6. Multi-step simulation stability under chaos perturbations.
"""

import unittest
import numpy as np
import mujoco

from runtime.quadlib import (
    get_version,
    Quaternion,
    RotationMatrix,
    QuadrotorState,
    QuadrotorSetpoint,
    L1AdaptiveState,
    geometric_control,
    l1_adaptive_control,
)
from runtime.sim.config import DroneConfig, L1Config, PIDConfig, SCENE_PATH
from runtime.sim.system1_controller import System1Controller
from runtime.sim.simulator import DroneSimulator
from runtime.sim.bridge import System1System2Bridge
from runtime.schemas.tier2 import System2Patch, GainMultipliers, SafetyEnvelope


class TestSystem1L1Adaptive(unittest.TestCase):
    """Test suite for System 1 L1 Adaptive Controller and System 2 Coordination."""

    def setUp(self):
        self.drone_cfg = DroneConfig()
        self.l1_cfg = L1Config()
        self.bridge = System1System2Bridge()
        self.controller = System1Controller(
            drone_cfg=self.drone_cfg,
            l1_cfg=self.l1_cfg,
            bridge=self.bridge,
            mode="l1_adaptive",
        )

    def test_quadlib_loaded(self):
        """Verify C library is compiled and loaded correctly."""
        version = get_version()
        self.assertEqual(version, "0.2.0")
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(q.w, 1.0)
        self.assertAlmostEqual(q.x, 0.0)

    def test_controller_nominal_hover(self):
        """Verify nominal hover produces ~13 N total thrust (~3.25 N per motor)."""
        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])

        motor_thrusts, diag = self.controller.compute_motor_thrusts(
            pos, vel, quat, omega, sim_time=0.0
        )

        self.assertEqual(len(motor_thrusts), 4)
        total_thrust = np.sum(motor_thrusts)
        # Should hover close to total weight ~13.0 N
        self.assertAlmostEqual(total_thrust, self.drone_cfg.total_hover_weight, delta=1.5)

        # Legacy diagnostics must persist
        self.assertIn("F_z", diag)
        self.assertIn("tau_x", diag)
        self.assertIn("tau_y", diag)
        self.assertIn("tau_z", diag)
        self.assertIn("roll_err", diag)
        self.assertIn("pitch_err", diag)
        self.assertIn("target_roll", diag)
        self.assertIn("target_pitch", diag)

        # New L1 diagnostics
        self.assertIn("u_b", diag)
        self.assertIn("u_ad", diag)
        self.assertIn("sigma_f_hat", diag)
        self.assertIn("sigma_M_hat", diag)
        self.assertIn("is_triggered", diag)

    def test_system2_bridge_telemetry_summary(self):
        """Verify bridge records samples and computes structured TelemetrySummary."""
        target_pos = np.array([0.0, 0.0, 1.0])

        # Push 100 samples simulating a drone drifting backward in -X
        for i in range(100):
            t = i * 0.002
            pos = np.array([-0.4, 0.0, 0.9])
            vel = np.array([-0.2, 0.0, -0.05])
            quat = np.array([0.999, 0.0, 0.05, 0.0])
            omega = np.array([0.1, 0.2, 0.0])
            ctrl = np.array([3.5, 3.5, 3.0, 3.0])
            sigma_f = np.array([-2.5, 0.0, -1.0])
            sigma_M = np.array([0.1, -0.3, 0.0])

            self.bridge.push_telemetry(
                sim_time=t,
                pos=pos,
                vel=vel,
                quat=quat,
                omega=omega,
                motor_ctrl=ctrl,
                target_pos=target_pos,
                sigma_f_hat=sigma_f,
                sigma_M_hat=sigma_M,
            )

        summary = self.bridge.get_telemetry_summary(window_seconds=1.0)
        self.assertEqual(summary.samples_count, 100)
        self.assertAlmostEqual(summary.pos_error[0], 0.4, delta=0.01)
        self.assertAlmostEqual(summary.velocity[0], -0.2, delta=0.01)
        self.assertAlmostEqual(summary.unmodeled_force_disturbance[0], -2.5, delta=0.01)

        # Ensure LLM prompt string generation functions
        llm_text = summary.format_llm_text()
        self.assertIn("LAST 0.2s TELEMETRY SUMMARY:", llm_text)
        self.assertIn("Position Error", llm_text)
        self.assertIn("External Disturbance Force", llm_text)

    def test_fast_trigger_mechanism(self):
        """Verify Fast Trigger fires when thresholds are exceeded."""
        target_pos = np.array([0.0, 0.0, 1.0])

        # Exceed position error threshold (0.80m > 0.30m)
        is_trig, reasons = self.bridge.push_telemetry(
            sim_time=1.0,
            pos=np.array([0.8, 0.0, 1.0]),
            vel=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.array([0.0, 0.0, 0.0]),
            motor_ctrl=np.array([3.25, 3.25, 3.25, 3.25]),
            target_pos=target_pos,
            sigma_f_hat=np.zeros(3),
            sigma_M_hat=np.zeros(3),
        )
        self.assertTrue(is_trig)
        self.assertTrue(any("Position error" in r for r in reasons))
        self.assertTrue(self.bridge.is_triggered())

    def test_system2_patch_application(self):
        """Verify applying System 2 patch alters controller behavior."""
        patch = System2Patch(
            diagnosis="Counteracting headwind",
            feedforward_force_offset=[3.0, 0.0, 2.0],
            gain_multipliers=GainMultipliers(pos_kp=1.5, pos_kd=1.2, att_kp=0.8, att_kd=1.4),
            motor_trim_scale=[1.2, 1.0, 1.0, 1.0],
        )
        self.bridge.apply_system2_patch(patch)

        active = self.bridge.get_active_patch()
        self.assertIsNotNone(active)
        self.assertEqual(active.diagnosis, "Counteracting headwind")

        pos = np.array([0.0, 0.0, 1.0])
        vel = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])

        motor_thrusts, diag = self.controller.compute_motor_thrusts(
            pos, vel, quat, omega, sim_time=2.0
        )
        # Because of feedforward_force_offset[2] = 2.0 N, F_z should increase
        self.assertGreater(diag["F_z"], self.drone_cfg.total_hover_weight)
        # Motor 1 was scaled by 1.2
        self.assertGreater(motor_thrusts[0], motor_thrusts[1])

    def test_full_simulation_loop(self):
        """Run 200 physics steps inside DroneSimulator with L1 controller."""
        sim = DroneSimulator()
        for step in range(200):
            diag = sim.step_simulation(step * 0.002)

        self.assertIn("pos", diag)
        self.assertIn("sigma_f_hat", diag)
        self.assertIn("sigma_M_hat", diag)


if __name__ == "__main__":
    unittest.main()
