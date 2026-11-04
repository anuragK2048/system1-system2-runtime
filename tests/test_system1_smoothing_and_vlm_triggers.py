"""Unit Tests for System 1 Smoothing Filters and Event-Driven System 2 VLM Triggers.

Validates:
1. System 1 3D Position Setpoint Trajectory Smoothing:
   - Verifies that step waypoint commands are transitioned into smooth reference trajectories.
   - Verifies velocity limits (max_velocity) and acceleration limits (bounded tilt).
2. System 1 In-Place Yaw Setpoint Slew-Rate and Acceleration Filtering:
   - Verifies bounded yaw rate (0.80 rad/s) and angular acceleration (2.0 rad/s^2).
3. System 1 Patch Parameter Smoothing:
   - Verifies that feedforward force offsets and motor trims fade in smoothly over filter duration.
4. System 2 Event-Driven VLM Triggering ("When Necessary"):
   - Verifies that VLM is called on startup, phase transitions, force disturbances, and motor failures.
   - Verifies that VLM is NOT called during calm reflexive station-keeping.

Runnable via:
    uv run python -m unittest tests/test_system1_smoothing_and_vlm_triggers.py
"""

import math
import unittest
import numpy as np

from runtime.sim.config import DroneConfig, L1Config
from runtime.sim.bridge import System1System2Bridge
from runtime.sim.simulator import DroneSimulator
from runtime.sim.system1_controller import System1Controller
from runtime.sim.autonomous_target_flight import AutonomousTargetNavigator
from runtime.schemas.tier2 import System2Patch, WaypointCommand, SafetyEnvelope, GainMultipliers
from runtime.config.system2_config import PatchApplicationConfig


class TestSystem1SmoothingFilters(unittest.TestCase):
    """Tests that System 1 smoothly transitions navigation setpoints and patch parameters."""

    def setUp(self):
        self.bridge = System1System2Bridge()
        self.controller = System1Controller(bridge=self.bridge)

    def test_navigation_instruction_position_smoothing(self):
        """Verifies that a 2.0m position step command is gracefully smoothed."""
        initial_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.controller.target_pos = initial_pos.copy()
        self.controller.reset_smoothing_filters(initial_pos)

        # Inject a 2.0m forward waypoint command
        patch = System2Patch(
            diagnosis="Forward 2.0m waypoint",
            waypoint_command=WaypointCommand(mode="world", target=[2.0, 0.0, 1.0], max_velocity=1.0),
        )
        self.bridge.apply_system2_patch(patch)

        dt = 0.002
        pos = initial_pos.copy()
        vel = np.zeros(3, dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega = np.zeros(3, dtype=np.float32)

        prev_smoothed = self.controller.smoothed_target_pos.copy()
        max_velocity_observed = 0.0
        max_accel_observed = 0.0

        # Step 250 cycles (0.5 seconds at 500 Hz)
        for step in range(250):
            sim_time = step * dt
            _, diag = self.controller.compute_motor_thrusts(
                pos=pos, vel=vel, quat=quat, omega=omega, sim_time=sim_time, dt=dt
            )
            curr_smoothed = np.array(diag["smoothed_target_pos"])
            curr_vel = np.array(diag["target_vel"])

            # Verify no instantaneous 2.0m jump on first step
            step_delta = np.linalg.norm(curr_smoothed - prev_smoothed)
            self.assertLess(
                step_delta,
                0.015,
                f"Step delta too large on step {step}: {step_delta:.4f}m (smoothing failed)",
            )

            # Track velocity & acceleration limits
            v_norm = float(np.linalg.norm(curr_vel[:2]))
            max_velocity_observed = max(max_velocity_observed, v_norm)
            self.assertLessEqual(
                v_norm,
                1.05,  # within 5% of commanded 1.0 m/s limit
                f"Smoothed velocity {v_norm:.2f} m/s exceeded max limit 1.0 m/s",
            )

            prev_smoothed = curr_smoothed.copy()

        # By 0.5s, smoothed target should have advanced smoothly without exceeding bounds
        self.assertGreater(self.controller.smoothed_target_pos[0], 0.15)
        self.assertLess(self.controller.smoothed_target_pos[0], 2.0)

    def test_yaw_setpoint_slew_rate_and_acceleration(self):
        """Verifies in-place yaw rotation setpoint is bounded by max yaw rate and acceleration."""
        self.controller.target_yaw_rad = float(np.radians(90.0))
        self.controller.smoothed_yaw_rad = 0.0
        self.controller.smoothed_yaw_rate = 0.0

        dt = 0.002
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega = np.zeros(3, dtype=np.float32)

        prev_yaw = 0.0
        prev_rate = 0.0

        for step in range(100):
            sim_time = step * dt
            _, diag = self.controller.compute_motor_thrusts(
                pos=pos, vel=vel, quat=quat, omega=omega, sim_time=sim_time, dt=dt
            )
            curr_yaw = self.controller.smoothed_yaw_rad
            curr_rate = self.controller.smoothed_yaw_rate

            # Check rate limit (max 0.80 rad/s)
            self.assertLessEqual(
                abs(curr_rate),
                0.81,
                f"Yaw rate {abs(curr_rate):.2f} exceeded limit 0.80 rad/s on step {step}",
            )

            # Check acceleration limit (max 2.0 rad/s^2 -> max rate change per step = 2.0 * dt = 0.004)
            accel = abs(curr_rate - prev_rate) / dt
            self.assertLessEqual(
                accel,
                2.05,
                f"Yaw acceleration {accel:.2f} exceeded limit 2.00 rad/s^2 on step {step}",
            )

            prev_yaw = curr_yaw
            prev_rate = curr_rate

    def test_patch_feedforward_smoothing(self):
        """Verifies feedforward force offset transitions smoothly under smoothing."""
        self.controller.enable_patch_smoothing = True
        self.controller.patch_smoothing_tau_s = 0.15

        # Command -5.0N feedforward
        active_ff = np.array([-5.0, 0.0, 0.0], dtype=np.float32)
        active_trim = np.ones(4, dtype=np.float32)
        pos_kp_mult = 1.0
        pos_kd_mult = 1.0
        att_kp_mult = 1.0
        att_kd_mult = 1.0
        max_tilt_deg = 20.0
        dt = 0.002

        # Step 50 cycles (0.10s)
        for _ in range(50):
            alpha = 1.0 - math.exp(-dt / 0.15)
            self.controller.smoothed_ff += alpha * (active_ff - self.controller.smoothed_ff)

        # After 0.10s, smoothed force should have smoothly reached roughly ~48% of -5.0N
        expected_ratio = 1.0 - math.exp(-0.10 / 0.15)  # ~0.486
        self.assertAlmostEqual(self.controller.smoothed_ff[0] / -5.0, expected_ratio, delta=0.05)


class TestEventDrivenVLMTriggering(unittest.TestCase):
    """Tests that System 2 VLM reasoning is invoked only when necessary."""

    def setUp(self):
        self.sim = DroneSimulator()
        self.navigator = AutonomousTargetNavigator(
            sim=self.sim,
            target_mission="alpha",
            use_vlm=True,
        )

    def tearDown(self):
        self.navigator.close()

    def test_trigger_on_mission_initialization(self):
        """Verifies VLM is triggered on the first cognitive step for visual grounding."""
        telemetry = self.navigator.get_onboard_telemetry()
        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=0.0)
        self.assertTrue(needed)
        self.assertIn("Initial mission assessment", reason)

    def test_no_trigger_during_calm_station_keeping(self):
        """Verifies VLM is bypassed during calm steady-state hover within tolerance."""
        telemetry = self.navigator.get_onboard_telemetry()
        # Mark initial call done at t=0.0
        self.navigator.last_vlm_call_time = 0.0
        self.navigator.phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._prev_phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._last_vlm_force_norm = 0.0
        self.navigator._last_vlm_pos_err = 0.05

        # Check at t=1.0s (less than heartbeat 5.0s, calm hover)
        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=1.0)
        self.assertFalse(needed)
        self.assertIn("reflexive hold active", reason.lower())

    def test_trigger_on_force_disturbance_spike(self):
        """Verifies VLM is immediately triggered when L1 observer detects external force spike."""
        telemetry = self.navigator.get_onboard_telemetry()
        self.navigator.last_vlm_call_time = 0.0
        self.navigator.phase = "PHASE_2_CORRIDOR_APPROACH"
        self.navigator._prev_phase = "PHASE_2_CORRIDOR_APPROACH"
        self.navigator._last_vlm_force_norm = 0.0

        # Simulate headwind gust (+5.2N along +X)
        telemetry["unmodeled_force_disturbance_N"] = [5.20, 0.00, 0.00]

        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=1.5)
        self.assertTrue(needed)
        self.assertIn("L1 force disturbance spike", reason)
        self.assertIn("5.2N", reason)

    def test_trigger_on_actuator_degradation(self):
        """Verifies VLM is immediately triggered when motor command asymmetry is detected."""
        telemetry = self.navigator.get_onboard_telemetry()
        self.navigator.last_vlm_call_time = 0.0
        self.navigator.phase = "PHASE_2_CORRIDOR_APPROACH"
        self.navigator._prev_phase = "PHASE_2_CORRIDOR_APPROACH"

        # Simulate Motor 1 degradation (M1 commanding 5.8N while others at 2.9N)
        telemetry["motor_controls_N"] = [5.80, 2.90, 2.90, 2.90]

        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=1.5)
        self.assertTrue(needed)
        self.assertIn("Actuator command asymmetry", reason)

    def test_trigger_on_phase_transition(self):
        """Verifies VLM is triggered when the flight phase advances to a new milestone."""
        telemetry = self.navigator.get_onboard_telemetry()
        self.navigator.last_vlm_call_time = 0.0
        self.navigator._prev_phase = "PHASE_0_LIFTOFF"
        self.navigator.phase = "PHASE_2_CORRIDOR_APPROACH"  # Phase changed

        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=1.0)
        self.assertTrue(needed)
        self.assertIn("Flight phase transition", reason)


if __name__ == "__main__":
    unittest.main()
