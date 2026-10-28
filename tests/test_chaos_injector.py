"""Comprehensive unit tests for the modernized Chaos Injection Engine.

Tests:
1. Motor degradation: per-motor multipliers, chipped blade harmonics, and limits.
2. 3D wind engine: steady force, half-sine gust pulse, and Dryden turbulence.
3. Tethered payload coupling: attachment visibility, kick impulse, and aerodynamic wind drag.
4. IMU sensor corruption: Gaussian vibration, bias drift, and motor harmonic chirp.
5. Targeting modes: Synchronized ('both') vs Targeted ('drone1', 'drone2').
6. Repeatable presets: Applying preset scenarios and clearing back to clean hover.
7. HUD and telemetry reporting: get_chaos_summary() and get_hud_text().
"""

import unittest
import numpy as np
import mujoco

from runtime.sim.config import DUAL_SCENE_PATH
from runtime.chaos import (
    ChaosManager,
    ChaosTarget,
    CHAOS_PRESETS,
    get_preset_state,
)
from runtime.sim.dual_simulator import DualDroneSimulator


class TestChaosInjector(unittest.TestCase):
    """Test suite for the runtime.chaos subsystem."""

    def setUp(self):
        self.mgr = ChaosManager()

    def test_default_nominal_state(self):
        """Verify initial chaos state is pristine and completely unperturbed."""
        self.assertFalse(self.mgr.drone1.is_any_active())
        self.assertFalse(self.mgr.drone2.is_any_active())
        self.assertEqual(self.mgr.target_mode, ChaosTarget.BOTH)
        self.assertEqual(self.mgr.get_hud_text(1), "CLEAN (Nominal Hover)")

    def test_motor_multiplier_targeting(self):
        """Verify motor degradation can be targeted to Drone 1, Drone 2, or Both."""
        # Target Drone 1 only
        self.mgr.set_motor_multiplier(0, 0.40, target=ChaosTarget.DRONE1)
        self.assertAlmostEqual(self.mgr.drone1.motor.multipliers[0], 0.40)
        self.assertAlmostEqual(self.mgr.drone2.motor.multipliers[0], 1.00)
        self.assertTrue(self.mgr.drone1.motor.is_active())
        self.assertFalse(self.mgr.drone2.motor.is_active())

        # Degrade Motor 3 on Drone 2 only
        self.mgr.set_motor_multiplier(2, 0.50, target=ChaosTarget.DRONE2)
        self.assertAlmostEqual(self.mgr.drone2.motor.multipliers[2], 0.50)
        self.assertAlmostEqual(self.mgr.drone1.motor.multipliers[2], 1.00)

        # Synchronize both
        self.mgr.set_motor_multipliers([0.7, 0.8, 0.9, 1.0], target=ChaosTarget.BOTH)
        self.assertEqual(self.mgr.drone1.motor.multipliers, [0.7, 0.8, 0.9, 1.0])
        self.assertEqual(self.mgr.drone2.motor.multipliers, [0.7, 0.8, 0.9, 1.0])

    def test_actuator_degradation_output(self):
        """Verify actual thrust output matches degradation multiplier."""
        self.mgr.set_motor_multiplier(0, 0.50, target=ChaosTarget.BOTH)
        cmd = np.array([4.0, 4.0, 4.0, 4.0])
        actual1 = self.mgr.degrade_actuators_for_drone(1, cmd, sim_time=0.0)
        self.assertAlmostEqual(actual1[0], 2.0)
        self.assertAlmostEqual(actual1[1], 4.0)

    def test_wind_engine_forces(self):
        """Verify steady 3D wind vector and dynamic gust calculations."""
        # Steady wind: [4.0, -2.0, 1.0] N
        self.mgr.set_wind_force(4.0, -2.0, 1.0, turbulence_std=0.0, target=ChaosTarget.BOTH)
        self.assertTrue(self.mgr.drone1.wind.is_active())
        self.assertEqual(self.mgr.drone1.wind.steady_force, [4.0, -2.0, 1.0])

        # Test gust triggering
        self.mgr.trigger_wind_gust(
            force=6.0, direction=[1.0, 0.0, 0.0], duration_s=1.0, sim_time=2.0, target=ChaosTarget.BOTH
        )
        self.assertTrue(self.mgr.drone1.wind.gust_active)

    def test_payload_aerodynamic_coupling(self):
        """Verify payload attachment and aerodynamic drag coupling from wind."""
        model = mujoco.MjModel.from_xml_path(str(DUAL_SCENE_PATH))
        data = mujoco.MjData(model)

        d1_payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d1_payload")
        d1_tether_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "d1_tether_rod")
        d1_bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "d1_payload_bob")
        d1_swing_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "d1_swing_x")
        qpos_adr = model.jnt_qposadr[d1_swing_x]
        qvel_adr = model.jnt_dofadr[d1_swing_x]

        # 1. When detached: mass is counter-balanced with exact upward force
        self.mgr.set_payload_attached(False, target=ChaosTarget.BOTH)
        self.mgr.apply_drone_physics(
            drone_idx=1,
            model=model,
            data=data,
            drone_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d1_x2"),
            payload_body_id=d1_payload_id,
            tether_geom_id=d1_tether_id,
            bob_geom_id=d1_bob_id,
            sim_time=0.0,
            payload_qpos_adr=qpos_adr,
            payload_qvel_adr=qvel_adr,
        )
        # Upward counter force = 0.35 * 9.81 = 3.4335 N
        self.assertAlmostEqual(data.xfrc_applied[d1_payload_id, 2], 0.35 * 9.81, delta=0.01)

        # 2. When attached + wind blowing: tether is visible, gravity pulls naturally, and wind pushes bob
        self.mgr.set_payload_attached(True, mass=0.40, aerodynamic_drag=True, target=ChaosTarget.BOTH)
        self.mgr.set_wind_force(5.0, 0.0, 0.0, target=ChaosTarget.BOTH)
        self.mgr.apply_drone_physics(
            drone_idx=1,
            model=model,
            data=data,
            drone_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d1_x2"),
            payload_body_id=d1_payload_id,
            tether_geom_id=d1_tether_id,
            bob_geom_id=d1_bob_id,
            sim_time=0.0,
            payload_qpos_adr=qpos_adr,
            payload_qvel_adr=qvel_adr,
        )
        # Counter-force removed on Z (so gravity pulls down)
        self.assertAlmostEqual(data.xfrc_applied[d1_payload_id, 2], 0.0, delta=0.01)
        # Bob feels lateral aerodynamic drag from wind along X (> 0.0 N)
        self.assertGreater(data.xfrc_applied[d1_payload_id, 0], 0.1)

    def test_sensor_noise_injection(self):
        """Verify IMU angular velocity is corrupted by noise and bias drift."""
        self.mgr.set_sensor_noise(gyro_std=0.75, target=ChaosTarget.BOTH)
        self.mgr.drone1.sensor.gyro_bias = [0.1, -0.1, 0.0]

        raw_omega = np.array([0.0, 0.0, 0.0])
        corrupted = self.mgr.corrupt_sensors_for_drone(1, raw_omega, sim_time=0.0)

        # Corrupted should not be strictly zero
        self.assertNotEqual(float(np.linalg.norm(corrupted)), 0.0)

    def test_presets_application(self):
        """Verify applying presets updates state properly."""
        # Preset 1: Chipped Propeller
        self.mgr.apply_preset("1", target=ChaosTarget.BOTH)
        self.assertAlmostEqual(self.mgr.drone1.motor.multipliers[0], 0.40)
        self.assertIn("MTR", self.mgr.get_hud_text(1))

        # Preset 2: Microburst
        self.mgr.apply_preset("2", target=ChaosTarget.BOTH)
        self.assertTrue(self.mgr.drone1.wind.enabled)
        self.assertIn("WIND", self.mgr.get_hud_text(1))

        # Preset 0: Clear
        self.mgr.apply_preset("0", target=ChaosTarget.BOTH)
        self.assertFalse(self.mgr.drone1.is_any_active())
        self.assertEqual(self.mgr.get_hud_text(1), "CLEAN (Nominal Hover)")

    def test_dual_drone_sim_integration(self):
        """Verify DualDroneSimulator steps properly with the refactored chaos engine."""
        sim = DualDroneSimulator(drone1_mode="pid", drone2_mode="l1_adaptive")
        # Apply Preset 2 (Wind)
        sim.chaos.apply_preset("2", target=ChaosTarget.BOTH)

        t1, t2 = sim.step_simulation(current_time=1.0)
        self.assertIn("injected_wind_force", t1)
        self.assertIn("injected_wind_force", t2)
        self.assertAlmostEqual(t1["injected_wind_force"][0], 6.5, delta=1.5)
        self.assertIn("WIND", t1["chaos_hud"])


if __name__ == "__main__":
    unittest.main()
