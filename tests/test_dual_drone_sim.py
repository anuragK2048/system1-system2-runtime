"""Unit and Integration Tests for Dual-Drone Comparative Simulation in MuJoCo.

Tests:
1. Dual scene XML loading and entity resolution.
2. DualDroneSimulator initialization across all supported control modes.
3. Stable multi-step hover for both drones at their respective lateral targets (Y = -1.2m, Y = +1.2m).
4. Synchronized chaos perturbations applied to both drones.
5. Independent bridge telemetry and patch isolation between Drone 1 and Drone 2.
"""

import unittest
import numpy as np
import mujoco

from runtime.sim.config import DUAL_SCENE_PATH, DroneConfig, SimConfig
from runtime.sim.dual_simulator import DualDroneSimulator
from runtime.schemas.tier2 import System2Patch, GainMultipliers


class TestDualDroneSim(unittest.TestCase):
    """Test suite for dual-drone simulation and independent controller pipelines."""

    def test_dual_scene_xml_compilation(self):
        """Verify dual_scene.xml compiles cleanly with 18 qpos and 8 actuators."""
        self.assertTrue(DUAL_SCENE_PATH.exists(), f"Missing {DUAL_SCENE_PATH}")
        model = mujoco.MjModel.from_xml_path(str(DUAL_SCENE_PATH))
        data = mujoco.MjData(model)

        # 18 qpos: 7 (d1 root) + 2 (d1 swing) + 7 (d2 root) + 2 (d2 swing)
        self.assertEqual(model.nq, 18)
        # 16 qvel: 6 (d1 root) + 2 (d1 swing) + 6 (d2 root) + 2 (d2 swing)
        self.assertEqual(model.nv, 16)
        # 8 actuators: 4 for d1, 4 for d2
        self.assertEqual(model.nu, 8)

    def test_dual_simulator_modes_initialization(self):
        """Verify simulator accepts all valid mode combinations."""
        modes = ["pid", "l1_adaptive", "pid+system2", "l1_adaptive+system2"]
        for m1 in modes:
            for m2 in modes:
                sim = DualDroneSimulator(drone1_mode=m1, drone2_mode=m2)
                self.assertEqual(sim.drone1_mode, m1)
                self.assertEqual(sim.drone2_mode, m2)
                self.assertIsNotNone(sim.controller_1)
                self.assertIsNotNone(sim.controller_2)

    def test_invalid_mode_raises(self):
        """Verify unsupported mode strings raise ValueError."""
        with self.assertRaises(ValueError):
            DualDroneSimulator(drone1_mode="invalid_mode", drone2_mode="pid")
        with self.assertRaises(ValueError):
            DualDroneSimulator(drone1_mode="pid", drone2_mode="unsupported")

    def test_dual_hover_stability(self):
        """Verify both drones remain stable at Y=-1.2m and Y=+1.2m during hover."""
        # Drone 1: PID, Drone 2: L1 Adaptive
        sim = DualDroneSimulator(drone1_mode="pid", drone2_mode="l1_adaptive")

        # Step 200 times (0.4s at 500 Hz)
        for step in range(200):
            t = step * 0.002
            t1, t2 = sim.step_simulation(current_time=t)

        # Verify Drone 1 position is close to (0, -1.2, 1.0)
        pos1 = t1["pos"]
        self.assertAlmostEqual(pos1[0], 0.0, delta=0.20)
        self.assertAlmostEqual(pos1[1], -1.2, delta=0.20)
        self.assertAlmostEqual(pos1[2], 1.0, delta=0.25)

        # Verify Drone 2 position is close to (0, +1.2, 1.0)
        pos2 = t2["pos"]
        self.assertAlmostEqual(pos2[0], 0.0, delta=0.20)
        self.assertAlmostEqual(pos2[1], +1.2, delta=0.20)
        self.assertAlmostEqual(pos2[2], 1.0, delta=0.25)

        # Verify separation distance remains ~2.4m
        separation = np.linalg.norm(pos2 - pos1)
        self.assertAlmostEqual(separation, 2.4, delta=0.35)

    def test_synchronized_chaos_affects_both_drones(self):
        """Verify activating wind gust injects lateral force on both drone bodies."""
        sim = DualDroneSimulator(drone1_mode="pid", drone2_mode="l1_adaptive")

        # Enable wind gust chaos
        sim.chaos.active_chaos["wind_gust"] = True
        t1, t2 = sim.step_simulation(current_time=1.0)

        # Both drone bodies must have the 5.0 N lateral force applied in MuJoCo
        fx1 = sim.data.xfrc_applied[sim.d1_body_id, 0]
        fx2 = sim.data.xfrc_applied[sim.d2_body_id, 0]
        self.assertAlmostEqual(fx1, 5.0, delta=0.01)
        self.assertAlmostEqual(fx2, 5.0, delta=0.01)

    def test_independent_bridge_patches(self):
        """Verify control patches applied to Bridge 1 do not leak into Bridge 2."""
        sim = DualDroneSimulator(drone1_mode="pid+system2", drone2_mode="l1_adaptive+system2")

        patch_1 = System2Patch(
            diagnosis="Countering wind on Drone 1",
            feedforward_force_offset=[-5.0, 0.0, 0.0],
            gain_multipliers=GainMultipliers(pos_kp=1.2, pos_kd=1.0, att_kp=1.0, att_kd=1.0),
        )

        sim.bridge_1.apply_system2_patch(patch_1)

        # Bridge 1 has the patch, Bridge 2 must be None
        self.assertIsNotNone(sim.bridge_1.get_active_patch())
        self.assertEqual(sim.bridge_1.get_active_patch().feedforward_force_offset, [-5.0, 0.0, 0.0])
        self.assertIsNone(sim.bridge_2.get_active_patch())


if __name__ == "__main__":
    unittest.main()
