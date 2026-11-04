"""Unit tests verifying Camera-Forward Oriented Flight (No Reverse Flight).

Verifies:
1. When target direction deviates from current heading (|Δψ| > 15°), the drone
   holds its current (X, Y) position in-place and rotates on the spot.
2. Only after yaw lock (|Δψ| <= 15°) does translation forward along camera boresight begin.
3. Transition from Platform Delta (X=8.0m) to Pad Alpha (X=5.5m) rotates in-place to 180°.
4. Transition from Window egress (Y=6.0m) back to Arena Hub rotates in-place to -90°.
"""

import math
import unittest
import numpy as np

from runtime.sim.config import SCENE_PATH, DroneConfig
from runtime.sim.simulator import DroneSimulator
from runtime.sim.autonomous_target_flight import AutonomousTargetNavigator


class TestCameraForwardOrientation(unittest.TestCase):
    """Test suite verifying camera-forward orientation and in-place turn-before-flight."""

    def setUp(self):
        self.drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
        self.sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=self.drone_cfg)
        self.nav = AutonomousTargetNavigator(sim=self.sim, target_mission="alpha", use_vlm=False)

    def test_in_place_turn_before_backward_translation(self):
        """Drone at X=8.0m heading 0° retargeted to X=5.5m must hold in-place and turn to 180°."""
        pos = np.array([8.0, 0.0, 1.0])
        yaw_deg = 0.0  # Facing forward (+X)
        dest_xyz = [5.5, 0.0, 1.0]

        cmd, diag, is_locked = self.nav._camera_forward_waypoint(
            current_pos=pos,
            current_yaw_deg=yaw_deg,
            dest_xyz=dest_xyz,
            nominal_yaw_deg=180.0,
            target_name="Pad Alpha",
        )

        # 1. Target is NOT locked because Δψ = 180° > 15°
        self.assertFalse(is_locked)
        # 2. Position MUST be held at current X=8.0 (NOT moving backwards to 5.5)
        self.assertAlmostEqual(cmd.target[0], 8.0, places=2)
        # 3. Commanded yaw must be directed at destination (180°)
        self.assertAlmostEqual(abs(cmd.yaw_deg), 180.0, places=1)
        # 4. Status reflects in-place rotation
        self.assertIn("Rotating in-place", diag)

    def test_advance_after_yaw_lock(self):
        """Once yaw aligns with destination (|Δψ| <= 15°), drone advances forward."""
        pos = np.array([8.0, 0.0, 1.0])
        yaw_deg = 175.0  # Aligned within 5° of 180°
        dest_xyz = [5.5, 0.0, 1.0]

        cmd, diag, is_locked = self.nav._camera_forward_waypoint(
            current_pos=pos,
            current_yaw_deg=yaw_deg,
            dest_xyz=dest_xyz,
            nominal_yaw_deg=180.0,
            target_name="Pad Alpha",
        )

        # 1. Target is locked
        self.assertTrue(is_locked)
        # 2. Position target advances forward to dest X=5.5
        self.assertAlmostEqual(cmd.target[0], 5.5, places=2)
        # 3. Status reflects advance
        self.assertIn("Advancing forward", diag)

    def test_window_egress_turn_around(self):
        """Drone outside Left Window (Y=6.0m) returning to Hub must turn from +90° to -90°."""
        pos = np.array([0.0, 6.0, 1.10])
        yaw_deg = 90.0  # Facing outward (+Y)
        dest_xyz = [0.0, 3.2, 1.10]  # Window entrance back to arena

        cmd, diag, is_locked = self.nav._camera_forward_waypoint(
            current_pos=pos,
            current_yaw_deg=yaw_deg,
            dest_xyz=dest_xyz,
            nominal_yaw_deg=-90.0,
            target_name="Window Aperture Ingress",
        )

        # Must hold Y=6.0 in-place and command -90°
        self.assertFalse(is_locked)
        self.assertAlmostEqual(cmd.target[1], 6.0, places=2)
        self.assertAlmostEqual(cmd.yaw_deg, -90.0, places=1)
        self.assertIn("Rotating in-place", diag)

    def test_arch_egress_turn_around(self):
        """Drone outside Right Arch (Y=-6.0m) returning to Hub must turn from -90° to +90°."""
        pos = np.array([0.0, -6.0, 0.90])
        yaw_deg = -90.0  # Facing outward (-Y)
        dest_xyz = [0.0, -3.2, 0.90]

        cmd, diag, is_locked = self.nav._camera_forward_waypoint(
            current_pos=pos,
            current_yaw_deg=yaw_deg,
            dest_xyz=dest_xyz,
            nominal_yaw_deg=90.0,
            target_name="Arch Aperture Ingress",
        )

        # Must hold Y=-6.0 in-place and command +90°
        self.assertFalse(is_locked)
        self.assertAlmostEqual(cmd.target[1], -6.0, places=2)
        self.assertAlmostEqual(cmd.yaw_deg, 90.0, places=1)
        self.assertIn("Rotating in-place", diag)


if __name__ == "__main__":
    unittest.main()
