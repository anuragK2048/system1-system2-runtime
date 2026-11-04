"""Unit tests for Autonomous Waypoint Command schema, System 1 controller resolution, and FPV assets."""

import unittest
import numpy as np
import mujoco

from runtime.schemas.tier2 import System2Patch, WaypointCommand, validate_patch
from runtime.sim.system1_controller import System1Controller
from runtime.sim.bridge import System1System2Bridge
from runtime.server.routes_fpv import CAMERAS


class TestWaypointSchemas(unittest.TestCase):
    """Tests for WaypointCommand serialization, deserialization, and safety validation."""

    def test_waypoint_command_defaults(self):
        cmd = WaypointCommand()
        self.assertEqual(cmd.mode, "relative_body")
        self.assertEqual(cmd.target, [0.0, 0.0, 0.0])
        self.assertEqual(cmd.yaw_deg, 0.0)
        self.assertEqual(cmd.max_velocity, 1.0)
        self.assertEqual(cmd.corridor_width, 0.6)

        d = cmd.to_dict()
        self.assertEqual(d["mode"], "relative_body")
        self.assertEqual(d["target"], [0.0, 0.0, 0.0])

    def test_system2_patch_with_waypoint(self):
        patch_dict = {
            "diagnosis": "Navigate to Alpha Target Pad",
            "waypoint_command": {
                "mode": "world",
                "target": [5.5, 0.0, 0.8],
                "yaw_deg": 15.0,
                "max_velocity": 1.5,
                "corridor_width": 0.8,
            },
            "confidence": 0.95,
        }
        patch = System2Patch.from_dict(patch_dict)
        self.assertIsNotNone(patch.waypoint_command)
        self.assertEqual(patch.waypoint_command.mode, "world")
        self.assertEqual(patch.waypoint_command.target, [5.5, 0.0, 0.8])
        self.assertAlmostEqual(patch.waypoint_command.max_velocity, 1.5)

        out_dict = patch.to_dict()
        self.assertIn("waypoint_command", out_dict)
        self.assertEqual(out_dict["waypoint_command"]["mode"], "world")
        self.assertEqual(out_dict["waypoint_command"]["target"], [5.5, 0.0, 0.8])

    def test_waypoint_validation_clamps(self):
        # Velocity and corridor width beyond limits should be clamped
        patch = System2Patch(
            diagnosis="Aggressive Sprint",
            waypoint_command=WaypointCommand(
                mode="relative_body",
                target=[2.0, 0.0, 0.0],
                max_velocity=10.0,  # exceeds 5.0 m/s
                corridor_width=0.01,  # below 0.1 m
            ),
        )
        validated = validate_patch(patch)
        self.assertEqual(validated.waypoint_command.max_velocity, 5.0)
        self.assertEqual(validated.waypoint_command.corridor_width, 0.1)

    def test_waypoint_validation_rejections(self):
        # Invalid mode
        with self.assertRaises(ValueError):
            validate_patch(
                System2Patch(
                    diagnosis="Invalid mode test",
                    waypoint_command=WaypointCommand(mode="orbit"),
                )
            )

        # Invalid target length
        with self.assertRaises(ValueError):
            validate_patch(
                System2Patch(
                    diagnosis="Short target test",
                    waypoint_command=WaypointCommand(target=[1.0, 2.0]),
                )
            )


class TestSystem1WaypointResolution(unittest.TestCase):
    """Tests for System1Controller resolving active waypoint commands."""

    def setUp(self):
        self.bridge = System1System2Bridge()
        self.controller = System1Controller(bridge=self.bridge, target_pos=np.array([0.0, 0.0, 0.5]))

    def test_world_mode_waypoint(self):
        patch = System2Patch(
            diagnosis="Move to Elevated Station Delta",
            waypoint_command=WaypointCommand(mode="world", target=[8.0, 0.0, 1.2]),
        )
        self.bridge.apply_system2_patch(patch)

        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.0,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=np.array([0.0, 0.0, 0.5]),
        )
        np.testing.assert_allclose(resolved_target, [8.0, 0.0, 1.2], atol=1e-4)

    def test_relative_body_mode_zero_yaw(self):
        # With zero yaw (facing +X): [2.0, 1.0, 0.2] in body frame -> [2.0, 1.0, 0.2] in world frame
        patch = System2Patch(
            diagnosis="Step forward-left relative",
            waypoint_command=WaypointCommand(mode="relative_body", target=[2.0, 1.0, 0.2]),
        )
        self.bridge.apply_system2_patch(patch)

        quat_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_pos = np.array([1.0, 1.0, 0.5], dtype=np.float32)

        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.0,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=base_pos,
            quat=quat_identity,
        )
        # Expected: base [1.0, 1.0, 0.5] + offset [2.0, 1.0, 0.2] = [3.0, 2.0, 0.7]
        np.testing.assert_allclose(resolved_target, [3.0, 2.0, 0.7], atol=1e-4)

    def test_relative_body_mode_90deg_yaw(self):
        # Quat for 90 deg rotation around +Z: cos(45)=0.7071, sin(45)=0.7071
        q_yaw_90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
        patch = System2Patch(
            diagnosis="Forward 2.0m relative",
            waypoint_command=WaypointCommand(mode="relative_body", target=[2.0, 0.0, 0.0]),
        )
        self.bridge.apply_system2_patch(patch)

        base_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.0,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=base_pos,
            quat=q_yaw_90,
        )
        # Facing +Y: forward 2.0m along body X maps to +2.0m along world Y
        np.testing.assert_allclose(resolved_target, [0.0, 2.0, 0.5], atol=1e-4)

    def test_yaw_setpoint_resolution(self):
        # World mode yaw setpoint
        patch = System2Patch(
            diagnosis="Orient left towards window",
            waypoint_command=WaypointCommand(mode="world", target=[0.0, 0.0, 1.0], yaw_deg=90.0),
        )
        self.bridge.apply_system2_patch(patch)
        self.controller._resolve_active_patch(
            sim_time=0.0, explicit_ff=None, explicit_trim=None, target_pos=np.array([0.0, 0.0, 1.0])
        )
        self.assertAlmostEqual(np.degrees(self.controller.target_yaw_rad), 90.0, places=2)

    def test_yaw_slew_rate_limiting(self):
        # Set target yaw to +90 degrees (pi/2)
        self.controller.target_yaw_rad = float(np.pi / 2.0)
        self.controller.smoothed_yaw_rad = 0.0
        dt = 0.002
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega = np.zeros(3, dtype=np.float32)
        _, diag = self.controller.compute_motor_thrusts(pos, vel, quat, omega, sim_time=0.0, dt=dt)
        # Slew rate: 1.2 rad/s * 0.002s = 0.0024 rad
        self.assertLessEqual(self.controller.smoothed_yaw_rad, 1.2 * dt + 1e-5)
        self.assertIn("target_yaw_deg", diag)
        self.assertIn("smoothed_yaw_deg", diag)
        self.assertIn("current_yaw_deg", diag)


class TestFPVCameraAndAssets(unittest.TestCase):
    """Tests that MuJoCo assets, FPV camera, and CAMERAS route mappings are configured."""

    def test_scene_loading_and_camera_fpv(self):
        m = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        cam_names = [m.camera(i).name for i in range(m.ncam)]
        self.assertIn("fpv", cam_names)
        self.assertIn("track", cam_names)

        # Check fpv camera downward pitch (~8.5 deg)
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
        self.assertGreaterEqual(cam_id, 0)

        # Verify key target geoms exist in scene
        geom_names = [m.geom(i).name for i in range(m.ngeom)]
        self.assertIn("pad_alpha_base", geom_names)
        self.assertIn("pad_bravo_base", geom_names)
        self.assertIn("pad_charlie_base", geom_names)
        self.assertIn("pedestal_delta", geom_names)
        self.assertIn("pylon_left", geom_names)
        self.assertIn("pylon_right", geom_names)

    def test_track_camera_rotates_with_drone_alignment(self):
        """Verifies that the TPP track camera is fixed to the drone body and rotates with heading."""
        m = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        d = mujoco.MjData(m)
        track_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        self.assertGreaterEqual(track_id, 0)
        # Verify mode is fixed (mjCAMLIGHT_FIXED == 0), so it rotates with body rather than staying static in world
        self.assertEqual(m.cam_mode[track_id], mujoco.mjtCamLight.mjCAMLIGHT_FIXED)

        # Baseline: drone at (0, 0, 1) facing +X (yaw 0)
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        initial_pos = np.copy(d.cam_xpos[track_id])
        initial_mat = np.copy(d.cam_xmat[track_id].reshape(3, 3))
        # Initial position should be along -X (behind drone)
        self.assertLess(initial_pos[0], 0.0)

        # Rotate drone 90 deg yaw to face +Y
        d.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        mujoco.mj_forward(m, d)
        yaw90_pos = np.copy(d.cam_xpos[track_id])
        yaw90_mat = np.copy(d.cam_xmat[track_id].reshape(3, 3))

        # In TPP, when drone faces +Y, camera must be behind drone along -Y
        self.assertLess(yaw90_pos[1], -0.5)
        # Orientation matrix must rotate 90 degrees around Z
        expected_mat = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]) @ initial_mat
        np.testing.assert_allclose(yaw90_mat, expected_mat, atol=1e-3)

    def test_tracking_camera_heading_alignment(self):
        """Verifies dynamic azimuth calculation aligns chase camera behind quadrotor across headings."""
        test_angles_deg = [0.0, 45.0, 90.0, 180.0, -90.0]
        for deg in test_angles_deg:
            rad = np.radians(deg)
            q = [np.cos(rad / 2.0), 0.0, 0.0, np.sin(rad / 2.0)]
            computed_yaw = float(np.degrees(
                np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
            ))
            self.assertAlmostEqual(computed_yaw, deg, places=3)

    def test_routes_fpv_camera_dict(self):
        # All required identifiers must be in CAMERAS dictionary
        self.assertIn("drone1", CAMERAS)
        self.assertIn("drone2", CAMERAS)
        self.assertIn("drone", CAMERAS)
        self.assertIn("single", CAMERAS)
        self.assertIn("fpv", CAMERAS)
        self.assertEqual(CAMERAS["fpv"], "fpv")
        self.assertEqual(CAMERAS["single"], "fpv")
        self.assertEqual(CAMERAS["drone"], "fpv")


if __name__ == "__main__":
    unittest.main()
