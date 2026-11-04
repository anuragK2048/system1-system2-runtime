"""Unit Tests for HUD Projection Helper (vlm/hud_overlay.py).

Verifies:
  - 3D Aperture vertex math for arch, door, window, and hoop shapes.
  - Perspective projection and near-plane clipping.
  - Artificial horizon calculation and RPY attitude extraction.
  - End-to-end PIL HUD burning.

Runnable via:
    uv run python -m unittest tests/test_hud_overlay.py
"""

import math
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from vlm.hud_overlay import Aperture3D, CameraExtrinsics, HUDProjector


class TestHUDOverlay(unittest.TestCase):
    def test_aperture_vertices_shapes(self):
        """Verifies 3D vertex generation for different aperture shapes."""
        # Arch
        arch = Aperture3D(
            name="TEST_ARCH",
            center=(2.0, 1.0, 1.0),
            width=1.0,
            height=1.2,
            yaw=0.0,
            shape="arch",
            safety_margin=0.15,
        )
        outer = arch.get_outer_vertices()
        inner = arch.get_inner_vertices()
        self.assertGreater(len(outer), 10)
        self.assertIsNotNone(inner)
        self.assertGreater(len(inner), 10)
        # Center is at [2.0, 1.0, 1.0]
        p_start, p_end = arch.get_approach_vector()
        np.testing.assert_allclose(p_end, [2.0, 1.0, 1.0])
        self.assertLess(p_start[0], p_end[0])  # In front of gate (X=0.5 < X=2.0)

        # Door
        door = Aperture3D(name="DOOR", center=(3.0, 0.0, 1.0), width=0.8, height=2.0, shape="door")
        self.assertEqual(len(door.get_outer_vertices()), 5)

        # Window
        win = Aperture3D(name="WIN", center=(3.0, 0.0, 1.0), width=0.9, height=0.9, shape="window")
        self.assertEqual(len(win.get_outer_vertices()), 5)

        # Hoop / Circle
        circle = Aperture3D(name="HOOP", center=(3.0, 0.0, 1.0), width=1.0, height=1.0, shape="circle")
        self.assertGreaterEqual(len(circle.get_outer_vertices()), 30)

    def test_camera_pinhole_projection(self):
        """Verifies 3D -> 2D pinhole projection matches analytical geometry."""
        projector = HUDProjector(width=480, height=360, fovy=90.0)

        cam = CameraExtrinsics(
            pos=np.array([0.0, 0.0, 1.0]),
            forward=np.array([1.0, 0.0, 0.0]),
            right=np.array([0.0, -1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )

        # Point directly centered at 2m in front: [2, 0, 1]
        pt_center = np.array([2.0, 0.0, 1.0])
        res = projector.project_point(pt_center, cam)
        self.assertIsNotNone(res)
        u, v, zc = res
        self.assertAlmostEqual(u, 240.0, places=3)
        self.assertAlmostEqual(v, 180.0, places=3)
        self.assertAlmostEqual(zc, 2.0, places=3)

        # Point to the right: [2, -1, 1] -> in camera right dir (since right is [0, -1, 0])
        pt_right = np.array([2.0, -1.0, 1.0])
        res_r = projector.project_point(pt_right, cam)
        self.assertIsNotNone(res_r)
        ur, vr, _ = res_r
        # u = 240 + fx * (1.0 / 2.0) = 240 + 180 * 0.5 = 330.0
        self.assertAlmostEqual(ur, 330.0, places=3)
        self.assertAlmostEqual(vr, 180.0, places=3)

        # Point behind camera: [-1, 0, 1]
        pt_behind = np.array([-1.0, 0.0, 1.0])
        self.assertIsNone(projector.project_point(pt_behind, cam))

    def test_near_plane_polyline_clipping(self):
        """Verifies that lines crossing the near plane are clipped cleanly."""
        projector = HUDProjector(width=480, height=360, fovy=90.0, near_clip=0.1)
        cam = CameraExtrinsics(
            pos=np.array([0.0, 0.0, 1.0]),
            forward=np.array([1.0, 0.0, 0.0]),
            right=np.array([0.0, -1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )

        # Line segment crossing the camera: from X=-1.0 (behind) to X=2.0 (in front)
        line_cross = [np.array([-1.0, 0.0, 1.0]), np.array([2.0, 0.0, 1.0])]
        segs = projector.project_polyline(line_cross, cam)
        self.assertEqual(len(segs), 1)
        # Check that clipped start has valid non-negative depth
        p_start = segs[0][0]
        p_end = segs[0][1]
        self.assertAlmostEqual(p_start[0], 240.0, places=2)
        self.assertAlmostEqual(p_end[0], 240.0, places=2)

        # Line completely behind: both X <= 0
        line_behind = [np.array([-2.0, 0.0, 1.0]), np.array([-0.5, 0.0, 1.0])]
        self.assertEqual(len(projector.project_polyline(line_behind, cam)), 0)

    def test_attitude_rpy_extraction(self):
        """Verifies Euler RPY calculation from camera axes."""
        # Neutral
        fwd = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        r, p, y = HUDProjector._compute_rpy(fwd, up, right)
        self.assertAlmostEqual(r, 0.0, places=4)
        self.assertAlmostEqual(p, 0.0, places=4)
        self.assertAlmostEqual(y, 0.0, places=4)

        # Banked Right 15 deg
        r_rad = math.radians(15.0)
        Rx = np.array([[1, 0, 0], [0, math.cos(r_rad), -math.sin(r_rad)], [0, math.sin(r_rad), math.cos(r_rad)]])
        fwd_bank = Rx @ fwd
        up_bank = Rx @ up
        right_bank = np.cross(fwd_bank, up_bank)
        r_deg, p_deg, y_deg = HUDProjector._compute_rpy(fwd_bank, up_bank, right_bank)
        self.assertAlmostEqual(r_deg, 15.0, places=4)

    def test_hud_overlay_execution(self):
        """Verifies complete HUD image burn pipeline."""
        img = Image.new("RGB", (480, 360), (30, 40, 50))
        projector = HUDProjector(width=480, height=360)

        cam = CameraExtrinsics(
            pos=np.array([0.16, 1.2, 1.06]),
            forward=np.array([1.0, 0.0, 0.0]),
            right=np.array([0.0, -1.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
        )

        arch = Aperture3D(name="ARCH_A", center=(2.5, 1.2, 1.0), width=1.0, height=1.2, shape="arch")
        annotated = projector.overlay_hud(
            img,
            cam=cam,
            drone_pos=np.array([0.0, 1.2, 1.0]),
            drone_vel=np.array([0.5, 0.0, 0.0]),
            apertures=[arch],
            radii=[1.0, 2.0, 3.0],
        )

        self.assertEqual(annotated.size, (480, 360))
        # Check that drawing actually modified pixels
        arr_orig = np.array(img)
        arr_ann = np.array(annotated)
        self.assertFalse(np.array_equal(arr_orig, arr_ann))
        diff_pixels = np.sum(arr_orig != arr_ann)
        self.assertGreater(diff_pixels, 500)  # Significant HUD markings burned


if __name__ == "__main__":
    unittest.main()
