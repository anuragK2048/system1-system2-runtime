"""Native 3D HUD Geometric Overlay Engine for MuJoCo Passive Viewer.

Injects dynamic 3D visual primitives directly into MuJoCo's `viewer.user_scn` (mjvScene):
1. Concentric ground clearance rings (1m, 2m, 3m) projected on Z=0 floor beneath drone.
2. 3D aperture wireframe bounding boxes (Left Window, Right Arch).
3. Active target waypoint marker sphere and line-of-sight guidance vector.
4. Trajectory history breadcrumb ribbon (decaying flight path spheres).
5. Interactive toggle: pressing 'H' in the native viewer window toggles 3D HUD on/off.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Deque, List, Optional, Tuple

import mujoco
import numpy as np


class Native3DHUDOverlay:
    """Manages geometric HUD overlays rendered directly inside MuJoCo's 3D viewport."""

    def __init__(self, max_history: int = 80) -> None:
        self.enabled: bool = True
        self.history: Deque[np.ndarray] = collections.deque(maxlen=max_history)
        self._last_sample_t: float = -1.0
        self._sample_interval: float = 0.10  # 10 Hz trajectory breadcrumbs

        # 3D Landmark Apertures (Center pos, half-size [w/2, d/2, h/2], RGBA color)
        self.apertures: List[dict] = [
            {
                "name": "WINDOW_LEFT",
                "pos": np.array([0.0, 5.0, 1.10], dtype=np.float64),
                "size": np.array([0.70, 0.06, 0.60], dtype=np.float64),
                "rgba": np.array([0.0, 0.9, 1.0, 0.9], dtype=np.float32),  # Neon Cyan
            },
            {
                "name": "ARCH_RIGHT",
                "pos": np.array([0.0, -5.0, 0.90], dtype=np.float64),
                "size": np.array([0.75, 0.08, 0.85], dtype=np.float64),
                "rgba": np.array([1.0, 0.45, 0.05, 0.9], dtype=np.float32),  # High-vis Orange
            },
        ]

    def toggle(self) -> bool:
        """Toggles 3D HUD overlay state."""
        self.enabled = not self.enabled
        state_str = "ON" if self.enabled else "OFF"
        print(f"👁️ [NATIVE 3D HUD] Geometric overlays: {state_str} (Press 'H' to toggle)")
        return self.enabled

    def handle_key(self, keycode: int) -> None:
        """GLFW key callback for passive viewer (intercepts 'H' key, keycode 72)."""
        if keycode in (72, 104):  # 'H' or 'h'
            self.toggle()

    def update_trajectory(self, drone_pos: np.ndarray, current_time: float) -> None:
        """Samples drone position for rolling trajectory ribbon."""
        if self._last_sample_t < 0 or (current_time - self._last_sample_t) >= self._sample_interval:
            self.history.append(drone_pos.copy())
            self._last_sample_t = current_time

    def render(self, user_scn: mujoco.MjvScene, sim: Any, coordinator: Any) -> None:
        """Injects dynamic 3D geometric primitives into MuJoCo user_scn."""
        if not self.enabled or user_scn is None:
            if user_scn:
                user_scn.ngeom = 0
            return

        user_scn.ngeom = 0
        max_geoms = user_scn.maxgeom

        drone_pos = np.array(sim.data.qpos[0:3], dtype=np.float64)
        sim_time = float(sim.data.time)
        self.update_trajectory(drone_pos, sim_time)

        identity_mat = np.eye(3).flatten().astype(np.float64)

        # ----------------------------------------------------------------------
        # 1. Concentric Ground Rings (1m, 2m, 3m discs projected on Z=0.005)
        # ----------------------------------------------------------------------
        for r, alpha in [(1.0, 0.18), (2.0, 0.12), (3.0, 0.08)]:
            if user_scn.ngeom >= max_geoms:
                break
            ring_pos = np.array([drone_pos[0], drone_pos[1], 0.005], dtype=np.float64)
            ring_size = np.array([r, 0.002, 0.0], dtype=np.float64)
            ring_rgba = np.array([0.0, 0.85, 1.0, alpha], dtype=np.float32)
            mujoco.mjv_initGeom(
                user_scn.geoms[user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                ring_size,
                ring_pos,
                identity_mat,
                ring_rgba,
            )
            user_scn.ngeom += 1

        # ----------------------------------------------------------------------
        # 2. Target Aperture Wireframe Boxes (Window & Arch)
        # ----------------------------------------------------------------------
        for ap in self.apertures:
            if user_scn.ngeom >= max_geoms:
                break
            mujoco.mjv_initGeom(
                user_scn.geoms[user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINEBOX,
                ap["size"],
                ap["pos"],
                identity_mat,
                ap["rgba"],
            )
            user_scn.ngeom += 1

        # ----------------------------------------------------------------------
        # 3. Active Waypoint Sphere & Guidance Vector Line
        # ----------------------------------------------------------------------
        active_patch = sim.bridge.get_active_patch()
        waypoint_target = None
        if active_patch and active_patch.waypoint_command and active_patch.waypoint_command.target:
            waypoint_target = np.array(active_patch.waypoint_command.target[:3], dtype=np.float64)
        else:
            waypoint_target = getattr(sim.controller, "target_pos", None)

        if waypoint_target is not None and user_scn.ngeom < max_geoms:
            # Waypoint Sphere
            wpt_size = np.array([0.10, 0.0, 0.0], dtype=np.float64)
            wpt_rgba = np.array([0.0, 1.0, 0.55, 0.85], dtype=np.float32)  # Bright Mint
            mujoco.mjv_initGeom(
                user_scn.geoms[user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                wpt_size,
                waypoint_target,
                identity_mat,
                wpt_rgba,
            )
            user_scn.ngeom += 1

            # Guidance Vector (Line from drone center to target)
            if user_scn.ngeom < max_geoms:
                line_midpoint = (drone_pos + waypoint_target) / 2.0
                vec = waypoint_target - drone_pos
                dist = np.linalg.norm(vec)
                if dist > 0.05:
                    line_size = np.array([0.008, 0.0, dist / 2.0], dtype=np.float64)
                    line_rgba = np.array([0.0, 0.9, 1.0, 0.45], dtype=np.float32)
                    # Compute orientation matrix aligning Z axis to vector
                    z_axis = vec / dist
                    # Find arbitrary orthogonal vector for X
                    temp_up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                    x_axis = np.cross(temp_up, z_axis)
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    rot_mat = np.column_stack([x_axis, y_axis, z_axis]).flatten().astype(np.float64)

                    mujoco.mjv_initGeom(
                        user_scn.geoms[user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        line_size,
                        line_midpoint,
                        rot_mat,
                        line_rgba,
                    )
                    user_scn.ngeom += 1

        # ----------------------------------------------------------------------
        # 4. Trajectory Breadcrumb Ribbon (Flight History)
        # ----------------------------------------------------------------------
        num_hist = len(self.history)
        if num_hist > 1:
            step = max(1, num_hist // 25)
            for idx in range(0, num_hist, step):
                if user_scn.ngeom >= max_geoms:
                    break
                p = self.history[idx]
                frac = (idx + 1) / num_hist
                alpha = float(0.10 + 0.65 * (frac ** 2))
                sphere_size = np.array([0.025, 0.0, 0.0], dtype=np.float64)
                sphere_rgba = np.array([0.0, 0.85, 1.0, alpha], dtype=np.float32)
                mujoco.mjv_initGeom(
                    user_scn.geoms[user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    sphere_size,
                    p,
                    identity_mat,
                    sphere_rgba,
                )
                user_scn.ngeom += 1


# Global Singleton Instance
native_3d_hud = Native3DHUDOverlay()
