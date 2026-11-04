"""HUD Projection Helper for Multimodal Drone Navigation & Spatial Reasoning.

Burns metric ground rings (1m, 2m, 3m), artificial horizon lines, pitch ladders,
and 3D passage aperture frames (doors, windows, arches, hoops) directly onto
offscreen FPV camera frames rendered from MuJoCo.

Provides high-contrast, perspective-accurate visual grounding to eliminate
depth ambiguity and scale errors for Vision-Language Models (System 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import mujoco
except ImportError:
    mujoco = None


# ==============================================================================
# 1. 3D APERTURE / GATE DEFINITION (EXTENSIBLE FOR DOORS, WINDOWS, ARCHES)
# ==============================================================================

@dataclass
class Aperture3D:
    """Represents a 3D geometric aperture (door, window, arch, gate) in world space.
    
    Attributes:
        name: Human-readable identifier (e.g. 'DOOR_01', 'WINDOW_B', 'ARCH_GATE').
        center: 3D coordinates [X, Y, Z] in world space (meters).
        width: Clear horizontal opening width (meters).
        height: Clear vertical opening height (meters).
        yaw: Normal heading orientation (radians or degrees via helper). Yaw=0 means
             the aperture faces the positive X axis (drone flying +X passes through).
        shape: Geometry profile: 'rectangle', 'door', 'window', 'arch', 'circle', 'hoop'.
        color: RGB tuple for outer frame outline (default bright Amber/Gold).
        inner_color: RGB tuple for safe clearance envelope (default Neon Mint).
        safety_margin: Inset margin (meters) from frame edges for rotor clearance.
        show_corridor: Whether to project a 3D approach vector leading into the gate.
        approach_distance: Length (meters) of the approach vector in front of gate.
        label: Custom HUD label. If None, auto-generated with distance and dimensions.
    """
    name: str = "GATE"
    center: Tuple[float, float, float] = (2.0, 1.2, 1.0)
    width: float = 1.0
    height: float = 1.2
    yaw: float = 0.0  # in radians
    shape: str = "arch"  # 'arch', 'door', 'window', 'rectangle', 'circle', 'hoop'
    color: Tuple[int, int, int] = (255, 190, 0)       # Amber Gold
    inner_color: Tuple[int, int, int] = (0, 255, 130)  # Neon Mint
    safety_margin: float = 0.15                         # 15 cm rotor margin
    show_corridor: bool = True
    approach_distance: float = 1.5                      # 1.5m lead-in line
    label: Optional[str] = None

    def get_rotation_matrix(self) -> np.ndarray:
        """Computes 3x3 rotation matrix for the aperture yaw."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def get_normal_vector(self) -> np.ndarray:
        """Returns unit vector normal to the aperture plane (pointing towards approach)."""
        # Yaw 0 faces -X towards approaching drone
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0], dtype=np.float64)

    def get_outer_vertices(self, num_curved_pts: int = 16) -> List[np.ndarray]:
        """Computes 3D world coordinates for the outer frame perimeter."""
        return self._generate_vertices(self.width, self.height, num_curved_pts)

    def get_inner_vertices(self, num_curved_pts: int = 16) -> Optional[List[np.ndarray]]:
        """Computes 3D world coordinates for the safe inner clearance boundary."""
        if self.safety_margin <= 0.0:
            return None
        w_in = self.width - 2.0 * self.safety_margin
        h_in = self.height - 2.0 * self.safety_margin
        if w_in <= 0.1 or h_in <= 0.1:
            return None
        return self._generate_vertices(w_in, h_in, num_curved_pts)

    def _generate_vertices(self, w: float, h: float, num_pts: int) -> List[np.ndarray]:
        """Generates 3D world vertices for a given width and height based on shape."""
        R = self.get_rotation_matrix()
        c = np.array(self.center, dtype=np.float64)
        hw = w / 2.0
        hh = h / 2.0
        shape_norm = self.shape.lower().strip()

        local_pts: List[np.ndarray] = []

        if shape_norm in ("circle", "hoop"):
            # Circle centered at (0, 0, 0)
            radius = min(hw, hh)
            for th in np.linspace(0, 2 * math.pi, num_pts * 2, endpoint=True):
                local_pts.append(np.array([0.0, radius * math.cos(th), radius * math.sin(th)]))

        elif shape_norm == "arch":
            # Arch: rectangular vertical jambs with a semicircular arch top
            arch_r = hw
            base_h = max(0.0, h - arch_r)
            z_bottom = -hh
            z_spring = z_bottom + base_h

            # Bottom Left -> Spring Left
            local_pts.append(np.array([0.0, -hw, z_bottom]))
            local_pts.append(np.array([0.0, -hw, z_spring]))

            # Semicircular arch curve from Left (-hw) over top to Right (+hw)
            for th in np.linspace(math.pi, 0.0, num_pts, endpoint=True):
                local_pts.append(np.array([
                    0.0,
                    arch_r * math.cos(th),
                    z_spring + arch_r * math.sin(th)
                ]))

            # Spring Right -> Bottom Right -> Bottom Left (closed loop)
            local_pts.append(np.array([0.0, hw, z_spring]))
            local_pts.append(np.array([0.0, hw, z_bottom]))
            local_pts.append(np.array([0.0, -hw, z_bottom]))

        elif shape_norm in ("door", "doorway"):
            # Standard doorway: floor sill at bottom
            local_pts = [
                np.array([0.0, -hw, -hh]),
                np.array([0.0, -hw,  hh]),
                np.array([0.0,  hw,  hh]),
                np.array([0.0,  hw, -hh]),
                np.array([0.0, -hw, -hh]),
            ]

        else:
            # Standard rectangle / window
            local_pts = [
                np.array([0.0, -hw, -hh]),
                np.array([0.0, -hw,  hh]),
                np.array([0.0,  hw,  hh]),
                np.array([0.0,  hw, -hh]),
                np.array([0.0, -hw, -hh]),
            ]

        # Transform to world coordinates: P_w = center + R @ P_local
        return [c + (R @ p) for p in local_pts]

    def get_approach_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (start_point, end_point) of the 3D approach glidepath."""
        c = np.array(self.center, dtype=np.float64)
        norm = self.get_normal_vector()
        # Approach line starts in front of the gate along the normal
        p_front = c - norm * self.approach_distance
        return p_front, c


# ==============================================================================
# 2. CAMERA INTRINSICS & EXTRINSICS
# ==============================================================================

@dataclass
class CameraExtrinsics:
    """3D Camera pose and optical frame orientation in world coordinates."""
    pos: np.ndarray        # [x, y, z] in world coordinates
    forward: np.ndarray    # unit vector along optical line of sight (+Z_c in front)
    right: np.ndarray      # unit vector along camera horizontal right (+X_c)
    up: np.ndarray         # unit vector along camera vertical up (+Y_c)
    roll_deg: float = 0.0  # Roll angle in degrees (positive = banked right)
    pitch_deg: float = 0.0 # Pitch angle in degrees (positive = nose up)
    yaw_deg: float = 0.0   # Yaw angle in degrees (positive = turn left)


@dataclass
class CameraIntrinsics:
    """Pinhole camera projection parameters."""
    width: int = 480
    height: int = 360
    fovy: float = 90.0     # vertical FOV in degrees
    fx: float = field(init=False)
    fy: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self):
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        # fy from vertical FOV
        fovy_rad = math.radians(self.fovy)
        self.fy = (self.height / 2.0) / math.tan(fovy_rad / 2.0)
        # MuJoCo square pixel convention: fx = fy
        self.fx = self.fy


# ==============================================================================
# 3. HIGH-PERFORMANCE HUD PROJECTOR
# ==============================================================================

class HUDProjector:
    """Transforms 3D MuJoCo world coordinates into calibrated HUD overlays.
    
    Burns:
      - Concentric Ground Metric Rings (1m, 2m, 3m, etc.) with distance labels
      - Artificial Horizon Line with aircraft boresight symbol & pitch ladder
      - 3D Aperture Frames (doors, windows, arches) with clearance envelopes
      - Approach Glidepaths & Bullseye Centroids
      - Real-time Flight Telemetry Ribbon (altitude, speed, pitch, roll, heading)
    """

    def __init__(
        self,
        camera_name: str = "d2_fpv",
        width: int = 480,
        height: int = 360,
        fovy: float = 90.0,
        near_clip: float = 0.05,
    ):
        self.camera_name = camera_name
        self.intrinsics = CameraIntrinsics(width=width, height=height, fovy=fovy)
        self.near_clip = near_clip

    # --------------------------------------------------------------------------
    # Pose Extraction
    # --------------------------------------------------------------------------

    def get_camera_extrinsics(
        self,
        model: Optional["mujoco.MjModel"],
        data: Optional["mujoco.MjData"],
        override_pos: Optional[np.ndarray] = None,
        override_forward: Optional[np.ndarray] = None,
        override_up: Optional[np.ndarray] = None,
    ) -> CameraExtrinsics:
        """Extracts camera extrinsics from MuJoCo data or explicit overrides."""
        if override_pos is not None and override_forward is not None and override_up is not None:
            fwd = override_forward / np.linalg.norm(override_forward)
            up = override_up / np.linalg.norm(override_up)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            rpy = self._compute_rpy(fwd, up, right)
            return CameraExtrinsics(
                pos=override_pos.astype(np.float64),
                forward=fwd,
                right=right,
                up=up,
                roll_deg=rpy[0],
                pitch_deg=rpy[1],
                yaw_deg=rpy[2],
            )

        if model is None or data is None:
            # Fallback to standard nominal hover pose
            fwd = np.array([1.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])
            right = np.array([0.0, -1.0, 0.0])
            return CameraExtrinsics(
                pos=np.array([0.16, 1.2, 1.06]),
                forward=fwd,
                right=right,
                up=up,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
            )

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in MuJoCo model.")

        pos = data.cam_xpos[cam_id].copy()
        mat = data.cam_xmat[cam_id].reshape(3, 3)

        # MuJoCo camera coordinate convention:
        # mat[:, 0] is Camera Right
        # mat[:, 1] is Camera Up
        # -mat[:, 2] is Camera Forward
        right = mat[:, 0].copy()
        up = mat[:, 1].copy()
        forward = -mat[:, 2].copy()

        # Normalize to safeguard against floating precision drift
        forward /= np.linalg.norm(forward)
        up /= np.linalg.norm(up)
        right /= np.linalg.norm(right)

        rpy = self._compute_rpy(forward, up, right)

        return CameraExtrinsics(
            pos=pos,
            forward=forward,
            right=right,
            up=up,
            roll_deg=rpy[0],
            pitch_deg=rpy[1],
            yaw_deg=rpy[2],
        )

    @staticmethod
    def _compute_rpy(fwd: np.ndarray, up: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
        """Extracts aeronautical (roll, pitch, yaw) angles in degrees."""
        # Pitch: angle of forward vector above horizontal plane
        pitch = math.asin(np.clip(fwd[2], -1.0, 1.0))
        # Yaw: heading azimuth of forward vector
        yaw = math.atan2(fwd[1], fwd[0])
        # Roll: tilt of right wing (positive = right wing dips down)
        roll = math.atan2(-right[2], up[2])
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    # --------------------------------------------------------------------------
    # 3D -> 2D Pinhole Projection with Near-Plane Clipping
    # --------------------------------------------------------------------------

    def project_point(
        self,
        point_3d: np.ndarray,
        cam: CameraExtrinsics,
    ) -> Optional[Tuple[float, float, float]]:
        """Projects a 3D world point to 2D pixel coordinates (u, v, depth).
        
        Returns None if point is behind the camera near-plane.
        """
        dp = point_3d - cam.pos
        zc = float(np.dot(dp, cam.forward))
        if zc <= self.near_clip:
            return None

        xc = float(np.dot(dp, cam.right))
        yc = float(np.dot(dp, cam.up))

        u = self.intrinsics.cx + self.intrinsics.fx * (xc / zc)
        v = self.intrinsics.cy - self.intrinsics.fy * (yc / zc)
        return (u, v, zc)

    def project_polyline(
        self,
        points_3d: List[np.ndarray],
        cam: CameraExtrinsics,
        closed: bool = False,
    ) -> List[List[Tuple[float, float]]]:
        """Projects a 3D polyline with segment near-plane clipping.
        
        Returns a list of 2D screen line segments (each is a list of (u, v) points).
        Prevents perspective inversion when parts of lines extend behind the camera.
        """
        if len(points_3d) < 2:
            return []

        pts = list(points_3d)
        if closed and not np.allclose(pts[0], pts[-1]):
            pts.append(pts[0])

        visible_polylines: List[List[Tuple[float, float]]] = []
        current_segment: List[Tuple[float, float]] = []

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]

            dp1 = p1 - cam.pos
            dp2 = p2 - cam.pos
            z1 = float(np.dot(dp1, cam.forward))
            z2 = float(np.dot(dp2, cam.forward))

            # Case 1: Both behind near clip
            if z1 <= self.near_clip and z2 <= self.near_clip:
                if len(current_segment) >= 2:
                    visible_polylines.append(current_segment)
                current_segment = []
                continue

            # Case 2: Both in front
            if z1 > self.near_clip and z2 > self.near_clip:
                u1 = self.intrinsics.cx + self.intrinsics.fx * (float(np.dot(dp1, cam.right)) / z1)
                v1 = self.intrinsics.cy - self.intrinsics.fy * (float(np.dot(dp1, cam.up)) / z1)
                u2 = self.intrinsics.cx + self.intrinsics.fx * (float(np.dot(dp2, cam.right)) / z2)
                v2 = self.intrinsics.cy - self.intrinsics.fy * (float(np.dot(dp2, cam.up)) / z2)

                if not current_segment:
                    current_segment.append((u1, v1))
                current_segment.append((u2, v2))
                continue

            # Case 3: Segment crosses near plane -> clip
            t = (self.near_clip - z1) / (z2 - z1)
            p_clip = p1 + t * (p2 - p1)
            dp_clip = p_clip - cam.pos
            z_clip = self.near_clip
            u_clip = self.intrinsics.cx + self.intrinsics.fx * (float(np.dot(dp_clip, cam.right)) / z_clip)
            v_clip = self.intrinsics.cy - self.intrinsics.fy * (float(np.dot(dp_clip, cam.up)) / z_clip)

            if z1 > self.near_clip:
                # p1 in front, p2 behind: line ends at p_clip
                u1 = self.intrinsics.cx + self.intrinsics.fx * (float(np.dot(dp1, cam.right)) / z1)
                v1 = self.intrinsics.cy - self.intrinsics.fy * (float(np.dot(dp1, cam.up)) / z1)
                if not current_segment:
                    current_segment.append((u1, v1))
                current_segment.append((u_clip, v_clip))
                visible_polylines.append(current_segment)
                current_segment = []
            else:
                # p1 behind, p2 in front: line starts at p_clip
                if len(current_segment) >= 2:
                    visible_polylines.append(current_segment)
                current_segment = [(u_clip, v_clip)]
                u2 = self.intrinsics.cx + self.intrinsics.fx * (float(np.dot(dp2, cam.right)) / z2)
                v2 = self.intrinsics.cy - self.intrinsics.fy * (float(np.dot(dp2, cam.up)) / z2)
                current_segment.append((u2, v2))

        if len(current_segment) >= 2:
            visible_polylines.append(current_segment)

        return visible_polylines

    # --------------------------------------------------------------------------
    # HUD Layer: Ground Metric Rings
    # --------------------------------------------------------------------------

    def draw_ground_rings(
        self,
        draw: ImageDraw.Draw,
        cam: CameraExtrinsics,
        drone_pos: np.ndarray,
        radii: List[float] = (1.0, 2.0, 3.0, 4.0),
        color: Tuple[int, int, int] = (0, 230, 200),
        ground_z: float = 0.0,
        num_segments: int = 72,
        show_labels: bool = True,
        show_corridor_rails: bool = True,
        corridor_width: float = 1.2,
    ) -> None:
        """Projects concentric ground metric rings onto the floor plane."""
        # Ground nadir point directly below the drone
        cx_w = drone_pos[0]
        cy_w = drone_pos[1]

        # Draw rings
        for r in radii:
            thetas = np.linspace(0, 2 * math.pi, num_segments, endpoint=True)
            ring_pts = [
                np.array([cx_w + r * math.cos(th), cy_w + r * math.sin(th), ground_z])
                for th in thetas
            ]

            segments = self.project_polyline(ring_pts, cam, closed=True)
            for seg in segments:
                draw.line(seg, fill=color, width=2)

            # Draw metric distance label on the ring (offset 18° right to avoid flightpath occlusion)
            if show_labels:
                # Forward projected onto XY plane, offset by 18° azimuth
                fwd_xy = np.array([cam.forward[0], cam.forward[1], 0.0])
                norm_xy = np.linalg.norm(fwd_xy)
                if norm_xy > 1e-3:
                    fwd_xy /= norm_xy
                    # Rotate fwd_xy by -18° (towards screen right)
                    th_offset = math.radians(-18.0)
                    c_off, s_off = math.cos(th_offset), math.sin(th_offset)
                    fwd_rot = np.array([
                        fwd_xy[0] * c_off - fwd_xy[1] * s_off,
                        fwd_xy[0] * s_off + fwd_xy[1] * c_off,
                        0.0
                    ])
                    label_pt_3d = np.array([cx_w + r * fwd_rot[0], cy_w + r * fwd_rot[1], ground_z])
                    proj = self.project_point(label_pt_3d, cam)
                    if proj is not None:
                        lu, lv, _ = proj
                        # Check within visible image bounds with margin
                        if 10 <= lu <= self.intrinsics.width - 40 and 10 <= lv <= self.intrinsics.height - 20:
                            text = f"{r:.0f}m" if r == int(r) else f"{r:.1f}m"
                            draw.text(
                                (lu - 8, lv - 12),
                                text,
                                fill=color,
                                stroke_width=2,
                                stroke_fill=(0, 0, 0),
                            )

        # Optional forward corridor boundary rails
        if show_corridor_rails:
            half_w = corridor_width / 2.0
            rail_color = (0, 180, 255)
            for side in (-1.0, 1.0):
                rail_pts = [
                    np.array([cx_w, cy_w + side * half_w, ground_z]),
                    np.array([cx_w + 5.0, cy_w + side * half_w, ground_z]),
                ]
                segments = self.project_polyline(rail_pts, cam)
                for seg in segments:
                    draw.line(seg, fill=rail_color, width=1)

    # --------------------------------------------------------------------------
    # HUD Layer: Artificial Horizon, Pitch Ladder, and Boresight Reticle
    # --------------------------------------------------------------------------

    def draw_horizon(
        self,
        draw: ImageDraw.Draw,
        cam: CameraExtrinsics,
        color: Tuple[int, int, int] = (0, 220, 255),
        line_width: int = 2,
        show_pitch_ladder: bool = True,
        show_boresight: bool = True,
    ) -> None:
        """Projects true optical artificial horizon line and pitch ladder rungs."""
        w = self.intrinsics.width
        h = self.intrinsics.height
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy

        # Horizon plane normal in world is n_w = [0, 0, 1]
        nx = float(cam.right[2])
        ny = float(cam.up[2])
        nz = float(cam.forward[2])

        # Line equation in pixel coordinates: A*u + B*v + C = 0
        # where (nx/fx)*(u - cx) - (ny/fy)*(v - cy) + nz = 0
        # A = nx / fx
        # B = -ny / fy
        # C = nz - A*cx - B*cy
        A = nx / fx
        B = -ny / fy
        C = nz - A * cx - B * cy

        # Find intersections with the 4 screen boundaries
        intersections: List[Tuple[float, float]] = []

        if abs(B) > 1e-7:
            # u = 0
            v0 = (-C) / B
            if 0 <= v0 <= h:
                intersections.append((0.0, v0))
            # u = w
            vw = (-A * w - C) / B
            if 0 <= vw <= h:
                intersections.append((float(w), vw))

        if abs(A) > 1e-7:
            # v = 0
            u0 = (-C) / A
            if 0 <= u0 <= w and (u0, 0.0) not in intersections:
                intersections.append((u0, 0.0))
            # v = h
            uh = (-B * h - C) / A
            if 0 <= uh <= w and (uh, float(h)) not in intersections:
                intersections.append((uh, float(h)))

        if len(intersections) >= 2:
            p1, p2 = intersections[0], intersections[1]
            draw.line([p1, p2], fill=color, width=line_width)

        # Draw Pitch Ladder Rungs (+10, -10, +20, -20 deg)
        if show_pitch_ladder and abs(ny) > 0.1:
            ladder_color = (0, 200, 255)
            # Screen unit vector parallel to horizon
            # Horizon line has normal (A, B). Tangent is (-B, A) normalized.
            t_len = math.hypot(B, A)
            if t_len > 1e-6:
                tx, ty = -B / t_len, A / t_len
                # Ladder rungs at pitch offsets
                for pitch_deg in (10.0, -10.0, 20.0, -20.0):
                    p_rad = math.radians(pitch_deg)
                    # Vector with elevation p_rad in camera heading plane
                    Dw = np.array([
                        math.cos(p_rad) * cam.forward[0],
                        math.cos(p_rad) * cam.forward[1],
                        math.sin(p_rad)
                    ])
                    zc = float(np.dot(Dw, cam.forward))
                    if zc > 0.1:
                        yc_pt = float(np.dot(Dw, cam.up))
                        v_center = cy - fy * (yc_pt / zc)
                        if 20 <= v_center <= h - 20:
                            rung_half_len = 24.0
                            r1 = (cx - rung_half_len * tx, v_center - rung_half_len * ty)
                            r2 = (cx + rung_half_len * tx, v_center + rung_half_len * ty)
                            # Draw rung
                            draw.line([r1, r2], fill=ladder_color, width=1)
                            # Draw tick mark at ends
                            tick_len = 6.0
                            # Tick points towards horizon
                            sign = -1.0 if pitch_deg > 0 else 1.0
                            tick1 = (r1[0] + sign * tick_len * ty, r1[1] - sign * tick_len * tx)
                            tick2 = (r2[0] + sign * tick_len * ty, r2[1] - sign * tick_len * tx)
                            draw.line([r1, tick1], fill=ladder_color, width=1)
                            draw.line([r2, tick2], fill=ladder_color, width=1)
                            # Rung angle text
                            draw.text(
                                (r2[0] + 5, r2[1] - 5),
                                f"{int(abs(pitch_deg))}",
                                fill=ladder_color,
                                stroke_width=1,
                                stroke_fill=(0, 0, 0),
                            )

        # Draw Center Boresight Aircraft Reticle
        if show_boresight:
            reticle_color = (0, 255, 255)
            # Center circle
            r_circle = 4
            draw.ellipse(
                [(cx - r_circle, cy - r_circle), (cx + r_circle, cy + r_circle)],
                outline=reticle_color,
                width=1,
            )
            # Horizontal wings
            wing_len = 16
            gap = 8
            draw.line([(cx - gap - wing_len, cy), (cx - gap, cy)], fill=reticle_color, width=2)
            draw.line([(cx + gap, cy), (cx + gap + wing_len, cy)], fill=reticle_color, width=2)
            # Top vertical pip
            draw.line([(cx, cy - gap - 6), (cx, cy - gap)], fill=reticle_color, width=1)

    # --------------------------------------------------------------------------
    # HUD Layer: 3D Portals, Apertures, Doors, Windows, and Arches
    # --------------------------------------------------------------------------

    def draw_aperture(
        self,
        draw: ImageDraw.Draw,
        cam: CameraExtrinsics,
        aperture: Aperture3D,
        drone_pos: Optional[np.ndarray] = None,
    ) -> None:
        """Projects an aperture (door/window/arch/gate) with safe clearance envelope."""
        # 1. Outer Frame Outline
        outer_pts = aperture.get_outer_vertices()
        outer_segments = self.project_polyline(outer_pts, cam, closed=True)
        for seg in outer_segments:
            draw.line(seg, fill=aperture.color, width=2)

        # 2. Inner Safety Clearance Frame
        inner_pts = aperture.get_inner_vertices()
        if inner_pts is not None:
            inner_segments = self.project_polyline(inner_pts, cam, closed=True)
            for seg in inner_segments:
                draw.line(seg, fill=aperture.inner_color, width=1)

        # 3. Approach Glidepath Line
        if aperture.show_corridor:
            p_start, p_end = aperture.get_approach_vector()
            approach_segs = self.project_polyline([p_start, p_end], cam)
            for seg in approach_segs:
                draw.line(seg, fill=(255, 220, 80), width=1)

        # 4. Center Target Reticle & Badge
        center_world = np.array(aperture.center, dtype=np.float64)
        c_proj = self.project_point(center_world, cam)
        if c_proj is not None:
            cu, cv, depth = c_proj

            # Precision Bullseye / Diamond
            d_size = 5
            diamond = [
                (cu, cv - d_size),
                (cu + d_size, cv),
                (cu, cv + d_size),
                (cu - d_size, cv),
                (cu, cv - d_size),
            ]
            draw.line(diamond, fill=(0, 255, 255), width=2)

            # Compute Euclidean Distance from drone or camera
            dist = float(np.linalg.norm(center_world - (drone_pos if drone_pos is not None else cam.pos)))

            # Metadata Badge Text
            label_text = aperture.label
            if label_text is None:
                shape_tag = aperture.shape.upper()
                label_text = f"[{aperture.name}:{shape_tag} | d={dist:.1f}m | {aperture.width:.1f}x{aperture.height:.1f}m]"

            # Place badge cleanly above the top of the aperture outer frame
            all_vs = [pt[1] for seg in outer_segments for pt in seg]
            if all_vs:
                v_top = min(all_vs)
                badge_y = v_top - 16
                if badge_y < 12:  # If too close to upper screen edge, place below frame
                    badge_y = max(all_vs) + 6
            else:
                badge_y = cv - 30

            # Center badge horizontally relative to aperture center
            est_width = len(label_text) * 6
            badge_x = int(cu - est_width // 2)
            badge_x = max(10, min(self.intrinsics.width - est_width - 10, badge_x))

            draw.text(
                (badge_x, badge_y),
                label_text,
                fill=aperture.color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

    # --------------------------------------------------------------------------
    # HUD Layer: Flight Telemetry Banner
    # --------------------------------------------------------------------------

    def draw_telemetry_banner(
        self,
        draw: ImageDraw.Draw,
        cam: CameraExtrinsics,
        drone_pos: np.ndarray,
        drone_vel: Optional[np.ndarray] = None,
        mode_str: str = "SYSTEM-2 SPATIAL NAV",
    ) -> None:
        """Burns aeronautical flight telemetry readouts at top and bottom."""
        w = self.intrinsics.width
        h = self.intrinsics.height

        alt = drone_pos[2]
        speed = float(np.linalg.norm(drone_vel)) if drone_vel is not None else 0.0

        # Top Banner: Attitude & Altitude
        top_text = (
            f"ALT: {alt:.2f}m | SPD: {speed:.1f}m/s | "
            f"PITCH: {cam.pitch_deg:+.1f}° | ROLL: {cam.roll_deg:+.1f}° | HDG: {cam.yaw_deg % 360:03.0f}°"
        )
        draw.text((10, 8), top_text, fill=(0, 255, 200), stroke_width=2, stroke_fill=(0, 0, 0))

        # Bottom Banner: Mode & Camera
        bot_text = f"CAM: {self.camera_name} ({w}x{h} FOV:{self.intrinsics.fovy:.0f}°) | MODE: {mode_str}"
        draw.text((10, h - 20), bot_text, fill=(0, 210, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    # --------------------------------------------------------------------------
    # Complete Pipeline: Overlay on PIL Image
    # --------------------------------------------------------------------------

    def overlay_hud(
        self,
        image: Image.Image,
        cam: CameraExtrinsics,
        drone_pos: np.ndarray,
        drone_vel: Optional[np.ndarray] = None,
        apertures: Optional[List[Aperture3D]] = None,
        radii: List[float] = (1.0, 2.0, 3.0, 4.0),
        show_horizon: bool = True,
        show_ground_rings: bool = True,
        show_telemetry: bool = True,
        mode_str: str = "SYSTEM-2 SPATIAL NAV",
    ) -> Image.Image:
        """Overlays complete HUD onto a PIL Image."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # 1. Ground Metric Rings
        if show_ground_rings:
            self.draw_ground_rings(draw, cam, drone_pos, radii=radii)

        # 2. Artificial Horizon & Pitch Ladder
        if show_horizon:
            self.draw_horizon(draw, cam)

        # 3. 3D Apertures (Doors, Windows, Arches, Gates)
        if apertures:
            for ap in apertures:
                self.draw_aperture(draw, cam, ap, drone_pos=drone_pos)

        # 4. Flight Telemetry Readouts
        if show_telemetry:
            self.draw_telemetry_banner(draw, cam, drone_pos, drone_vel, mode_str=mode_str)

        return annotated

    # --------------------------------------------------------------------------
    # All-in-One Renderer & MuJoCo Auto-Discovery
    # --------------------------------------------------------------------------

    def render_and_overlay(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        renderer: Optional["mujoco.Renderer"] = None,
        apertures: Optional[List[Aperture3D]] = None,
        drone_pos: Optional[np.ndarray] = None,
        drone_vel: Optional[np.ndarray] = None,
        radii: List[float] = (1.0, 2.0, 3.0, 4.0),
        auto_discover_apertures: bool = True,
    ) -> Image.Image:
        """Captures offscreen MuJoCo camera frame and applies HUD projection in one step."""
        cam = self.get_camera_extrinsics(model, data)

        # Determine drone position from joint if not provided
        if drone_pos is None:
            # Check Drone 2 root joint address
            d2_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "d2_root")
            if d2_root >= 0:
                adr = model.jnt_qposadr[d2_root]
                drone_pos = data.qpos[adr:adr + 3].copy()
            else:
                drone_pos = cam.pos.copy()

        # Render offscreen frame
        close_renderer = False
        if renderer is None:
            renderer = mujoco.Renderer(model, height=self.intrinsics.height, width=self.intrinsics.width)
            close_renderer = True

        renderer.update_scene(data, camera=self.camera_name)
        rgb = renderer.render()
        if close_renderer:
            renderer.close()

        base_img = Image.fromarray(rgb)

        # Combine explicit apertures with any auto-discovered from model
        all_apertures: List[Aperture3D] = list(apertures) if apertures else []
        if auto_discover_apertures:
            discovered = self.discover_apertures_from_model(model, data)
            all_apertures.extend(discovered)

        return self.overlay_hud(
            base_img,
            cam=cam,
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            apertures=all_apertures,
            radii=radii,
        )

    @staticmethod
    def discover_apertures_from_model(
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
    ) -> List[Aperture3D]:
        """Auto-discovers doors, windows, arches, and gates from MuJoCo bodies/sites.
        
        Scans for body or site names starting with 'gate', 'door', 'window', 'arch'.
        Extracts position, size, and orientation automatically.
        """
        discovered: List[Aperture3D] = []

        # 1. Scan bodies
        for b_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
            if not name:
                continue
            lower_name = name.lower()
            if any(lower_name.startswith(p) for p in ("gate", "door", "window", "arch", "hoop")):
                pos = data.xpos[b_id].copy()
                mat = data.xmat[b_id].reshape(3, 3)
                # Compute yaw from body orientation matrix (local X direction)
                yaw = math.atan2(mat[1, 0], mat[0, 0])

                # Determine shape category
                if "arch" in lower_name:
                    shape = "arch"
                    w, h = 1.0, 1.2
                elif "door" in lower_name:
                    shape = "door"
                    w, h = 0.85, 2.0
                elif "window" in lower_name:
                    shape = "window"
                    w, h = 0.9, 0.9
                elif "hoop" in lower_name:
                    shape = "circle"
                    w, h = 1.0, 1.0
                else:
                    shape = "rectangle"
                    w, h = 1.0, 1.0

                discovered.append(
                    Aperture3D(
                        name=name.upper(),
                        center=(float(pos[0]), float(pos[1]), float(pos[2])),
                        width=w,
                        height=h,
                        yaw=yaw,
                        shape=shape,
                    )
                )

        return discovered
