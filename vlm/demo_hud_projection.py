"""Demonstration & Verification Script for HUD Projection Helper.

Generates multiple high-fidelity FPV frames illustrating:
  1. Neutral Hover: Ground rings (1m, 2m, 3m, 4m), horizon line, boresight reticle,
     and an Arch passage aperture directly on the flight corridor.
  2. Pitched-Down Forward Approach: Approaching an elevated Window aperture at 15° pitch down.
  3. Banked Roll Manoeuvre: 20° right roll, demonstrating tilted horizon, pitch ladder,
     and perspective-corrected 3D Doorway aperture with inner clearance envelope.

Usage:
    uv run python vlm/demo_hud_projection.py
"""

import sys
from pathlib import Path
import math
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
from PIL import Image

from vlm.hud_overlay import Aperture3D, HUDProjector

SCENE_PATH = Path("assets/skydio_x2/dual_scene.xml")
OUTPUT_DIR = Path("vlm/demo_outputs")


def setup_simulation():
    """Loads MuJoCo scene model and data."""
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene XML not found at {SCENE_PATH.resolve()}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    return model, data


def set_drone_pose(model, data, pos, rpy_deg):
    """Sets Drone 2 position and Euler roll/pitch/yaw angles.
    
    Conforms to standard aeronautical convention:
      - Positive pitch = nose UP (+Y body rotation negated in MuJoCo frame)
      - Negative pitch = nose DOWN
      - Positive roll  = banked right
    """
    roll = math.radians(rpy_deg[0])
    # Invert pitch for MuJoCo skydio body frame so negative pitch tilts nose DOWN
    pitch = -math.radians(rpy_deg[1])
    yaw = math.radians(rpy_deg[2])

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    d2_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "d2_root")
    if d2_root >= 0:
        adr = model.jnt_qposadr[d2_root]
        data.qpos[adr + 0] = pos[0]
        data.qpos[adr + 1] = pos[1]
        data.qpos[adr + 2] = pos[2]
        data.qpos[adr + 3] = qw
        data.qpos[adr + 4] = qx
        data.qpos[adr + 5] = qy
        data.qpos[adr + 6] = qz

    mujoco.mj_forward(model, data)


def run_demo():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, data = setup_simulation()
    projector = HUDProjector(camera_name="d2_fpv", width=480, height=360, fovy=90.0)

    print("=" * 72)
    print("🚁 [HUD PROJECTION DEMONSTRATOR] Generating Multimodal FPV Test Views")
    print("=" * 72)

    # --------------------------------------------------------------------------
    # Demo 1: Neutral Hover with Arch Gate ahead
    # --------------------------------------------------------------------------
    print("\n[Scene 1] Neutral Hover (1.0m alt) facing Arch Gate @ X=2.5m...")
    set_drone_pose(model, data, pos=[0.0, 1.2, 1.0], rpy_deg=[0.0, 0.0, 0.0])

    arch_gate = Aperture3D(
        name="ARCH_01",
        center=(2.5, 1.2, 1.0),
        width=1.1,
        height=1.3,
        yaw=0.0,
        shape="arch",
        color=(255, 190, 0),         # Amber Gold
        inner_color=(0, 255, 130),    # Neon Mint
        safety_margin=0.15,
        show_corridor=True,
        approach_distance=1.5,
    )

    img1 = projector.render_and_overlay(
        model, data,
        apertures=[arch_gate],
        radii=[1.0, 2.0, 3.0, 4.0],
    )
    p1 = OUTPUT_DIR / "demo_hud_neutral.jpg"
    img1.save(p1, quality=90)
    print(f"  ✅ Saved: {p1}")

    # --------------------------------------------------------------------------
    # Demo 2: Pitched-Down (12 deg) Forward Approach toward Window Aperture
    # --------------------------------------------------------------------------
    print("\n[Scene 2] Pitched-Down Forward Flight (-12° pitch) facing Window @ X=2.8m...")
    set_drone_pose(model, data, pos=[0.5, 1.2, 1.1], rpy_deg=[0.0, -12.0, 0.0])

    window_gate = Aperture3D(
        name="WINDOW_B2",
        center=(2.8, 1.2, 0.95),
        width=0.9,
        height=0.9,
        yaw=0.0,
        shape="window",
        color=(255, 160, 0),
        inner_color=(50, 255, 180),
        safety_margin=0.12,
        show_corridor=True,
        approach_distance=1.2,
    )

    img2 = projector.render_and_overlay(
        model, data,
        apertures=[window_gate],
        radii=[1.0, 2.0, 3.0],
        drone_vel=np.array([1.2, 0.0, -0.1]),
    )
    p2 = OUTPUT_DIR / "demo_hud_pitch_down.jpg"
    img2.save(p2, quality=90)
    print(f"  ✅ Saved: {p2}")

    # --------------------------------------------------------------------------
    # Demo 3: Banked Turn (+18 deg Roll) facing Doorway Aperture
    # --------------------------------------------------------------------------
    print("\n[Scene 3] Banked Turn (+18° roll, +4° pitch) facing Doorway @ X=3.0m...")
    set_drone_pose(model, data, pos=[0.2, 1.1, 1.05], rpy_deg=[18.0, 4.0, -5.0])

    door_gate = Aperture3D(
        name="DOORWAY_MAIN",
        center=(3.0, 1.2, 1.0),
        width=0.85,
        height=1.8,
        yaw=0.0,
        shape="door",
        color=(255, 215, 0),
        inner_color=(0, 255, 120),
        safety_margin=0.15,
        show_corridor=True,
        approach_distance=1.8,
    )

    img3 = projector.render_and_overlay(
        model, data,
        apertures=[door_gate],
        radii=[1.0, 2.0, 3.0, 4.0],
        drone_vel=np.array([0.9, 0.3, 0.0]),
    )
    p3 = OUTPUT_DIR / "demo_hud_banked.jpg"
    img3.save(p3, quality=90)
    print(f"  ✅ Saved: {p3}")

    print("\n" + "=" * 72)
    print("🎯 [ALL DEMO FRAMES GENERATED SUCCESSFULLY]")
    print(f"   Outputs located in: {OUTPUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
