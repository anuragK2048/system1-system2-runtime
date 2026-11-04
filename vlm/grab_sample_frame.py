"""Grab Sample FPV Frame from MuJoCo Skydio X2 Scene.

Renders an offscreen 480x360 FPV frame from the drone's onboard camera (d2_fpv)
and saves it to vlm/sample_fpv.jpg. Used as visual input for local VLM testing.

Usage:
    uv run python vlm/grab_sample_frame.py
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm.hud_overlay import HUDProjector

SCENE_PATH = Path("assets/skydio_x2/dual_scene.xml")
OUTPUT_PATH = Path("vlm/sample_fpv.jpg")
WIDTH = 480
HEIGHT = 360


def grab_fpv_frame(
    add_hud: bool = True,
    camera_name: str = "d2_fpv",
) -> Path:
    """Renders offscreen FPV frame with calibrated HUD projection and saves to disk."""
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene XML not found at {SCENE_PATH.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # Initialize nominal hover height (1.0m)
    d2_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "d2_root")
    if d2_root >= 0:
        adr = model.jnt_qposadr[d2_root]
        data.qpos[adr + 0] = 0.0   # x
        data.qpos[adr + 1] = 1.2   # y (left lane)
        data.qpos[adr + 2] = 1.0   # z (1.0m hover)
        data.qpos[adr + 3] = 1.0   # qw
        data.qpos[adr + 4] = 0.0
        data.qpos[adr + 5] = 0.0
        data.qpos[adr + 6] = 0.0

    mujoco.mj_forward(model, data)

    projector = HUDProjector(camera_name=camera_name, width=WIDTH, height=HEIGHT, fovy=90.0)

    if add_hud:
        img = projector.render_and_overlay(
            model=model,
            data=data,
            radii=[1.0, 2.0, 3.0, 4.0],
            drone_vel=np.array([0.0, 0.0, 0.0]),
        )
    else:
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        renderer.close()
        img = Image.fromarray(rgb)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH, format="JPEG", quality=85)
    print(f"✅ Saved calibrated FPV HUD frame to: {OUTPUT_PATH.resolve()} ({WIDTH}x{HEIGHT})")
    return OUTPUT_PATH


if __name__ == "__main__":
    grab_fpv_frame()
