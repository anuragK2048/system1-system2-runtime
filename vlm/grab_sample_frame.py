"""Grab Sample FPV Frame from MuJoCo Skydio X2 Scene.

Renders an offscreen 480x360 FPV frame from the drone's onboard camera (d2_fpv)
and saves it to vlm/sample_fpv.jpg. Used as visual input for local VLM testing.

Usage:
    uv run python vlm/grab_sample_frame.py
"""

from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw

SCENE_PATH = Path("assets/skydio_x2/dual_scene.xml")
OUTPUT_PATH = Path("vlm/sample_fpv.jpg")
WIDTH = 480
HEIGHT = 360


def grab_fpv_frame(
    add_hud: bool = True,
    camera_name: str = "d2_fpv",
) -> Path:
    """Renders offscreen FPV frame and saves it to disk."""
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene XML not found at {SCENE_PATH.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # Initialize keyframe / nominal hover height (1.0m)
    # Drone 2 root joint address
    d2_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "d2_root")
    if d2_root >= 0:
        adr = model.jnt_qposadr[d2_root]
        data.qpos[adr + 0] = 0.0   # x
        data.qpos[adr + 1] = 1.2   # y (left lane)
        data.qpos[adr + 2] = 1.0   # z (1.0m hover)
        # Quaternion [w, x, y, z] neutral orientation
        data.qpos[adr + 3] = 1.0
        data.qpos[adr + 4] = 0.0
        data.qpos[adr + 5] = 0.0
        data.qpos[adr + 6] = 0.0

    mujoco.mj_forward(model, data)

    # Offscreen renderer
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    renderer.close()

    img = Image.fromarray(rgb)

    if add_hud:
        # Draw light HUD overlay (crosshairs + boresight marker)
        draw = ImageDraw.Draw(img)
        cx, cy = WIDTH // 2, HEIGHT // 2
        
        # Center boresight reticle
        r = 10
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(0, 255, 255), width=1)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(0, 255, 255), width=1)
        draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], outline=(0, 255, 255), width=1)

        # Subtle text HUD
        draw.text((10, 10), "FPV: d2_fpv | ALT: 1.00m | FOV: 90°", fill=(0, 255, 200))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH, format="JPEG", quality=80)
    print(f"✅ Saved sample FPV frame to: {OUTPUT_PATH.resolve()} ({WIDTH}x{HEIGHT})")
    return OUTPUT_PATH


if __name__ == "__main__":
    grab_fpv_frame()
