"""FPV camera MJPEG streaming for the web dashboard.

Serves a first-person-view feed from a MuJoCo camera rigidly attached to a
drone body (see ``d1_fpv`` / ``d2_fpv`` in ``assets/skydio_x2/dual_scene.xml``),
so the view tilts with the drone. Consumed by a plain ``<img>`` tag as
``multipart/x-mixed-replace`` -- no WebSocket involved.

Cross-Platform Compatibility Notes:
  - MuJoCo offscreen GL contexts (CGL on macOS, WGL on Windows) are thread-bound.
    All rendering is serialized on a dedicated single-worker ThreadPoolExecutor
    so that the OpenGL context remains pinned to the same thread and never deadlocks
    or fails with context errors.
  - The physics state is snapshotted under `sim.physics_lock` so mid-step resets
    or 500 Hz updates cannot tear the camera transform or corrupt MjData on any OS.
  - The generator is async and checks `request.is_disconnected()` and
    `sim_context.shutdown_event` each frame so closing the browser tab or hitting
    Ctrl+C immediately stops the render loop without blocking server shutdown.
  - Throttled to ~15 FPS at 640x480 to stay light next to 30 Hz telemetry.
"""

import asyncio
import concurrent.futures
import io
import math
import threading
import time
from typing import AsyncIterator, Optional

import numpy as np
import mujoco
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from runtime.server.state import sim_context
from vlm.hud_overlay import HUDProjector, Aperture3D

router = APIRouter(prefix="/api/fpv", tags=["FPV Camera"])

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STREAM_FPS = 15.0
JPEG_QUALITY = 80

# Known arena apertures for HUD projection
ARENA_APERTURES = [
    Aperture3D(
        name="WINDOW_LEFT",
        center=(0.0, 5.0, 1.10),
        width=1.40,
        height=1.20,
        yaw=math.radians(90.0),
        shape="window",
        color=(255, 190, 0),       # Amber Gold
        inner_color=(0, 255, 130),  # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    ),
    Aperture3D(
        name="ARCH_RIGHT",
        center=(0.0, -5.0, 0.90),
        width=1.50,
        height=1.85,
        yaw=math.radians(-90.0),
        shape="arch",
        color=(255, 140, 0),       # High-Vis Orange
        inner_color=(0, 255, 130),  # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    ),
]

# Dashboard drone id -> MJCF camera name (child of that drone's body).
CAMERAS = {
    "drone1": "d1_fpv",
    "drone2": "d2_fpv",
    "drone": "fpv",
    "single": "fpv",
    "fpv": "fpv",
}

_renderer: Optional[mujoco.Renderer] = None
_renderer_model: Optional[mujoco.MjModel] = None
_hud_projector: Optional[HUDProjector] = None
_render_lock = threading.Lock()
_render_data: Optional[mujoco.MjData] = None

# (for macOS & Windows cross-platform compatibility: OpenGL offscreen contexts
# - CGL on macOS, WGL on Windows - are thread-bound. Calling renderer functions across
# arbitrary worker threads in asyncio.to_thread causes deadlocks on macOS and
# 'wglMakeCurrent' context failures on Windows. Pinning offscreen rendering to a
# single dedicated worker thread guarantees stability across all platforms.)
_render_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_render_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Returns the dedicated single-worker render thread pool, recreating if shut down."""
    global _render_pool
    if _render_pool is None:
        _render_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="MujocoRenderWorker",
        )
    return _render_pool


def _get_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
    """Returns the shared renderer, (re)creating it if the model changed."""
    global _renderer, _renderer_model
    if _renderer is None or _renderer_model is not model:
        if _renderer is not None:
            try:
                _renderer.close()
            except Exception:
                pass
        _renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
        _renderer_model = model
    return _renderer


def _get_hud(camera_name: str) -> HUDProjector:
    """Returns the shared HUD projector for the camera."""
    global _hud_projector
    if _hud_projector is None or _hud_projector.camera_name != camera_name:
        _hud_projector = HUDProjector(
            camera_name=camera_name,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )
    return _hud_projector


def cleanup_renderer() -> None:
    """Closes the MuJoCo renderer and shuts down the dedicated render thread.

    (for macOS & Windows cross-platform compatibility: cleanly releases the OpenGL context
    on the exact thread that initialized it to avoid driver crashes on shutdown.)
    """
    global _renderer, _renderer_model, _render_data, _render_pool, _hud_projector
    if _render_pool is None:
        return

    def _do_close():
        global _renderer, _renderer_model, _render_data, _hud_projector
        with _render_lock:
            if _renderer is not None:
                try:
                    _renderer.close()
                except Exception:
                    pass
                _renderer = None
                _renderer_model = None
                _render_data = None
                _hud_projector = None

    try:
        fut = _render_pool.submit(_do_close)
        fut.result(timeout=0.5)
    except Exception:
        pass
    _render_pool.shutdown(wait=False)
    _render_pool = None


def _encode_jpeg(rgb: np.ndarray) -> bytes:
    """Encodes an (H, W, 3) uint8 RGB array as JPEG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _render_frame(sim, camera_name: str, enable_hud: bool = True) -> bytes:
    """Renders one frame and returns JPEG bytes. Blocking; called on dedicated render thread."""
    global _render_data
    with _render_lock:
        renderer = _get_renderer(sim.model)
        if _render_data is None:
            _render_data = mujoco.MjData(sim.model)

        # (for macOS & Windows cross-platform thread safety: snapshot simulation qpos/qvel
        # under physics_lock to prevent mid-step state tearing while physics loop runs concurrently)
        p_lock = getattr(sim, "physics_lock", None)
        if p_lock:
            with p_lock:
                np.copyto(_render_data.qpos, sim.data.qpos)
                np.copyto(_render_data.qvel, sim.data.qvel)
        else:
            np.copyto(_render_data.qpos, sim.data.qpos)
            np.copyto(_render_data.qvel, sim.data.qvel)

        mujoco.mj_forward(sim.model, _render_data)
        renderer.update_scene(_render_data, camera=camera_name)
        frame = renderer.render()

        if enable_hud:
            try:
                from PIL import Image

                hud = _get_hud(camera_name)
                cam = hud.extract_camera_calibration(sim.model, _render_data)

                # Resolve drone body position & velocity across single and dual models
                d_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "x2")
                if d_id < 0:
                    d_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
                if d_id < 0 and camera_name.startswith("d2_"):
                    d_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "x2_2")
                if d_id < 0 and camera_name.startswith("d1_"):
                    d_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "x2_1")

                drone_pos = _render_data.xpos[d_id] if d_id >= 0 else None
                drone_vel = _render_data.cvel[d_id][3:6] if d_id >= 0 else None

                base_img = Image.fromarray(frame)
                overlaid = hud.overlay_hud(
                    base_img,
                    cam=cam,
                    drone_pos=drone_pos,
                    drone_vel=drone_vel,
                    apertures=ARENA_APERTURES,
                    radii=[1.0, 2.0, 3.0],
                )
                buf = io.BytesIO()
                overlaid.save(buf, format="JPEG", quality=JPEG_QUALITY)
                return buf.getvalue()
            except Exception:
                pass

        return _encode_jpeg(frame)


async def _frame_generator(camera_name: str, request: Request, enable_hud: bool = True) -> AsyncIterator[bytes]:
    interval = 1.0 / STREAM_FPS
    loop = asyncio.get_running_loop()
    pool = _get_render_pool()

    # (for macOS & Windows cross-platform compatibility: check shutdown_event every loop
    # to terminate stream immediately on Ctrl+C without waiting for uvicorn timeout)
    while not sim_context.shutdown_event.is_set():
        if await request.is_disconnected():
            return
        t0 = time.time()
        sim = sim_context.sim
        if sim is None:
            await asyncio.sleep(0.1)
            continue
        try:
            jpeg = await loop.run_in_executor(pool, _render_frame, sim, camera_name, enable_hud)
        except asyncio.CancelledError:
            return
        except Exception:
            # (for cross-platform stream resilience: during resets or transient physics spikes,
            # sleep briefly and retry rather than permanently severing the HTTP stream)
            await asyncio.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        elapsed = time.time() - t0
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)


@router.get("/stream")
async def fpv_stream(
    request: Request,
    drone: str = Query("drone2", pattern="^(drone1|drone2|drone|single|fpv)$"),
    hud: bool = Query(True, description="Overlay calibrated avionics HUD (ground rings, horizon, aperture frames)"),
) -> StreamingResponse:
    """MJPEG stream from the selected drone's body-fixed FPV camera."""
    sim = sim_context.sim
    if sim is None:
        raise HTTPException(
            status_code=503,
            detail="Simulator is not running; start the viewer or --headless server first.",
        )
    camera_name = CAMERAS.get(drone, "fpv")
    if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
        # Fallback resolution across single and dual drone scene models:
        # If camera not found by mapped name, try fallback candidates
        fallback_candidates = [camera_name, "fpv", "d1_fpv", "d2_fpv"]
        found_name = None
        for candidate in fallback_candidates:
            if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, candidate) >= 0:
                found_name = candidate
                break
        if found_name is not None:
            camera_name = found_name
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Camera '{camera_name}' not found in the loaded MJCF model.",
            )
    return StreamingResponse(
        _frame_generator(camera_name, request, enable_hud=hud),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
