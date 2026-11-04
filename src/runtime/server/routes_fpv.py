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
import threading
import time
from typing import AsyncIterator, Optional

import numpy as np
import mujoco
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from runtime.server.state import sim_context

router = APIRouter(prefix="/api/fpv", tags=["FPV Camera"])

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STREAM_FPS = 15.0
JPEG_QUALITY = 80

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


def cleanup_renderer() -> None:
    """Closes the MuJoCo renderer and shuts down the dedicated render thread.

    (for macOS & Windows cross-platform compatibility: cleanly releases the OpenGL context
    on the exact thread that initialized it to avoid driver crashes on shutdown.)
    """
    global _renderer, _renderer_model, _render_data, _render_pool
    if _render_pool is None:
        return

    def _do_close():
        global _renderer, _renderer_model, _render_data
        with _render_lock:
            if _renderer is not None:
                try:
                    _renderer.close()
                except Exception:
                    pass
                _renderer = None
                _renderer_model = None
                _render_data = None

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


def _render_frame(sim, camera_name: str) -> bytes:
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
    return _encode_jpeg(frame)


async def _frame_generator(camera_name: str, request: Request) -> AsyncIterator[bytes]:
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
            jpeg = await loop.run_in_executor(pool, _render_frame, sim, camera_name)
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
        _frame_generator(camera_name, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
