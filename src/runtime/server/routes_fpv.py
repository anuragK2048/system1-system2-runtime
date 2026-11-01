"""FPV camera MJPEG streaming for the web dashboard.

Serves a first-person-view feed from a MuJoCo camera rigidly attached to a
drone body (see ``d1_fpv`` / ``d2_fpv`` in ``assets/skydio_x2/dual_scene.xml``,
so the view tilts with the drone. Consumed by a plain ``<img>`` tag as
``multipart/x-mixed-replace`` -- no WebSocket involved.

Rendering notes:
  - One ``mujoco.Renderer`` is shared by all clients and serialised with a
    lock: MuJoCo offscreen GL contexts are not thread-safe, and the physics
    thread mutates ``sim.data`` at 500 Hz while frames are rendered here.
  - The generator is async and checks ``request.is_disconnected()`` each
    frame, so closing the browser tab stops the render loop (no zombie
    threads). Blocking render calls run via ``asyncio.to_thread``.
  - Throttled to ~15 FPS at 640x480 to stay light next to 30 Hz telemetry.
"""

import asyncio
import io
import threading
import time
from typing import AsyncIterator, Optional

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
}

_renderer: Optional[mujoco.Renderer] = None
_renderer_model: Optional[mujoco.MjModel] = None
_render_lock = threading.Lock()
_render_data: Optional[mujoco.MjData] = None

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


def _encode_jpeg(rgb) -> bytes:
    """Encodes an (H, W, 3) uint8 RGB array as JPEG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _render_frame(sim, camera_name: str) -> bytes:
    """Renders one frame and returns JPEG bytes. Blocking; call via to_thread."""
    global _render_data
    with _render_lock:
        renderer = _get_renderer(sim.model)
        # Snapshot sim state into a private MjData so the 500 Hz physics
        # thread can't tear the camera transform mid-render.
        if _render_data is None:
            _render_data = mujoco.MjData(sim.model)
        _render_data.qpos[:] = sim.data.qpos
        _render_data.qvel[:] = sim.data.qvel
        mujoco.mj_forward(sim.model, _render_data)
        renderer.update_scene(_render_data, camera=camera_name)
        frame = renderer.render()
    return _encode_jpeg(frame)



async def _frame_generator(camera_name: str, request: Request) -> AsyncIterator[bytes]:
    interval = 1.0 / STREAM_FPS
    while True:
        if await request.is_disconnected():
            return
        t0 = time.time()
        sim = sim_context.sim
        if sim is None:
            return
        try:
            jpeg = await asyncio.to_thread(_render_frame, sim, camera_name)
        except Exception:
            # Sim restarted / renderer invalidated mid-stream: end this
            # stream; the browser <img> will be re-pointed by the dashboard.
            return
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
    drone: str = Query("drone2", pattern="^(drone1|drone2)$"),
) -> StreamingResponse:
    """MJPEG stream from the selected drone's body-fixed FPV camera."""
    sim = sim_context.sim
    if sim is None:
        raise HTTPException(
            status_code=503,
            detail="Simulator is not running; start the viewer or --headless server first.",
        )
    camera_name = CAMERAS[drone]
    if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
        raise HTTPException(
            status_code=503,
            detail=f"Camera '{camera_name}' not found in the loaded MJCF model.",
        )
    return StreamingResponse(
        _frame_generator(camera_name, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
