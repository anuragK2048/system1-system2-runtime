"""FPV / 3rd-person camera MJPEG streaming for the web dashboard.

Serves a first-person-view feed from a MuJoCo camera rigidly attached to a
drone body (see ``d1_fpv`` / ``d2_fpv`` in ``assets/skydio_x2/dual_scene.xml``,
so the view tilts with the drone, or — with ``?view=third`` — a 3rd-person
feed from the fixed ``arena_3rd`` camera in the same scene. Consumed by a
plain ``<img>`` tag as ``multipart/x-mixed-replace`` -- no WebSocket involved.

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
import pathlib
import subprocess
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

# Fixed 3rd-person camera (worldbody in dual_scene.xml, overlooking arena).
THIRD_PERSON_CAMERA = "arena_3rd"
VIEWS = ("fpv", "third")

_renderer: Optional[mujoco.Renderer] = None
_renderer_model: Optional[mujoco.MjModel] = None
_renderer_camera: Optional[str] = None
_render_lock = threading.Lock()
_render_data: Optional[mujoco.MjData] = None
_render_data_model: Optional[mujoco.MjModel] = None   # <-- add this line
_stream_generation = 0

def _get_renderer(model: mujoco.MjModel, camera_name: str) -> mujoco.Renderer:
    """Single shared renderer, recreated when model or camera changes.
    (Multiple simultaneous renderers exhaust GL contexts on Windows;
    but reusing one across cameras corrupts its scene state.)"""
    global _renderer, _renderer_model, _renderer_camera
    if _renderer is None or _renderer_model is not model or _renderer_camera != camera_name:
        if _renderer is not None:
            try:
                _renderer.close()
            except Exception:
                pass
        _renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
        _renderer_model = model
        _renderer_camera = camera_name
    return _renderer

def _encode_jpeg(rgb) -> bytes:
    """Encodes an (H, W, 3) uint8 RGB array as JPEG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class _StaleStream(Exception):
    """Raised when a render belongs to a superseded stream (tab switched)."""


def _render_frame(sim, camera_name: str, generation: int) -> bytes:
    """Renders one frame and returns JPEG bytes. Blocking; call via to_thread."""
    global _render_data, _render_data_model
    with _render_lock:
        if generation != _stream_generation:
            # Superseded while queued: die without touching the shared renderer,
            # so only the live camera ever recreates GL state (no flicker).
            raise _StaleStream(camera_name)
        renderer = _get_renderer(sim.model, camera_name)
        # Snapshot sim state into a private MjData so the 500 Hz physics
        # thread can't tear the camera transform mid-render. Recreate it
        # when the model changes (e.g. after a setup-page relaunch), just
        # like the renderer above.
        if _render_data is None or _render_data_model is not sim.model:
            _render_data = mujoco.MjData(sim.model)
            _render_data_model = sim.model
        _render_data.qpos[:] = sim.data.qpos
        _render_data.qvel[:] = sim.data.qvel
        mujoco.mj_forward(sim.model, _render_data)
        renderer.update_scene(_render_data, camera=camera_name)
        frame = renderer.render()
    return _encode_jpeg(frame)

async def _frame_generator(camera_name: str, request: Request, generation: int) -> AsyncIterator[bytes]:
    interval = 1.0 / STREAM_FPS
    while True:
        if generation != _stream_generation:
            return
        if await request.is_disconnected():
            return
        t0 = time.time()
        sim = sim_context.sim
        if sim is None:
            return
        try:
            jpeg = await asyncio.to_thread(_render_frame, sim, camera_name, generation)
        except Exception:
            # Sim restarted / renderer invalidated / stream superseded
            # mid-stream: end this stream; the browser <img> will be
            # re-pointed by the dashboard.
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
    drone: str = Query("drone2"),
    view: str = Query("fpv"),
) -> StreamingResponse:
    """MJPEG stream: FPV from a drone camera, or 3rd-person arena view.

    ``view=fpv`` (default) uses ``drone`` (drone1|drone2); ``view=third``
    uses the fixed ``arena_3rd`` camera and ignores ``drone``. Unknown
    ``view`` (or unknown ``drone`` in FPV mode) -> 422.
    """
    if view not in VIEWS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown view '{view}'. Valid: {list(VIEWS)}",
        )
    if view == "third":
        camera_name = THIRD_PERSON_CAMERA
    else:
        if drone not in CAMERAS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown drone '{drone}'. Valid: {sorted(CAMERAS)}",
            )
        camera_name = CAMERAS[drone]
    global _stream_generation
    _stream_generation += 1
    my_generation = _stream_generation
    sim = sim_context.sim
    if sim is None:
        raise HTTPException(
            status_code=503,
            detail="Simulator is not running; start the viewer or --headless server first.",
        )
    if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
        raise HTTPException(
            status_code=503,
            detail=f"Camera '{camera_name}' not found in the loaded MJCF model.",
        )
    return StreamingResponse(
        _frame_generator(camera_name, request, my_generation),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/launch-viewer")
async def launch_viewer():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    try:
        subprocess.Popen(["uv", "run", "python", "src/runtime/sim/view_server_sim.py"], cwd=str(repo_root))
        return {"status": "launched"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sim-state")
async def sim_state():
    sim = sim_context.sim
    if sim is None: raise HTTPException(status_code=503, detail="Sim not running")
    return {"qpos": sim.data.qpos.tolist(), "time": float(sim.data.time)}

