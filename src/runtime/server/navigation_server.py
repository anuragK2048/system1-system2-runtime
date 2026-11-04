"""FastAPI Server for Single-Drone Autonomous Target Navigation & Performance Dashboard.

Serves on a dedicated port (default: 8080) to provide:
1. Dedicated Navigation Cockpit & Evals Dashboard UI (`frontend/navigation_dashboard.html`).
2. High-performance MJPEG live stream of the single drone's FPV camera with calibrated
   avionics HUD overlay (ground rings 1m/2m/3m, artificial horizon line, target aperture frames).
3. 30-50 Hz WebSocket telemetry broadcasting position, velocity, attitude, L1 disturbance forces,
   motor thrusts, active flight phase, System 2 patches, and mission performance metrics.
4. Interactive REST APIs for mission selection (Alpha, Delta, Runway Poles, Window, Arch),
   chaos injection (wind, motor), and automated benchmark test suite execution.

Usage:
------
uv run python -m runtime.server.navigation_server --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from runtime.server.navigation_state import nav_coordinator
from runtime.chaos.presets import CHAOS_PRESETS
from runtime.brain.intent_parser import intent_parser


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_HTML = PROJECT_ROOT / "frontend" / "navigation_dashboard.html"


class MissionLaunchRequest(BaseModel):
    target: str = Field("alpha", description="Target: alpha | delta | runway_poles | window | arch | home | grand_tour")
    use_vlm: bool = Field(False, description="Enable Meta Llama 3.2 Vision reasoning on Modal GPU")
    model: str = Field("meta-llama/Llama-3.2-11B-Vision-Instruct (Modal A100 GPU)", description="VLM model identifier")
    chaos: Optional[str] = Field(None, description="Physical disturbance: wind | motor | None")
    duration: Optional[float] = Field(None, description="Max mission duration in seconds")
    reset: bool = Field(False, description="Force reset to origin pad (False allows seamless in-flight retargeting)")


class InstructionRequest(BaseModel):
    instruction: str = Field(..., description="Natural language flight instruction")
    use_llm: bool = Field(True, description="Enable LLM intent parsing")
    model: Optional[str] = Field(None, description="Optional LLM model override")


class ChaosRequest(BaseModel):
    chaos_type: Optional[str] = Field("wind", description="wind | motor | 1..5 | none")
    active: bool = Field(True, description="Enable or disable disturbance")


class ChaosPresetRequest(BaseModel):
    preset_id: str = Field(..., description="Preset ID ('1'..'5' or '0' for clean nominal)")
    toggle: bool = Field(False, description="If True, toggles preset on/off additively")
    additive: bool = Field(True, description="Whether to additively combine with active chaos")


class ChaosGustRequest(BaseModel):
    force: float = Field(6.0, ge=0.0, description="Peak gust force in Newtons")
    direction: List[float] = Field(default=[1.0, 0.0, 0.0], description="Gust 3D unit direction vector")
    duration_s: float = Field(0.8, gt=0.0, description="Gust duration in seconds")


# ==============================================================================
# WEBSOCKET CONNECTION MANAGER
# ==============================================================================

class NavigationWebSocketManager:
    """Manages active WebSocket telemetry streaming clients."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        async with self._lock:
            dead = []
            for ws in self.active_connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active_connections.discard(ws)


ws_manager = NavigationWebSocketManager()


# ==============================================================================
# FASTAPI APPLICATION & LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages startup and graceful shutdown of background broadcast loops."""
    print("🚁 [NAV SERVER] Single-Drone Autonomous Navigation API & Dashboard starting...")
    nav_coordinator.shutdown_event.clear()

    # Launch background WebSocket broadcaster task (at 30-40 Hz)
    broadcast_task = asyncio.create_task(_telemetry_broadcast_worker())

    yield

    print("🛑 [NAV SERVER] Shutting down navigation server...")
    nav_coordinator.shutdown_event.set()
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    nav_coordinator.close()


async def _telemetry_broadcast_worker():
    """Continuously broadcasts real-time telemetry to all connected WebSocket clients."""
    while not nav_coordinator.shutdown_event.is_set():
        try:
            if ws_manager.active_connections:
                telemetry = nav_coordinator.get_telemetry_snapshot()
                telemetry["server_time"] = time.time()
                await ws_manager.broadcast({
                    "type": "telemetry",
                    "data": telemetry,
                })
            await asyncio.sleep(0.025)  # 40 Hz telemetry broadcast
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.05)


def create_navigation_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Single-Drone Target Navigation & Performance Dashboard API",
        description="Autonomous single-drone target navigation, live FPV stream with avionics HUD, and evals testbed.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------------
    # Frontend Dashboard Route
    # --------------------------------------------------------------------------
    @app.get("/")
    def serve_dashboard():
        """Serves the Single-Drone Navigation Cockpit & Evals Dashboard."""
        if FRONTEND_HTML.exists():
            return FileResponse(
                str(FRONTEND_HTML),
                media_type="text/html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return {"status": "online", "message": "Navigation dashboard HTML not found."}

    # --------------------------------------------------------------------------
    # FPV Live Video Stream with HUD
    # --------------------------------------------------------------------------
    @app.get("/api/fpv/stream")
    async def fpv_live_stream(
        request: Request,
        hud: bool = Query(True, description="Overlay calibrated avionics HUD (ground rings, horizon, apertures)"),
    ) -> StreamingResponse:
        """Serves real-time multipart/x-mixed-replace MJPEG stream from the drone's FPV camera."""
        return StreamingResponse(
            nav_coordinator.frame_stream_generator(request, enable_hud=hud),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/fpv/frame")
    async def fpv_single_frame(
        hud: bool = Query(True, description="Overlay calibrated avionics HUD"),
    ) -> Response:
        """Retrieves a single high-resolution JPEG snapshot from the FPV camera."""
        try:
            frame_bytes = await nav_coordinator.get_fpv_frame_async(enable_hud=hud)
            return Response(
                content=frame_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --------------------------------------------------------------------------
    # WebSocket Real-Time Telemetry & Bi-Directional Control
    # --------------------------------------------------------------------------
    @app.websocket("/ws/telemetry")
    async def telemetry_websocket(websocket: WebSocket):
        """Bi-directional WebSocket for 40 Hz telemetry stream and real-time mission controls."""
        await ws_manager.connect(websocket)
        try:
            # Send initial state snapshot immediately upon connection
            init_snap = nav_coordinator.get_telemetry_snapshot()
            await websocket.send_text(json.dumps({"type": "init", "data": init_snap}))

            while not nav_coordinator.shutdown_event.is_set():
                # Listen for client control commands (e.g. start/stop/chaos/reset)
                data_text = await websocket.receive_text()
                try:
                    cmd = json.loads(data_text)
                    action = cmd.get("action")
                    if action == "start":
                        target = cmd.get("target", "alpha")
                        use_vlm = bool(cmd.get("use_vlm", False))
                        model = cmd.get("model", "meta-llama/Llama-3.2-11B-Vision-Instruct (Modal A100 GPU)")
                        chaos = cmd.get("chaos", None)
                        duration = cmd.get("duration", None)
                        reset = bool(cmd.get("reset", False))
                        res = nav_coordinator.start_mission(
                            target=target,
                            use_vlm=use_vlm,
                            model=model,
                            chaos=chaos,
                            duration=duration,
                            reset=reset,
                        )
                        await websocket.send_text(json.dumps({"type": "command_ack", "data": res}))
                    elif action == "retarget":
                        target = cmd.get("target", "alpha")
                        res = nav_coordinator.start_mission(target=target, reset=False)
                        await websocket.send_text(json.dumps({"type": "command_ack", "data": res}))
                    elif action == "stop":
                        res = nav_coordinator.stop_mission()
                        await websocket.send_text(json.dumps({"type": "command_ack", "data": res}))
                    elif action == "reset":
                        res = nav_coordinator.reset_to_origin()
                        await websocket.send_text(json.dumps({"type": "command_ack", "data": res}))
                    elif action == "chaos":
                        c_type = cmd.get("chaos_type")
                        c_active = bool(cmd.get("active", True))
                        res = nav_coordinator.set_chaos(c_type, c_active)
                        await websocket.send_text(json.dumps({"type": "command_ack", "data": res}))
                    elif action == "instruction":
                        inst_text = cmd.get("instruction", "")
                        use_llm = bool(cmd.get("use_llm", True))
                        model_override = cmd.get("model")
                        intent = await intent_parser.parse_instruction_async(
                            instruction=inst_text,
                            use_llm=use_llm,
                            model_override=model_override,
                        )
                        res = nav_coordinator.execute_instruction(intent)
                        await websocket.send_text(json.dumps({"type": "instruction_ack", "data": res}))
                except Exception as parse_err:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(parse_err)}))
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(websocket)

    # --------------------------------------------------------------------------
    # REST Navigation Mission Controls
    # --------------------------------------------------------------------------

    @app.post("/api/navigation/start")
    def start_mission_endpoint(req: MissionLaunchRequest):
        """Starts an autonomous flight mission (seamless in-flight retarget if airborne)."""
        return nav_coordinator.start_mission(
            target=req.target,
            use_vlm=req.use_vlm,
            model=req.model,
            chaos=req.chaos,
            duration=req.duration,
            reset=req.reset,
        )

    @app.post("/api/navigation/retarget")
    def retarget_mission_endpoint(req: MissionLaunchRequest):
        """Seamlessly transitions the airborne drone to a new target without resetting."""
        return nav_coordinator.start_mission(
            target=req.target,
            use_vlm=req.use_vlm,
            model=req.model,
            chaos=req.chaos,
            duration=req.duration,
            reset=False,
        )

    @app.post("/api/navigation/instruction")
    async def natural_language_instruction_endpoint(req: InstructionRequest):
        """Parses a natural language instruction using System 2 LLM and retargets airborne flight."""
        intent = await intent_parser.parse_instruction_async(
            instruction=req.instruction,
            use_llm=req.use_llm,
            model_override=req.model,
        )
        return nav_coordinator.execute_instruction(intent)

    @app.post("/api/navigation/stop")
    def stop_mission_endpoint():
        """Stops active mission."""
        return nav_coordinator.stop_mission()

    @app.post("/api/navigation/reset")
    def reset_mission_endpoint():
        """Resets drone back to origin pad."""
        return nav_coordinator.reset_to_origin()

    @app.post("/api/navigation/chaos")
    def chaos_endpoint(req: ChaosRequest):
        """Backward-compatible disturbance toggle during flight."""
        return nav_coordinator.set_chaos(req.chaos_type, req.active)

    @app.post("/api/navigation/chaos/preset")
    @app.post("/api/chaos/preset")
    def chaos_preset_endpoint(req: ChaosPresetRequest):
        """Applies or toggles a named chaos preset ('1'..'5' or '0' for clean nominal)."""
        return nav_coordinator.apply_chaos_preset(req.preset_id, toggle=req.toggle, additive=req.additive)

    @app.post("/api/navigation/chaos/toggle")
    @app.post("/api/chaos/preset/toggle")
    def chaos_toggle_endpoint(req: ChaosPresetRequest):
        """Explicit endpoint to toggle a named failure preset on or off additively."""
        return nav_coordinator.apply_chaos_preset(req.preset_id, toggle=True, additive=True)

    @app.post("/api/navigation/chaos/gust")
    @app.post("/api/chaos/gust")
    def chaos_gust_endpoint(req: ChaosGustRequest):
        """Triggers an instantaneous half-sine crosswind gust."""
        return nav_coordinator.trigger_wind_gust(force=req.force, direction=req.direction, duration_s=req.duration_s)

    @app.post("/api/navigation/chaos/clear")
    @app.post("/api/chaos/clear")
    def chaos_clear_endpoint():
        """Clears all active disturbances back to pristine nominal flight."""
        return nav_coordinator.clear_all_chaos()

    @app.get("/api/navigation/chaos/state")
    @app.get("/api/chaos/state")
    def chaos_state_endpoint():
        """Returns the full serialization of current chaos state."""
        return {
            "summary": nav_coordinator.sim.chaos.get_chaos_summary(),
            "active_presets": nav_coordinator.sim.chaos.get_active_preset_keys(),
            "chaos_mode": nav_coordinator.chaos_mode,
        }

    @app.get("/api/chaos/presets")
    def chaos_presets_catalog_endpoint():
        """Returns the catalog of all available failure presets with metadata."""
        catalog = []
        for k, p in CHAOS_PRESETS.items():
            catalog.append({
                "key": k,
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "motor_active": p["motor"].is_active(),
                "wind_active": p["wind"].is_active(),
                "payload_active": p["payload"].is_active(),
                "sensor_active": p["sensor"].is_active(),
            })
        return catalog

    @app.get("/api/navigation/telemetry")
    def telemetry_endpoint():
        """Returns latest telemetry snapshot and recent rolling history."""
        return {
            "latest": nav_coordinator.get_telemetry_snapshot(),
            "history": list(nav_coordinator.telemetry_history),
        }

    @app.get("/api/navigation/events")
    def events_endpoint():
        """Returns recent cognitive decisions, triggers, and phase transitions."""
        return {"events": list(nav_coordinator.vlm_events)}

    @app.get("/api/navigation/tests")
    def list_tests_endpoint():
        """Returns the catalog of available navigation test scenarios."""
        return {
            "tests": [
                {
                    "id": "NAV-01",
                    "target": "alpha",
                    "name": "Target Pad Alpha Alignment & Hover",
                    "description": "Corridor advancement past hazard pylons and station-keeping hover over Pad Alpha concentric rings (X=5.5m).",
                    "phase_flow": ["LIFTOFF", "CORRIDOR_APPROACH", "PRECISION_HOVER"],
                    "nominal_duration": 14.0,
                    "target_pos": [5.5, 0.0, 1.0],
                },
                {
                    "id": "NAV-02",
                    "target": "runway_poles",
                    "name": "Runway Gate: Hover Centered Between Poles",
                    "description": "Navigate forward into entrance gate corridor and achieve centered hover between runway poles at X=3.5m, Y=±0.9m.",
                    "phase_flow": ["LIFTOFF", "CORRIDOR_APPROACH", "PRECISION_HOVER"],
                    "nominal_duration": 14.0,
                    "target_pos": [3.5, 0.0, 1.0],
                },
                {
                    "id": "NAV-03",
                    "target": "delta",
                    "name": "Elevated Platform Delta: Caution Disc Hover",
                    "description": "Fly beyond Pad Alpha to the elevated pedestal (0.7m high) and hold station atop the Delta caution disc at Z=1.10m.",
                    "phase_flow": ["LIFTOFF", "CORRIDOR_APPROACH", "PRECISION_HOVER"],
                    "nominal_duration": 14.0,
                    "target_pos": [8.0, 0.0, 1.10],
                },
                {
                    "id": "NAV-04",
                    "target": "window",
                    "name": "Left Window: Saccade +90° & Aperture Ingress",
                    "description": "Perform in-place +90° yaw saccade to bring unobservable window into FPV FOV, advance along glidepath, clamp tilt <= 12°, and traverse.",
                    "phase_flow": ["LIFTOFF", "SECTOR_SCAN", "CORRIDOR_APPROACH", "APERTURE_INGRESS", "EGRESS_STABILIZE"],
                    "nominal_duration": 24.0,
                    "target_pos": [0.0, 6.0, 1.10],
                },
                {
                    "id": "NAV-05",
                    "target": "arch",
                    "name": "Right Arch: Saccade -90° & Aperture Ingress",
                    "description": "Perform in-place -90° yaw saccade to visually frame the right arch, advance along glidepath, clamp tilt <= 12°, and traverse opening.",
                    "phase_flow": ["LIFTOFF", "SECTOR_SCAN", "CORRIDOR_APPROACH", "APERTURE_INGRESS", "EGRESS_STABILIZE"],
                    "nominal_duration": 24.0,
                    "target_pos": [0.0, -6.0, 0.90],
                },
            ]
        }

    @app.post("/api/navigation/run_suite")
    async def run_suite_endpoint():
        """Executes full automated test suite in a non-blocking thread."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, nav_coordinator.run_suite_evals)
        return {"status": "completed", "results": results}

    @app.get("/api/navigation/suite_results")
    def get_suite_results_endpoint():
        """Returns the latest benchmark evaluation scorecard."""
        return {"results": nav_coordinator.suite_results}

    @app.post("/api/sim/toggle_native_hud")
    def toggle_native_hud_endpoint():
        """Native 3D HUD overlay disabled per user request (pure native MuJoCo view)."""
        return {"native_hud_enabled": False}

    @app.get("/api/status")
    def status_endpoint():
        """Health and coordinator status endpoint."""
        return {
            "status": "online",
            "simulator": "DroneSimulator (Single Drone Arena)",
            "camera": "fpv (with HUDProjector ground rings & apertures)",
            "mission": nav_coordinator.target_mission,
            "phase": nav_coordinator.navigator.phase,
            "mission_status": nav_coordinator.mission_status,
            "sim_time": float(round(nav_coordinator.sim.data.time, 2)),
            "use_vlm": nav_coordinator.use_vlm,
            "vlm_model": nav_coordinator.vlm_model,
            "chaos_mode": nav_coordinator.chaos_mode,
            "native_hud_enabled": False,
        }

    return app


app = create_navigation_app()


def main():
    """Parses CLI arguments and launches navigation server on specified port."""
    import os
    import sys
    import threading
    parser = argparse.ArgumentParser(description="Single-Drone Target Navigation Dashboard & Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--autostart", type=str, default=None, choices=["alpha", "delta", "runway_poles", "window", "arch", "grand_tour"], help="Autostart target mission on launch")
    parser.add_argument("--use-vlm", action="store_true", help="Enable Meta Llama 3.2 Vision on Modal GPU on launch")
    parser.add_argument("--gui", action="store_true", help="Launch native 60 FPS MuJoCo 3D viewer window in parallel with web cockpit")
    args = parser.parse_args()

    # macOS MuJoCo Native Viewer Trampoline
    if args.gui and sys.platform == "darwin":
        try:
            import mujoco.viewer
            if not isinstance(getattr(mujoco.viewer, "_MJPYTHON", None), getattr(mujoco.viewer, "_MjPythonBase", object)):
                exe_dir = Path(sys.executable).parent
                mjpy = exe_dir / "mjpython"
                if mjpy.exists():
                    os.execv(str(mjpy), [str(mjpy)] + sys.argv)
        except Exception:
            pass

    if args.autostart:
        nav_coordinator.start_mission(target=args.autostart, use_vlm=args.use_vlm)

    print("\n" + "═" * 78)
    print(f"🚁 [SINGLE-DRONE TARGET NAVIGATION DASHBOARD]")
    print(f"🌐 Server URL:     http://localhost:{args.port}")
    print(f"📺 Live FPV Feed:  http://localhost:{args.port}/api/fpv/stream?hud=true")
    print(f"📡 WebSocket:      ws://localhost:{args.port}/ws/telemetry")
    if args.gui:
        print(f"🖥️ Parallel 3D:    Native MuJoCo OpenGL 60 FPS Viewport Active")
        print(f"   • Auto-tracking quadrotor chase camera")
        print(f"   • Press '[' or ']' in 3D window to switch camera perspectives")
        print(f"   • Press '1'-'5' for Chaos Presets | '0' to Clear All")
    print("═" * 78 + "\n")

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_graceful_shutdown=0.5,
    )
    server = uvicorn.Server(config)

    try:
        if args.gui:
            import mujoco.viewer

            # Run uvicorn on daemon thread, MuJoCo Cocoa/GLFW window on macOS main thread
            server_thread = threading.Thread(target=server.run, daemon=True, name="UvicornServerThread")
            server_thread.start()

            try:
                with mujoco.viewer.launch_passive(
                    nav_coordinator.sim.model,
                    nav_coordinator.sim.data,
                    key_callback=nav_coordinator.sim.chaos.handle_key,
                ) as viewer:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = nav_coordinator.sim.drone_body_id
                    viewer.cam.distance = 2.8
                    viewer.cam.elevation = -18.0

                    last_cam_mode = None

                    while viewer.is_running() and not nav_coordinator.shutdown_event.is_set():
                        # Synchronize with physics step loop so viewer.sync never copies mid-step
                        with nav_coordinator.sim_lock:
                            # Current drone orientation yaw angle from simulation quaternion [w, qx, qy, qz]
                            q = nav_coordinator.sim.data.qpos[3:7]
                            drone_yaw_deg = float(np.degrees(
                                np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
                            ))

                            with viewer.lock():
                                # Keep chase camera locked behind drone and rotating with heading in TPP
                                if viewer.cam.fixedcamid == -1:
                                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                                    viewer.cam.trackbodyid = nav_coordinator.sim.drone_body_id
                                    viewer.cam.azimuth = drone_yaw_deg
                                    if viewer.cam.distance <= 0.1:
                                        viewer.cam.distance = 2.8
                                    if viewer.cam.elevation > -5.0 or viewer.cam.elevation < -85.0:
                                        viewer.cam.elevation = -18.0
                                elif viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                                    viewer.cam.azimuth = drone_yaw_deg

                                current_cam_tuple = (int(viewer.cam.type), int(viewer.cam.fixedcamid))
                                if current_cam_tuple != last_cam_mode:
                                    last_cam_mode = current_cam_tuple
                                    if viewer.cam.fixedcamid == 1:
                                        print("🎥 [VIEWPORT CAMERA] First-Person Cockpit View (FPV Nose Mount)")
                                    elif viewer.cam.fixedcamid == 0:
                                        print("🎥 [VIEWPORT CAMERA] Fixed Third-Person Chase Camera (Attached to Drone Frame)")
                                    else:
                                        print("🎥 [VIEWPORT CAMERA] Dynamic Third-Person Chase Camera (Auto-Aligning with Heading)")

                                viewer.sync()
                        time.sleep(1.0 / 60.0)
            except RuntimeError as gui_err:
                print(f"\n⚠️  [GUI LAUNCH ERROR] {gui_err}")
                print("👉 On macOS, launch with mjpython to open the native 3D OpenGL window:")
                print(f"   uv run mjpython scripts/run_navigation_dashboard.py --gui --port {args.port}\n")
                print(f"🌐 Web Cockpit is online at: http://localhost:{args.port}")
                while not nav_coordinator.shutdown_event.is_set():
                    time.sleep(0.5)
        else:
            server.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav_coordinator.shutdown_event.set()
        nav_coordinator.close()


if __name__ == "__main__":
    main()
