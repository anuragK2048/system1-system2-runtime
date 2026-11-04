"""Thread-Safe Single-Drone Navigation Simulation Coordinator & State Manager.

Coordinates:
1. 500 Hz MuJoCo single-drone physics simulation in the navigation arena (Pad Alpha,
   Platform Delta, Runway Poles, Left Window, Right Arch).
2. AutonomousTargetNavigator state machine (Liftoff -> Sector Scan -> Corridor Approach ->
   Aperture Ingress -> Precision Hover -> Egress).
3. System 1 graceful smoothing filter and condition-based System 2 VLM triggering.
4. Offscreen FPV camera rendering with calibrated avionics HUD overlay (ground metric rings,
   artificial horizon line, aperture 3D frames, and reticle).
5. High-frequency telemetry snapshots and benchmark evals tracking.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from PIL import Image

from runtime.sim.config import DroneConfig, SimConfig
from runtime.sim.simulator import DroneSimulator
from runtime.sim.autonomous_target_flight import AutonomousTargetNavigator, SCENE_PATH
from vlm.hud_overlay import HUDProjector, Aperture3D


# Standard Arena Apertures for FPV HUD Overlay
ARENA_APERTURES = [
    Aperture3D(
        name="WINDOW_LEFT",
        center=(0.0, 5.0, 1.10),
        width=1.40,
        height=1.20,
        yaw=math.radians(90.0),
        shape="window",
        color=(255, 190, 0),        # Amber Gold
        inner_color=(0, 255, 130),   # Neon Mint
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
        color=(255, 140, 0),        # High-Vis Orange
        inner_color=(0, 255, 130),   # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    ),
]

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
STREAM_FPS = 15.0
JPEG_QUALITY = 80


class NavigationSimulationCoordinator:
    """Thread-safe coordinator for single-drone autonomous flight simulation and dashboard streaming."""

    _instance: Optional["NavigationSimulationCoordinator"] = None
    _singleton_lock = threading.RLock()

    def __new__(cls) -> "NavigationSimulationCoordinator":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        with self._singleton_lock:
            self._lock = threading.RLock()
            self.shutdown_event = threading.Event()

            # Single Drone Simulator and Navigator instances
            self.drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
            self.sim: DroneSimulator = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=self.drone_cfg)
            self.target_mission: str = "alpha"
            self.use_vlm: bool = False
            self.vlm_model: str = "gemini-3.5-flash-lite"
            self.chaos_mode: Optional[str] = None
            self.max_duration: float = 14.0

            # Offscreen OpenGL Renderer State (lazily initialized on the dedicated worker thread)
            self._renderer: Optional[mujoco.Renderer] = None
            self._hud: Optional[HUDProjector] = None
            self._render_data: Optional[mujoco.MjData] = None
            self._render_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="NavRendererWorker"
            )

            # Instantiate Navigator
            self.navigator: AutonomousTargetNavigator = AutonomousTargetNavigator(
                sim=self.sim,
                target_mission=self.target_mission,
                use_vlm=self.use_vlm,
                vlm_model=self.vlm_model,
                chaos_type=self.chaos_mode,
            )

            # Simulation Execution State
            self.is_running: bool = False
            self.is_paused: bool = False
            self.mission_status: str = "idle"  # "idle" | "running" | "completed" | "aborted"
            self.mission_start_time: float = 0.0
            self.mission_elapsed_s: float = 0.0

            # Background Worker Thread
            self._sim_thread: Optional[threading.Thread] = None

            # Telemetry Ring Buffer (up to 120 samples ~ 4 seconds at 30Hz)
            self.telemetry_history: Deque[Dict[str, Any]] = deque(maxlen=120)
            self.latest_telemetry: Dict[str, Any] = {}

            # Event / Trigger Log (up to 50 entries)
            self.vlm_events: Deque[Dict[str, Any]] = deque(maxlen=50)

            # Performance Metrics Accumulators
            self.peak_tilt_deg: float = 0.0
            self.peak_l1_force_N: float = 0.0
            self.tracking_errors: List[float] = []

            # Suite evaluation results cache
            self.suite_results: List[Dict[str, Any]] = []

            self._initialized = True

    # ==========================================================================
    # MISSION CONTROLS
    # ==========================================================================

    def start_mission(
        self,
        target: str = "alpha",
        use_vlm: bool = False,
        model: str = "gemini-3.5-flash-lite",
        chaos: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Configures and launches autonomous flight mission."""
        with self._lock:
            # Stop existing run if active
            self.stop_mission(status="aborted")

            # Determine mission duration
            if duration is not None and duration > 0.0:
                self.max_duration = float(duration)
            else:
                self.max_duration = 24.0 if target in ("window", "arch", "grand_tour") else 14.0

            self.target_mission = target
            self.use_vlm = use_vlm
            self.vlm_model = model
            self.chaos_mode = chaos if chaos in ("wind", "motor", "noise") else None

            # Reset simulator physics & drone pose
            self.sim.reset_simulation()
            self.sim.data.time = 0.0
            self.sim.data.qpos[0:3] = [0.0, 0.0, 0.8]
            self.sim.data.qvel[:] = 0.0
            mujoco.mj_forward(self.sim.model, self.sim.data)

            # Close old navigator renderer
            try:
                self.navigator.close()
            except Exception:
                pass

            # Create fresh Navigator
            self.navigator = AutonomousTargetNavigator(
                sim=self.sim,
                target_mission=self.target_mission,
                use_vlm=self.use_vlm,
                vlm_model=self.vlm_model,
                chaos_type=self.chaos_mode,
            )

            # Reset performance metrics
            self.peak_tilt_deg = 0.0
            self.peak_l1_force_N = 0.0
            self.tracking_errors.clear()
            self.telemetry_history.clear()
            self.mission_status = "running"
            self.is_running = True
            self.is_paused = False
            self.mission_start_time = time.time()
            self.mission_elapsed_s = 0.0

            # Launch physics thread
            self._sim_thread = threading.Thread(
                target=self._simulation_loop,
                daemon=True,
                name="NavPhysicsLoop",
            )
            self._sim_thread.start()

            # Record launch event
            self._record_event(
                sim_time=0.0,
                event_type="MISSION_START",
                reason=f"Launched mission '{self.target_mission.upper()}' (VLM: {self.use_vlm}, Chaos: {self.chaos_mode})",
                diagnosis=f"Flight Phase: {self.navigator.phase}",
            )

            return {
                "status": "started",
                "mission": self.target_mission,
                "use_vlm": self.use_vlm,
                "model": self.vlm_model,
                "chaos": self.chaos_mode,
                "duration": self.max_duration,
            }

    def stop_mission(self, status: str = "stopped") -> Dict[str, Any]:
        """Stops active mission execution and transitions drone to reflexive hold."""
        with self._lock:
            self.is_running = False
            self.mission_status = status
            sim_th = self._sim_thread

        if sim_th and sim_th.is_alive() and threading.current_thread() != sim_th:
            try:
                sim_th.join(timeout=1.0)
            except Exception:
                pass
        return {"status": self.mission_status, "sim_time": float(self.sim.data.time)}

    def reset_to_origin(self) -> Dict[str, Any]:
        """Resets the drone back to origin pad and halts movement."""
        with self._lock:
            self.stop_mission(status="idle")
            self.sim.reset_simulation()
            self.sim.data.time = 0.0
            self.sim.data.qpos[0:3] = [0.0, 0.0, 0.8]
            self.sim.data.qvel[:] = 0.0
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.navigator.phase = "PHASE_0_LIFTOFF"
            self.navigator.mission_complete = False
            self.telemetry_history.clear()
            self.peak_tilt_deg = 0.0
            self.peak_l1_force_N = 0.0
            self.tracking_errors.clear()
            return {"status": "reset", "pos": [0.0, 0.0, 0.8]}

    def set_chaos(self, chaos_type: Optional[str], active: bool) -> Dict[str, Any]:
        """Toggles real-time physical disturbance during flight."""
        with self._lock:
            if not active or chaos_type == "none":
                self.chaos_mode = None
                self.sim.data.xfrc_applied[self.sim.drone_body_id, :] = 0.0
                self.sim.chaos.active_chaos["motor_failure"] = False
                self.sim.chaos.active_chaos["wind_gust"] = False
            elif chaos_type == "wind":
                self.chaos_mode = "wind"
                self.sim.data.xfrc_applied[self.sim.drone_body_id, 0] = 5.0
            elif chaos_type == "motor":
                self.chaos_mode = "motor"
                self.sim.chaos.active_chaos["motor_failure"] = True

            return {"chaos_mode": self.chaos_mode}

    # ==========================================================================
    # BACKGROUND SIMULATION LOOP (500 Hz Physics + Cognitive Loop)
    # ==========================================================================

    def _simulation_loop(self) -> None:
        """Runs the 500 Hz physics stepping synchronized to real-time execution."""
        dt = self.sim.sim_cfg.physics_timestep  # 0.002s (500 Hz)
        wall_start = time.time()
        sim_start_t = float(self.sim.data.time)

        try:
            while self.is_running and not self.shutdown_event.is_set():
                t_sim = float(self.sim.data.time)

                if t_sim >= self.max_duration:
                    with self._lock:
                        self.mission_status = "completed" if self.navigator.mission_complete else "timeout"
                        self.is_running = False
                    break

                # 1. System 2 Cognitive Cycle (2 Hz or event-driven)
                entry = self.navigator.step_cognitive_cycle(t_sim)
                if entry and entry.get("vlm_invoked"):
                    self._record_event(
                        sim_time=t_sim,
                        event_type="VLM_INVOCATION",
                        reason=entry.get("trigger_reason", "Event triggered"),
                        diagnosis=entry.get("diagnosis", ""),
                    )
                elif entry and entry.get("trigger_reason") and "transition" in entry.get("trigger_reason", "").lower():
                    self._record_event(
                        sim_time=t_sim,
                        event_type="PHASE_TRANSITION",
                        reason=entry.get("trigger_reason"),
                        diagnosis=entry.get("diagnosis", ""),
                    )

                # 2. Chaos Disturbance Injection
                if self.chaos_mode:
                    self.navigator.apply_chaos(t_sim)

                # 3. System 1 Reflex Cycle (500 Hz MuJoCo Step with Smoothing Filter)
                self.sim.step_simulation(current_time=time.time())

                # 4. Check for Mission Completion
                if self.navigator.mission_complete and (t_sim - self.navigator.last_s2_time > 1.0):
                    with self._lock:
                        self.mission_status = "completed"
                        self.is_running = False
                    self._record_event(
                        sim_time=t_sim,
                        event_type="MISSION_SUCCESS",
                        reason=f"Target '{self.target_mission.upper()}' reached successfully!",
                        diagnosis="Station-keeping precision hold confirmed.",
                    )
                    break

                # 5. Capture Telemetry Snapshot (Throttled to ~30 Hz sampling)
                if len(self.telemetry_history) == 0 or (t_sim - self.telemetry_history[-1]["time"]) >= 0.033:
                    snap = self.get_telemetry_snapshot()
                    with self._lock:
                        self.telemetry_history.append(snap)
                        self.latest_telemetry = snap

                # 6. Real-Time Execution Throttling
                # Ensures 500 physics cycles per simulated second run at 1x real-time rate
                target_wall_elapsed = float(self.sim.data.time) - sim_start_t
                actual_wall_elapsed = time.time() - wall_start
                sleep_needed = target_wall_elapsed - actual_wall_elapsed
                if sleep_needed > 0.001:
                    time.sleep(sleep_needed)
        finally:
            # Ensure any OpenGL context created on this thread is safely freed before the thread terminates
            try:
                self.navigator.close()
            except Exception:
                pass

    def _record_event(self, sim_time: float, event_type: str, reason: str, diagnosis: str) -> None:
        """Records an event in the sliding-window log for dashboard rendering."""
        event = {
            "time": round(sim_time, 2),
            "type": event_type,
            "reason": reason,
            "diagnosis": diagnosis,
            "phase": self.navigator.phase,
        }
        with self._lock:
            self.vlm_events.appendleft(event)

    # ==========================================================================
    # TELEMETRY SNAPSHOT COMPILATION
    # ==========================================================================

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Gathers comprehensive sensory, controller, and performance metrics."""
        pos = self.sim.data.qpos[0:3].copy()
        vel = self.sim.data.qvel[0:3].copy()
        quat = self.sim.data.qpos[3:7].copy()
        omega = self.sim.data.sensordata[0:3].copy()

        # Euler angles from quaternion [w, x, y, z]
        w, qx, qy, qz = quat
        roll_deg = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
        yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))))
        tilt_deg = float(np.hypot(roll_deg, pitch_deg))

        # L1 Adaptive Disturbance Observer
        sigma_f = self.sim.controller.l1_state.force_disturbance_estimate
        sigma_f_mujoco = self.sim.controller._T @ sigma_f
        force_norm = float(np.linalg.norm(sigma_f_mujoco))

        # Motor commands
        motor_ctrl = [float(round(m, 2)) for m in self.sim.data.ctrl[:4]]

        # Target setpoint & smoothed reference
        smoothed_target = getattr(self.sim.controller, "smoothed_target_pos", self.sim.controller.target_pos)
        pos_err = float(np.linalg.norm(smoothed_target - pos))

        # Distance to commanded mission waypoint
        active_patch = self.sim.bridge.get_active_patch()
        waypoint_tgt = [0.0, 0.0, 1.0]
        if active_patch and active_patch.waypoint_command and active_patch.waypoint_command.target:
            waypoint_tgt = [float(x) for x in active_patch.waypoint_command.target[:3]]
        dist_to_waypoint = float(np.linalg.norm(np.array(waypoint_tgt) - pos))

        # Performance metric tracking
        self.peak_tilt_deg = max(self.peak_tilt_deg, tilt_deg)
        self.peak_l1_force_N = max(self.peak_l1_force_N, force_norm)
        self.tracking_errors.append(pos_err)
        if len(self.tracking_errors) > 500:
            self.tracking_errors.pop(0)
        mean_err = float(np.mean(self.tracking_errors)) if self.tracking_errors else 0.0

        # Feedforward and trim values
        ff_force = [0.0, 0.0, 0.0]
        motor_trim = [1.0, 1.0, 1.0, 1.0]
        max_tilt_clamp = 20.0
        diagnosis_str = "Nominal flight"
        if active_patch:
            if active_patch.feedforward_force_offset:
                ff_force = [float(round(x, 2)) for x in active_patch.feedforward_force_offset]
            if active_patch.motor_trim_scale:
                motor_trim = [float(round(x, 2)) for x in active_patch.motor_trim_scale]
            if active_patch.safety_envelope and active_patch.safety_envelope.max_tilt_deg:
                max_tilt_clamp = float(active_patch.safety_envelope.max_tilt_deg)
            if active_patch.diagnosis:
                diagnosis_str = active_patch.diagnosis

        return {
            "time": float(round(self.sim.data.time, 2)),
            "mission": self.target_mission,
            "status": self.mission_status,
            "is_running": self.is_running,
            "mission_complete": bool(self.navigator.mission_complete),
            "phase": self.navigator.phase,
            "pos": [float(round(p, 3)) for p in pos],
            "vel": [float(round(v, 3)) for v in vel],
            "speed": float(round(float(np.linalg.norm(vel)), 2)),
            "altitude": float(round(pos[2], 3)),
            "attitude": {
                "roll": float(round(roll_deg, 2)),
                "pitch": float(round(pitch_deg, 2)),
                "yaw": float(round(yaw_deg, 2)),
                "tilt": float(round(tilt_deg, 2)),
            },
            "gyro": [float(round(w_axis, 3)) for w_axis in omega],
            "gyro_rms": float(round(float(np.linalg.norm(omega)), 3)),
            "waypoint": [float(round(w_val, 3)) for w_val in waypoint_tgt],
            "dist_to_waypoint": float(round(dist_to_waypoint, 3)),
            "smoothed_target": [float(round(st, 3)) for st in smoothed_target],
            "pos_error": float(round(pos_err, 3)),
            "mean_pos_error": float(round(mean_err, 3)),
            "l1_force": [float(round(f, 2)) for f in sigma_f_mujoco],
            "l1_force_norm": float(round(force_norm, 2)),
            "motor_thrusts": motor_ctrl,
            "motor_trim": motor_trim,
            "feedforward_force": ff_force,
            "max_tilt_clamp": max_tilt_clamp,
            "diagnosis": diagnosis_str,
            "use_vlm": self.use_vlm,
            "vlm_model": self.vlm_model,
            "vlm_call_count": getattr(self.navigator, "_vlm_call_count", 0),
            "chaos_mode": self.chaos_mode,
            "perf": {
                "peak_tilt": float(round(self.peak_tilt_deg, 1)),
                "peak_l1_force": float(round(self.peak_l1_force_N, 2)),
                "mean_tracking_error": float(round(mean_err, 3)),
                "max_duration": self.max_duration,
            },
        }

    # ==========================================================================
    # OFFSCREEN FPV CAMERA RENDERING & HUD OVERLAY
    # ==========================================================================

    def _render_fpv_frame(self, enable_hud: bool = True) -> bytes:
        """Renders single FPV frame with calibrated HUD overlay (runs on dedicated pool)."""
        with self._lock:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.sim.model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
            if self._hud is None:
                self._hud = HUDProjector(camera_name="fpv", width=FRAME_WIDTH, height=FRAME_HEIGHT)
            if self._render_data is None:
                self._render_data = mujoco.MjData(self.sim.model)

            np.copyto(self._render_data.qpos, self.sim.data.qpos)
            np.copyto(self._render_data.qvel, self.sim.data.qvel)
            mujoco.mj_forward(self.sim.model, self._render_data)

            self._renderer.update_scene(self._render_data, camera="fpv")
            raw_rgb = self._renderer.render()

            if enable_hud:
                try:
                    cam = self._hud.extract_camera_calibration(self.sim.model, self._render_data)
                    d_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "x2")
                    drone_pos = self._render_data.xpos[d_id] if d_id >= 0 else None
                    drone_vel = self._render_data.cvel[d_id][3:6] if d_id >= 0 else None

                    base_img = Image.fromarray(raw_rgb)
                    overlaid = self._hud.overlay_hud(
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

            buf = io.BytesIO()
            Image.fromarray(raw_rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
            return buf.getvalue()

    async def get_fpv_frame_async(self, enable_hud: bool = True) -> bytes:
        """Asynchronously dispatches frame rendering to the dedicated OpenGL thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._render_pool, self._render_fpv_frame, enable_hud)

    async def frame_stream_generator(self, request, enable_hud: bool = True) -> AsyncIterator[bytes]:
        """Yields MJPEG stream packets at STREAM_FPS (~15 FPS)."""
        interval = 1.0 / STREAM_FPS
        while not self.shutdown_event.is_set():
            if await request.is_disconnected():
                return
            t0 = time.time()
            try:
                jpeg_bytes = await self.get_fpv_frame_async(enable_hud=enable_hud)
            except asyncio.CancelledError:
                return
            except Exception:
                await asyncio.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
            elapsed = time.time() - t0
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

    # ==========================================================================
    # BENCHMARK SUITE EVALS RUNNER
    # ==========================================================================

    def run_suite_evals(self) -> List[Dict[str, Any]]:
        """Executes full navigation test suite and tabulates scorecard."""
        scenarios = [
            {"id": "SD-01", "name": "Nominal Pad Alpha Hover & Alignment", "target": "alpha", "desc": "Approach and hover centered over Pad Alpha at X=5.5m"},
            {"id": "SD-08", "name": "Runway Gate: Precision Hover Between Poles", "target": "runway_poles", "desc": "Station-keeping centered between pylon gate at X=3.5m, Y=±0.9m"},
            {"id": "SD-02", "name": "Delta: Elevated Caution Disc Platform", "target": "delta", "desc": "Overfly Alpha and achieve elevated hover atop Delta disc at Z=1.10m"},
            {"id": "SD-11", "name": "Left Window: +90° Saccade & Aperture Ingress", "target": "window", "desc": "Active perception turn and safely traverse window clearance frame"},
            {"id": "SD-13", "name": "Right Arch: -90° Saccade & Aperture Ingress", "target": "arch", "desc": "Active perception turn and safely traverse right arch opening"},
        ]

        results = []
        for sc in scenarios:
            tgt = sc["target"]
            # Run headless test with duration
            dur = 22.0 if tgt in ("window", "arch") else 14.0
            drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
            sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=drone_cfg)
            nav = AutonomousTargetNavigator(sim=sim, target_mission=tgt, use_vlm=False)
            success = nav.run_headless(max_duration=dur)

            # Extract metrics
            final_pos = sim.data.qpos[0:3]
            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "target": tgt,
                "desc": sc["desc"],
                "status": "PASS" if success else "FAIL",
                "final_pos": [round(float(p), 2) for p in final_pos],
                "duration": round(float(sim.data.time), 2),
                "phase": nav.phase,
            })

        with self._lock:
            self.suite_results = results
        return results

    def close(self) -> None:
        """Releases renderer and worker pool resources."""
        self.shutdown_event.set()
        self.is_running = False
        if self._sim_thread and self._sim_thread.is_alive() and threading.current_thread() != self._sim_thread:
            try:
                self._sim_thread.join(timeout=1.0)
            except Exception:
                pass

        def _do_close():
            with self._lock:
                if self._renderer is not None:
                    try:
                        self._renderer.close()
                    except Exception:
                        pass
                    self._renderer = None
                    self._render_data = None
                    self._hud = None

        try:
            fut = self._render_pool.submit(_do_close)
            fut.result(timeout=1.0)
        except Exception:
            pass
        try:
            self._render_pool.shutdown(wait=True)
        except Exception:
            pass


# Global singleton instance
nav_coordinator = NavigationSimulationCoordinator()
