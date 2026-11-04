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
from runtime.brain.intent_parser import ARENA_LANDMARKS, NaturalLanguageIntent
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
            self.sim_lock = threading.RLock()  # Synchronizes physics steps, resets, and viewer.sync
            self.shutdown_event = threading.Event()

            # Single Drone Simulator and Navigator instances
            self.drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
            self.sim: DroneSimulator = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=self.drone_cfg)
            self.target_mission: str = "alpha"
            self.active_semantic_label: str = "Orange Landing Pad"
            self.active_instruction: str = ""
            self.last_parsed_intent: Optional[Dict[str, Any]] = None
            self.use_vlm: bool = False
            self.vlm_model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct (Modal A100 GPU)"
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
            self.camera_mode: str = "free"  # "free" (manual cursor rotation/orbit) | "drone_fixed" (heading-aligned TPP)

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
        model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct (Modal A100 GPU)",
        chaos: Optional[str] = None,
        duration: Optional[float] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Configures and launches autonomous flight mission.
        
        If the drone is already airborne and reset is False, seamlessly transitions
        the drone in mid-air to the new target without interrupting physics stepping.
        """
        with self._lock:
            target_clean = target.lower().strip()
            if target_clean in ARENA_LANDMARKS:
                self.active_semantic_label = ARENA_LANDMARKS[target_clean]["semantic_label"]

            # Check if drone is currently airborne and we can seamlessly retarget
            is_airborne = (
                self.is_running
                and self._sim_thread is not None
                and self._sim_thread.is_alive()
                and float(self.sim.data.qpos[2]) > 0.4
            )

            if is_airborne and not reset:
                self.target_mission = target_clean
                self.use_vlm = use_vlm
                self.vlm_model = model
                self.chaos_mode = chaos if chaos in ("wind", "motor", "noise") else None
                self.mission_status = "running"
                self.max_duration = float(self.sim.data.time) + (float(duration) if duration else 45.0)
                self.navigator.retarget(self.target_mission)
                self._record_event(
                    sim_time=float(self.sim.data.time),
                    event_type="RETARGET_COMMAND",
                    reason=f"Seamless in-flight transition to '{self.target_mission.upper()}'",
                    diagnosis=f"Flight Phase: {self.navigator.phase}",
                )
                return {
                    "status": "retargeted",
                    "mission": self.target_mission,
                    "phase": self.navigator.phase,
                    "sim_time": float(round(self.sim.data.time, 2)),
                    "use_vlm": self.use_vlm,
                }

            # Stop existing run if active
            self.stop_mission(status="aborted")
            self.shutdown_event.clear()

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
            with self.sim_lock:
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

    def execute_instruction(self, intent: NaturalLanguageIntent) -> Dict[str, Any]:
        """Translates a parsed natural language instruction into airborne navigation execution."""
        with self._lock:
            self.active_instruction = intent.raw_instruction
            self.active_semantic_label = intent.semantic_label
            self.last_parsed_intent = intent.to_dict()

            # Record high-visibility System 2 Intent Event in telemetry log
            t_sim = float(self.sim.data.time)
            self._record_event(
                sim_time=t_sim,
                event_type="SYSTEM2_INTENT",
                reason=f"\"{intent.raw_instruction}\" ➔ {intent.semantic_label}",
                diagnosis=f"{intent.intent_reasoning} | Visual: {intent.visual_features} (Confidence: {int(intent.confidence*100)}%, via Autonomous VLA Parser)",
            )

        # Launch or retarget airborne flight
        res = self.start_mission(
            target=intent.target_action,
            use_vlm=True,
            reset=False,
        )

        return {
            "status": "executed",
            "instruction": intent.raw_instruction,
            "target_action": intent.target_action,
            "semantic_label": intent.semantic_label,
            "visual_features": intent.visual_features,
            "intent_reasoning": intent.intent_reasoning,
            "confidence": intent.confidence,
            "source": intent.source,
            "flight_status": res.get("status"),
            "sim_time": float(round(self.sim.data.time, 2)),
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
            with self.sim_lock:
                self.sim.chaos.clear_all()
                self.chaos_mode = None
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

    def apply_chaos_preset(self, preset_id: str, toggle: bool = False, additive: bool = False) -> Dict[str, Any]:
        """Applies or toggles a named chaos preset ('1'..'5' or '0') using ChaosManager."""
        with self._lock:
            pid = str(preset_id).strip()
            if pid in ("0", "clean", "none"):
                return self.clear_all_chaos()
            elif toggle:
                res = self.sim.chaos.toggle_preset(pid)
                active_keys = self.sim.chaos.get_active_preset_keys()
                self.chaos_mode = active_keys[0] if active_keys else None
                msg = f"Preset '{pid}' {res.get('action', 'updated')}"
            else:
                msg = self.sim.chaos.apply_preset(pid, additive=additive)
                self.chaos_mode = pid

            return {
                "status": "ok",
                "message": msg,
                "preset_id": pid,
                "chaos_mode": self.chaos_mode,
                "active_presets": self.sim.chaos.get_active_preset_keys(),
                "summary": self.sim.chaos.get_chaos_summary(),
            }

    def trigger_wind_gust(
        self,
        force: float = 6.0,
        direction: Optional[List[float]] = None,
        duration_s: float = 0.8,
    ) -> Dict[str, Any]:
        """Triggers an instantaneous half-sine crosswind gust."""
        with self._lock:
            sim_t = float(self.sim.data.time)
            self.sim.chaos.trigger_wind_gust(
                force=force,
                direction=direction or [1.0, 0.0, 0.0],
                duration_s=duration_s,
                sim_time=sim_t,
            )
            return {"status": "ok", "message": f"Crosswind gust triggered ({force:.1f}N for {duration_s:.1f}s)"}

    def toggle_camera_mode(self, mode: Optional[str] = None) -> str:
        """Toggles or sets camera view mode between 'free' and 'drone_fixed' [Cmd+F]."""
        with self._lock:
            if mode in ("free", "drone_fixed"):
                self.camera_mode = mode
            else:
                self.camera_mode = "drone_fixed" if self.camera_mode == "free" else "free"
            current = self.camera_mode

        try:
            t_sim = float(self.sim.data.time)
        except Exception:
            t_sim = 0.0

        mode_title = "Free View (Cursor Orbit Enabled)" if current == "free" else "Drone Fixed View (Heading-Locked TPP)"
        self._record_event(
            sim_time=t_sim,
            event_type="CAMERA_MODE_CHANGED",
            reason=f"Viewport camera mode set to {current.upper()} [Cmd+F]",
            diagnosis=f"Camera mode: {mode_title}",
        )
        print(f"🎥 [VIEWPORT CAMERA] Viewport camera mode: {current.upper()} [Cmd+F]")
        return current

    def clear_all_chaos(self) -> Dict[str, Any]:
        """Clears all active failure modes back to pristine nominal flight."""
        with self._lock:
            self.sim.chaos.clear_all()
            self.chaos_mode = None
            return {
                "status": "ok",
                "message": "All chaos cleared; pristine nominal flight restored",
                "chaos_mode": None,
                "active_presets": [],
                "summary": self.sim.chaos.get_chaos_summary(),
            }

    def set_chaos(self, chaos_type: Optional[str], active: bool) -> Dict[str, Any]:
        """Backward-compatible disturbance toggle (maps to presets 1..5, 0)."""
        with self._lock:
            if not active or chaos_type in (None, "none", "clean", "0"):
                return self.clear_all_chaos()

            mapping = {
                "wind": "2",
                "motor": "1",
                "payload": "3",
                "cargo": "3",
                "sensor": "4",
                "noise": "4",
                "compound": "5",
                "stress": "5",
            }
            preset_key = mapping.get(str(chaos_type).lower(), str(chaos_type))
            res = self.apply_chaos_preset(preset_key, toggle=False, additive=False)
            self.chaos_mode = chaos_type
            res["chaos_mode"] = chaos_type
            return res

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
                    # If holding station at target or in grand tour, extend time budget to hover indefinitely
                    if self.navigator.mission_complete or self.mission_status in ("holding", "completed"):
                        self.max_duration = t_sim + 60.0
                    else:
                        with self._lock:
                            self.mission_status = "timeout"
                        self._record_event(
                            sim_time=t_sim,
                            event_type="MISSION_TIMEOUT",
                            reason=f"Mission '{self.target_mission.upper()}' duration budget reached",
                            diagnosis="Holding station at current position",
                        )
                        self.max_duration = t_sim + 60.0

                # 1. System 2 Cognitive Cycle (2 Hz or event-driven)
                entry = self.navigator.step_cognitive_cycle(t_sim)
                if entry and entry.get("vlm_invoked"):
                    self._record_event(
                        sim_time=t_sim,
                        event_type="SYSTEM2_COGNITIVE_DISPATCH",
                        reason=f"[Modal A100 GPU] {entry.get('trigger_reason', 'Cognitive trigger')}",
                        diagnosis=entry.get("diagnosis", ""),
                    )
                if entry and entry.get("vlm_completed"):
                    self._record_event(
                        sim_time=t_sim,
                        event_type="SYSTEM2_PATCH_APPLIED",
                        reason="Injected Meta Llama 3.2 Vision control patch into System 1 reflex loop",
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
                with self.sim_lock:
                    self.sim.step_simulation(current_time=time.time())

                # 4. Check for Mission Completion (Continues holding in air, does not stop simulation)
                if self.navigator.mission_complete:
                    if self.mission_status not in ("holding", "completed"):
                        with self._lock:
                            self.mission_status = "holding"
                        self._record_event(
                            sim_time=t_sim,
                            event_type="MISSION_SUCCESS",
                            reason=f"Target '{self.target_mission.upper()}' reached successfully!",
                            diagnosis="Station-keeping precision hold confirmed (awaiting next waypoint)",
                        )

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
        with self.sim_lock:
            pos = self.sim.data.qpos[0:3].copy()
            vel = self.sim.data.qvel[0:3].copy()
            quat = self.sim.data.qpos[3:7].copy()
            omega = self.sim.data.sensordata[0:3].copy()
            motor_ctrl = [float(round(m, 2)) for m in self.sim.data.ctrl[:4]]
            sim_time_val = float(round(self.sim.data.time, 2))

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
            "time": sim_time_val,
            "mission": self.target_mission,
            "semantic_target": self.active_semantic_label,
            "active_instruction": self.active_instruction,
            "last_intent": self.last_parsed_intent,
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
            "cmd_target_pos": [float(round(st, 2)) for st in smoothed_target],
            "cmd_yaw_deg": float(round(float(np.degrees(getattr(self.sim.controller, "smoothed_yaw_rad", self.sim.controller.target_yaw_rad))), 1)),
            "cmd_max_vel": float(round(getattr(active_patch.waypoint_command, "max_velocity", 1.0), 2)) if (active_patch and active_patch.waypoint_command) else 1.0,
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
            "vla_latency_ms": getattr(self.navigator, "_last_vlm_latency_ms", 242),
            "camera_mode": self.camera_mode,
            "chaos_mode": self.chaos_mode,
            "active_presets": self.sim.chaos.get_active_preset_keys(),
            "chaos": self.sim.chaos.get_chaos_summary(),
            "grand_tour": {
                "active": getattr(self.navigator, "grand_tour_mode", False),
                "step": getattr(self.navigator, "grand_tour_step_idx", 0),
                "total": len(getattr(self.navigator, "grand_tour_queue", [])),
                "target": self.target_mission,
            },
            "events": list(self.vlm_events)[:10],
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

            with self.sim_lock:
                np.copyto(self._render_data.qpos, self.sim.data.qpos)
                np.copyto(self._render_data.qvel, self.sim.data.qvel)
            mujoco.mj_forward(self.sim.model, self._render_data)

            self._renderer.update_scene(self._render_data, camera="fpv")
            raw_rgb = self._renderer.render()

            if enable_hud:
                try:
                    cam = self._hud.get_camera_extrinsics(self.sim.model, self._render_data)
                    d_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "x2")
                    if d_id < 0:
                        d_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
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
                except Exception as hud_err:
                    print(f"⚠️ [HUD OVERLAY ERROR] {hud_err}")

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
