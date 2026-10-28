"""Thread-Safe Global Simulation Context & State Coordinator.

Bridges the 500 Hz MuJoCo physics loop with the asynchronous FastAPI & WebSocket server:
1. Holds shared references to DualDroneSimulator, ChaosManager, and System1System2Bridges.
2. Maintains sliding-window telemetry ring buffers for real-time charting.
3. Supports dual execution modes:
   - Attached mode: Synchronizes with active 3D OpenGL viewer (`view_dual_drone.py`).
   - Headless mode: Runs physics loop autonomously in a background worker thread.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

from runtime.chaos.manager import ChaosManager
from runtime.chaos.models import ChaosTarget
from runtime.sim.config import get_system_naming, get_drone_orientation

if TYPE_CHECKING:
    from runtime.sim.dual_simulator import DualDroneSimulator


class SimulationContext:
    """Thread-safe singleton coordinating MuJoCo simulation, chaos engine, and web server."""

    _instance: Optional["SimulationContext"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "SimulationContext":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            self.sim: Optional[DualDroneSimulator] = None
            self.chaos: ChaosManager = ChaosManager()

            # Latest telemetry snapshot
            self.latest_frame: Dict[str, Any] = {}
            self.last_update_time: float = 0.0

            # Rolling telemetry history for live graphs (100 samples ~ 3.3 seconds at 30Hz)
            self.telemetry_history: deque = deque(maxlen=120)

            # Bridge trigger & patch log
            self.bridge_logs: deque = deque(maxlen=50)

            # Background headless simulation control
            self._headless_running: bool = False
            self._headless_thread: Optional[threading.Thread] = None

            self._initialized = True

    # ==========================================================================
    # SIMULATOR ATTACHMENT & LIFECYCLE
    # ==========================================================================
    def attach_simulator(self, sim: DualDroneSimulator) -> None:
        """Attaches an active DualDroneSimulator instance to the server context."""
        with self._lock:
            self.sim = sim
            self.chaos = sim.chaos

    def ensure_simulator(
        self,
        drone1_mode: str = "pid",
        drone2_mode: str = "l1_adaptive+system2",
    ) -> DualDroneSimulator:
        """Ensures an active simulator instance exists, creating one if needed."""
        with self._lock:
            if self.sim is None:
                from runtime.sim.dual_simulator import DualDroneSimulator
                self.sim = DualDroneSimulator(
                    drone1_mode=drone1_mode,
                    drone2_mode=drone2_mode,
                    chaos_cfg=self.chaos.cfg,
                )
                # Keep shared chaos reference
                self.sim.chaos = self.chaos
            return self.sim

    def start_headless(
        self,
        drone1_mode: str = "pid",
        drone2_mode: str = "l1_adaptive+system2",
    ) -> None:
        """Starts headless 500 Hz physics simulation loop in a background thread."""
        with self._lock:
            if self._headless_running:
                return

            sim = self.ensure_simulator(drone1_mode, drone2_mode)
            # Start System 2 workers if configured
            if sim.worker_1:
                try:
                    sim.worker_1.start()
                except Exception as e:
                    print(f"⚠️ [SERVER] Worker 1 start failed: {e}")
            if sim.worker_2:
                try:
                    sim.worker_2.start()
                except Exception as e:
                    print(f"⚠️ [SERVER] Worker 2 start failed: {e}")

            self._headless_running = True
            self._headless_thread = threading.Thread(
                target=self._headless_loop,
                name="HeadlessSimLoop",
                daemon=True,
            )
            self._headless_thread.start()
            print("🚀 [SERVER] Headless 500 Hz MuJoCo simulation started.")

    def stop_headless(self) -> None:
        """Stops the headless simulation loop."""
        with self._lock:
            self._headless_running = False
            if self._headless_thread:
                self._headless_thread.join(timeout=1.0)
                self._headless_thread = None

            if self.sim:
                if self.sim.worker_1:
                    self.sim.worker_1.stop()
                if self.sim.worker_2:
                    self.sim.worker_2.stop()
            print("🛑 [SERVER] Headless simulation stopped.")

    def _headless_loop(self) -> None:
        """Background physics execution loop (500 Hz)."""
        dt = 0.002
        while self._headless_running and self.sim:
            t0 = time.time()
            now = time.time()

            if self.chaos.reset_requested:
                self.sim.reset_simulation()
                self.chaos.reset_requested = False

            t1, t2 = self.sim.step_simulation(now)
            self.record_telemetry(t1, t2)

            elapsed = time.time() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ==========================================================================
    # TELEMETRY RECORDING (Called from 500Hz loop)
    # ==========================================================================
    def record_telemetry(self, t1: dict, t2: dict) -> None:
        """Pushes latest physical state and formats a clean telemetry packet."""
        now = time.time()
        # Throttle telemetry processing to ~35 Hz (28ms) so 500 Hz physics loop is 100% lightweight
        if (now - self.last_update_time) < 0.028:
            return

        record_history = True

        pos1, quat1 = t1["pos"], t1["quat"]
        pos2, quat2 = t2["pos"], t2["quat"]

        # Euler angles from quaternions [w, x, y, z]
        roll1 = float(np.degrees(np.arctan2(2.0 * (quat1[0] * quat1[1] + quat1[2] * quat1[3]), 1.0 - 2.0 * (quat1[1]**2 + quat1[2]**2))))
        pitch1 = float(np.degrees(np.arcsin(np.clip(2.0 * (quat1[0] * quat1[2] - quat1[3] * quat1[1]), -1.0, 1.0))))
        yaw1 = float(np.degrees(np.arctan2(2.0 * (quat1[0] * quat1[3] + quat1[1] * quat1[2]), 1.0 - 2.0 * (quat1[2]**2 + quat1[3]**2))))

        roll2 = float(np.degrees(np.arctan2(2.0 * (quat2[0] * quat2[1] + quat2[2] * quat2[3]), 1.0 - 2.0 * (quat2[1]**2 + quat2[2]**2))))
        pitch2 = float(np.degrees(np.arcsin(np.clip(2.0 * (quat2[0] * quat2[2] - quat2[3] * quat2[1]), -1.0, 1.0))))
        yaw2 = float(np.degrees(np.arctan2(2.0 * (quat2[0] * quat2[3] + quat2[1] * quat2[2]), 1.0 - 2.0 * (quat2[2]**2 + quat2[3]**2))))

        err1 = float(np.linalg.norm(pos1 - t1["target"]))
        err2 = float(np.linalg.norm(pos2 - t2["target"]))

        stat1 = "CRASH" if pos1[2] < 0.15 else ("DRIFT" if err1 > 0.40 else "HOVER")
        stat2 = "CRASH" if pos2[2] < 0.15 else ("DRIFT" if err2 > 0.40 else "HOVER")

        # Disturbance estimates
        f_est1 = [float(x) for x in t1.get("sigma_f_hat", [0.0, 0.0, 0.0])]
        f_est2 = [float(x) for x in t2.get("sigma_f_hat", [0.0, 0.0, 0.0])]

        # Ground truth wind
        f_true1 = [float(x) for x in t1.get("injected_wind_force", [0.0, 0.0, 0.0])]
        f_true2 = [float(x) for x in t2.get("injected_wind_force", [0.0, 0.0, 0.0])]

        # Active patches & patch age timers
        patch1_dict = None
        patch2_dict = None
        p1_age = None
        p2_age = None
        if self.sim:
            p1 = self.sim.bridge_1.get_active_patch()
            if p1:
                patch1_dict = p1.to_dict()
            p1_age = self.sim.bridge_1.get_patch_age_s()

            p2 = self.sim.bridge_2.get_active_patch()
            if p2:
                patch2_dict = p2.to_dict()
            p2_age = self.sim.bridge_2.get_patch_age_s()

        # System 2 Meta-Controller live observability & metrics
        system2_status = None
        if self.sim:
            w = self.sim.worker_2 or self.sim.worker_1
            if w and hasattr(w, "get_status_dict"):
                system2_status = w.get_status_dict()

        if system2_status is None or system2_status.get("last_latency_s") is None:
            fallback = self._get_latest_llm_log_status()
            if fallback:
                if system2_status is None:
                    system2_status = fallback
                else:
                    for k, v in fallback.items():
                        if system2_status.get(k) is None:
                            system2_status[k] = v

        d1_mode = self.sim.drone1_mode if self.sim else "pid"
        d2_mode = self.sim.drone2_mode if self.sim else "l1_adaptive+system2"
        d1_sys = get_system_naming(d1_mode)
        d2_sys = get_system_naming(d2_mode)
        d1_orient = get_drone_orientation(1)
        d2_orient = get_drone_orientation(2)

        packet = {
            "timestamp": now,
            "drone1": {
                "mode": d1_mode,
                "system_name": d1_sys["system_name"],
                "system_short": d1_sys["system_short"],
                "system1_type": d1_sys["system1_type"],
                "system2_enabled": d1_sys["system2_enabled"],
                "screen_side": d1_orient["screen_side"],
                "color_name": d1_orient["color_name"],
                "hex_color": d1_orient["hex_color"],
                "display_title": d1_orient["display_title"],
                "pos": [round(float(x), 3) for x in pos1],
                "vel": [round(float(x), 3) for x in t1["vel"]],
                "rpy_deg": [round(roll1, 1), round(pitch1, 1), round(yaw1, 1)],
                "pos_error_m": round(err1, 3),
                "thrust_actual": [round(float(x), 2) for x in t1["actual_thrust"]],
                "status": stat1,
                "f_dist_estimated": [round(x, 2) for x in f_est1],
                "f_dist_ground_truth": [round(x, 2) for x in f_true1],
                "hud": t1.get("chaos_hud", ""),
                "active_patch": patch1_dict,
                "patch_age_s": round(p1_age, 2) if p1_age is not None else None,
            },
            "drone2": {
                "mode": d2_mode,
                "system_name": d2_sys["system_name"],
                "system_short": d2_sys["system_short"],
                "system1_type": d2_sys["system1_type"],
                "system2_enabled": d2_sys["system2_enabled"],
                "screen_side": d2_orient["screen_side"],
                "color_name": d2_orient["color_name"],
                "hex_color": d2_orient["hex_color"],
                "display_title": d2_orient["display_title"],
                "pos": [round(float(x), 3) for x in pos2],
                "vel": [round(float(x), 3) for x in t2["vel"]],
                "rpy_deg": [round(roll2, 1), round(pitch2, 1), round(yaw2, 1)],
                "pos_error_m": round(err2, 3),
                "thrust_actual": [round(float(x), 2) for x in t2["actual_thrust"]],
                "status": stat2,
                "f_dist_estimated": [round(x, 2) for x in f_est2],
                "f_dist_ground_truth": [round(x, 2) for x in f_true2],
                "hud": t2.get("chaos_hud", ""),
                "active_patch": patch2_dict,
                "patch_age_s": round(p2_age, 2) if p2_age is not None else None,
            },
            "chaos": self.chaos.get_chaos_summary(),
            "system2": system2_status,
        }

        with self._lock:
            self.latest_frame = packet
            self.last_update_time = now
            if record_history:
                self.telemetry_history.append(packet)

    def _get_latest_llm_log_status(self) -> Optional[Dict[str, Any]]:
        """Reads latest entry from logs/llm_calls.jsonl as an ultra-fast cached fallback."""
        now = time.time()
        cached = getattr(self, "_cached_llm_log_status", None)
        cached_t = getattr(self, "_cached_llm_log_time", 0.0)
        if cached is not None and (now - cached_t) < 2.0:
            return cached

        import json
        from pathlib import Path
        from runtime.logging.llm_logger import get_llm_logger
        log_path = Path("logs/llm_calls.jsonl")
        if not log_path.exists():
            return None
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                buffer_size = min(size, 8192)
                f.seek(-buffer_size, 2)
                lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                if not lines:
                    return None
                last_line = lines[-1]
                data = json.loads(last_line)
                ts_unix = float(data.get("timestamp_unix", 0.0))
                duration = float(data.get("duration_s", 0.0))
                model_str = str(data.get("model", ""))
                provider = data.get("metadata", {}).get("provider", "groq")
                if ":" in model_str:
                    parts = model_str.split(":", 1)
                    provider = parts[0]
                    model_name = parts[1]
                else:
                    model_name = model_str

                patch_dict = data.get("parsed_patch")
                result = {
                    "drone_name": "Drone 2 (Right)",
                    "provider": provider,
                    "model": model_name,
                    "worker_state": "IDLE",
                    "in_flight": False,
                    "last_latency_s": round(duration, 3),
                    "last_call_dispatched_at": ts_unix - duration,
                    "last_call_received_at": ts_unix,
                    "call_count": len(lines),
                    "patch_applied_time": ts_unix,
                    "patch_age_s": round(max(0.0, time.time() - ts_unix), 2) if patch_dict else None,
                    "provider_usage": get_llm_logger().get_provider_usage_summary(),
                }
                self._cached_llm_log_status = result
                self._cached_llm_log_time = now
                return result
        except Exception:
            return None


# Global singleton instance
sim_context = SimulationContext()
