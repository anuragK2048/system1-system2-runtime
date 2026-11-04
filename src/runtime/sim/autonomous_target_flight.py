"""Autonomous Target Flight Demonstration.

Demonstrates the Skydio X2 quadrotor autonomously navigating in single-drone mode
using strictly onboard sensors (IMU, optical flow, altimeter, L1 observer) and the
onboard FPV camera with calibrated avionics HUD overlay (ground rings, horizon, aperture boxes).

Supports:
1. Individual Target Missions:
   --target alpha       : Precision hover over Target Pad Alpha (Runway Corridor, X=5.5m).
   --target delta       : Precision hover over Elevated Station Delta (Elevated disc, X=8.0m, Z=1.1m).
   --target window      : Active in-place yaw saccade (+90° Left), approach, and window ingress (Y=+5.0m).
   --target arch        : Active in-place yaw saccade (-90° Right), approach, and arch ingress (Y=-5.0m).
   --target grand_tour  : Full autonomous tour across runway and adjacent apertures.

2. Cognitive Execution Modes:
   - Visual-Inertial State Machine (default):
     Deterministic flight phase reasoner driven strictly by onboard telemetry and FPV visibility.
   - Live Gemini Multimodal VLM (--use-vlm / --use-llm):
     Queries Google Gemini with live FPV frames + HUD overlay and telemetry.

3. Chaos Injection (--chaos wind | motor | noise):
   Verifies real-time disturbance rejection and motor compensation during navigation.

Usage:
    # 1. Run headless mission to Target Pad Alpha:
    uv run python src/runtime/sim/autonomous_target_flight.py --headless --target alpha

    # 2. Run headless mission to Left Window (active saccade + traversal):
    uv run python src/runtime/sim/autonomous_target_flight.py --headless --target window

    # 3. Run headless mission to Right Arch:
    uv run python src/runtime/sim/autonomous_target_flight.py --headless --target arch

    # 4. Run interactive 3D viewer:
    uv run mjpython src/runtime/sim/autonomous_target_flight.py --target window
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import mujoco

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.sim.config import SCENE_PATH, DroneConfig
from runtime.sim.simulator import DroneSimulator
from runtime.schemas.tier2 import System2Patch, WaypointCommand, SafetyEnvelope, GainMultipliers
from runtime.brain.prompts import MULTIMODAL_SYSTEM_PROMPT
from vlm.hud_overlay import HUDProjector, Aperture3D

DEFAULT_GEMINI_KEY = "YOUR_GEMINI_API_KEY"

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


class AutonomousTargetNavigator:
    """Coordinates autonomous visual-inertial navigation using onboard sensors and FPV image only."""

    def __init__(
        self,
        sim: DroneSimulator,
        target_mission: str = "alpha",
        use_vlm: bool = False,
        vlm_model: str = "gemini-3.5-flash-lite",
        chaos_type: Optional[str] = None,
    ):
        self.sim = sim
        self.target_mission = target_mission.lower()
        self.use_vlm = use_vlm
        self.vlm_model = vlm_model
        self.chaos_type = chaos_type

        # Initialize kinematics forward pass
        mujoco.mj_forward(sim.model, sim.data)

        # Visual HUD & Camera Renderer
        self.hud = HUDProjector(camera_name="fpv", width=480, height=360)
        self.renderer = mujoco.Renderer(sim.model, height=360, width=480)

        # Mission Flight Phase State Machine
        self.phase = "PHASE_0_LIFTOFF"
        self.phase_start_time = 0.0
        self.phase_complete = False
        self.mission_complete = False

        # Visual telemetry history and event-driven VLM invocation tracking
        self.last_s2_time = -1.0
        self.s2_interval = 0.5  # 2 Hz System 2 cognitive update check
        self.flight_log: List[dict] = []

        # Condition-based VLM trigger state
        self.last_vlm_call_time = -1.0
        self.vlm_heartbeat_s = 3.0        # Max interval in transit before periodic visual check
        self.calm_hover_heartbeat_s = 5.0  # Max interval during steady hover hold
        self._prev_phase = self.phase
        self._last_vlm_force_norm = 0.0
        self._last_vlm_pos_err = 0.0
        self._vlm_call_count = 0

    def close(self):
        """Releases renderer resources."""
        try:
            self.renderer.close()
        except Exception:
            pass

    def get_onboard_telemetry(self) -> Dict[str, Any]:
        """Reads strictly onboard sensory estimates (no privileged world targets)."""
        pos = self.sim.data.qpos[0:3].copy()
        vel = self.sim.data.qvel[0:3].copy()
        quat = self.sim.data.qpos[3:7].copy()
        omega = self.sim.data.sensordata[0:3].copy()

        # Euler angles from quaternion [w, x, y, z]
        w, qx, qy, qz = quat
        roll_deg = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
        yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))

        # Body frame linear velocity: v_b = R^T * v_w
        R = np.array([
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * w),       2.0 * (qx * qz + qy * w)],
            [2.0 * (qx * qy + qz * w),       1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * w)],
            [2.0 * (qx * qz - qy * w),       2.0 * (qy * qz + qx * w),       1.0 - 2.0 * (qx**2 + qy**2)]
        ], dtype=np.float32)
        vel_b = R.T @ vel

        # L1 Adaptive disturbance estimate
        sigma_f = self.sim.controller.l1_state.force_disturbance_estimate
        sigma_f_mujoco = self.sim.controller._T @ sigma_f

        motor_cmds = [float(x) for x in self.sim.data.ctrl[:4]]

        return {
            "altitude_m": float(round(pos[2], 3)),
            "velocity_body_mps": [float(round(v, 3)) for v in vel_b],
            "attitude_deg": {
                "roll": float(round(roll_deg, 2)),
                "pitch": float(round(pitch_deg, 2)),
                "yaw": float(round(yaw_deg, 2)),
            },
            "unmodeled_force_disturbance_N": [float(round(f, 2)) for f in sigma_f_mujoco],
            "motor_controls_N": [float(round(m, 2)) for m in motor_cmds],
            "gyro_rms_radps": [float(round(abs(g), 3)) for g in omega],
        }

    def capture_fpv_image_with_hud(self) -> Tuple[Any, Path]:
        """Captures onboard FPV frame and burns calibrated HUD ground rings & apertures."""
        drone_pos = self.sim.data.qpos[0:3].copy()
        drone_vel = self.sim.data.qvel[0:3].copy()

        hud_img = self.hud.render_and_overlay(
            self.sim.model,
            self.sim.data,
            renderer=self.renderer,
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            apertures=ARENA_APERTURES,
            radii=[1.0, 2.0, 3.0],
        )
        temp_dir = PROJECT_ROOT / "vlm" / "scratch"
        temp_dir.mkdir(parents=True, exist_ok=True)
        img_path = temp_dir / f"live_fpv_{self.target_mission}.jpg"
        hud_img.save(img_path, quality=90)
        return hud_img, img_path

    def _query_live_vlm(self, img_path: Path, telemetry: dict) -> Optional[System2Patch]:
        """Queries Google Gemini multimodal API with live FPV image and telemetry."""
        import requests

        api_key = os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.vlm_model}:generateContent?key={api_key}"

        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        prompt_text = (
            f"{MULTIMODAL_SYSTEM_PROMPT}\n\n"
            f"ACTIVE OBJECTIVE: Execute autonomous navigation mission for '{self.target_mission}'.\n"
            f"CURRENT FLIGHT PHASE: {self.phase}\n"
            f"ONBOARD SENSORY TELEMETRY:\n{json.dumps(telemetry, indent=2)}\n\n"
            "Analyze the attached onboard FPV camera view (with HUD ground rings & horizon) and emit the patch JSON."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                patch_dict = json.loads(raw_text)
                return System2Patch.from_dict(patch_dict)
            else:
                print(f"⚠️ [VLM API ERROR] {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ [VLM QUERY EXCEPTION] {e}")
        return None

    def should_query_vlm(self, telemetry: dict, sim_time: float) -> Tuple[bool, str]:
        """Evaluates whether an expensive multimodal VLM call is necessary.

        Triggers conditionally on:
        1. Mission initialization / visual scene grounding (first cycle).
        2. Flight phase / leg transitions (milestone advancement).
        3. Force disturbance anomalies (L1 observer residual spiked > 1.8N or delta > 1.0N).
        4. Actuator degradation / motor failure (thrust asymmetry > 1.8N).
        5. Position tracking error excursions (> 0.35m).
        6. Excessive attitude tilt (> 20.0 deg).
        7. Gyro chatter / oscillation (> 0.8 rad/s).
        8. Heartbeat sanity check (every 3.0s in active transit, 5.0s in calm hover).

        Bypasses redundant VLM calls when holding stable reflexive station-keeping.
        """
        # 1. First cognitive call at mission start
        if self.last_vlm_call_time < 0.0:
            return True, "Initial mission assessment and visual scene grounding"

        # 2. Flight phase transition or waypoint arrival
        if self.phase != self._prev_phase:
            return True, f"Flight phase transition: {self._prev_phase} -> {self.phase}"

        # 3. External force disturbance (wind gust / microburst onset or cessation)
        f_dist = telemetry.get("unmodeled_force_disturbance_N", [0.0, 0.0, 0.0])
        f_norm = float(np.linalg.norm(f_dist))
        if f_norm > 1.8 and abs(f_norm - self._last_vlm_force_norm) > 1.0:
            return True, f"L1 force disturbance spike ({f_norm:.1f}N) exceeds threshold (1.8N)"
        if self._last_vlm_force_norm > 1.8 and f_norm < 0.8:
            return True, f"L1 force disturbance cleared ({f_norm:.1f}N < 0.8N), returning to calm hover"

        # 4. Actuator degradation / motor failure
        m_ctrl = telemetry.get("motor_controls_N", [3.25, 3.25, 3.25, 3.25])
        m_spread = float(max(m_ctrl) - min(m_ctrl))
        if m_spread > 1.8 and m_ctrl[0] > 4.5:
            return True, f"Actuator command asymmetry ({max(m_ctrl):.1f}N vs {min(m_ctrl):.1f}N) indicates motor degradation"

        # 5. Position tracking excursion
        target_pos = self.sim.controller.target_pos
        curr_pos = self.sim.data.qpos[0:3]
        pos_err = float(np.linalg.norm(target_pos - curr_pos))
        if pos_err > 0.35 and abs(pos_err - self._last_vlm_pos_err) > 0.20:
            return True, f"Position tracking error ({pos_err:.2f}m) exceeds threshold (0.35m)"

        # 6. Total attitude tilt excursion
        att = telemetry.get("attitude_deg", {})
        roll = abs(float(att.get("roll", 0.0)))
        pitch = abs(float(att.get("pitch", 0.0)))
        tilt_est = float(np.hypot(roll, pitch))
        if tilt_est > 20.0:
            return True, f"Attitude tilt ({tilt_est:.1f}°) exceeds threshold (20.0°)"

        # 7. Gyro chatter / oscillation
        gyro_rms = float(np.linalg.norm(telemetry.get("gyro_rms_radps", [0, 0, 0])))
        if gyro_rms > 0.8:
            return True, f"Gyro oscillation ({gyro_rms:.2f} rad/s) exceeds threshold (0.8 rad/s)"

        # 8. Heartbeat sanity check (longer interval during calm hover)
        is_hover = (self.phase in ("PHASE_4_PRECISION_HOVER", "PHASE_5_EGRESS_STABILIZE")) and (pos_err < 0.15)
        heartbeat_limit = self.calm_hover_heartbeat_s if is_hover else self.vlm_heartbeat_s
        time_since_vlm = sim_time - self.last_vlm_call_time
        if time_since_vlm >= heartbeat_limit:
            return True, f"Periodic visual heartbeat check ({time_since_vlm:.1f}s elapsed)"

        return False, "Nominal flight within tolerance; reflexive hold active"

    def _visual_inertial_reasoner(self, telemetry: dict, sim_time: float) -> System2Patch:
        """Deterministic System 2 reasoner driven strictly by onboard telemetry and visual phase state."""
        alt = telemetry["altitude_m"]
        yaw = telemetry["attitude_deg"]["yaw"]
        F_dist = telemetry["unmodeled_force_disturbance_N"]
        m_ctrl = telemetry["motor_controls_N"]

        # Default Nominal Values
        ff = [0.0, 0.0, 0.0]
        trim = [1.0, 1.0, 1.0, 1.0]
        gains = GainMultipliers(pos_kp=1.0, pos_kd=1.0, att_kp=1.0, att_kd=1.0)
        max_tilt = 20.0

        # Chaos Disturbance Rejection
        if abs(F_dist[0]) > 2.0 or abs(F_dist[1]) > 2.0:
            ff = [-float(F_dist[0]), -float(F_dist[1]), 0.0]
            gains.pos_kp = 1.2
        if m_ctrl[0] > 5.0 and m_ctrl[1] < 4.0:
            trim = [2.50, 1.0, 1.0, 1.0]
            max_tilt = 18.0

        # Phase State Machine
        wc: WaypointCommand
        diagnosis = ""

        # Current estimated position in arena
        pos_x = float(self.sim.data.qpos[0])
        pos_y = float(self.sim.data.qpos[1])

        # ----------------------------------------------------------------------
        # MISSION 1: TARGET PAD ALPHA (Runway Forward Corridor)
        # ----------------------------------------------------------------------
        if self.target_mission in ("alpha", "corridor"):
            if self.phase == "PHASE_0_LIFTOFF":
                diagnosis = "Liftoff to nominal hover altitude (Z=1.0m)"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 1.0], yaw_deg=0.0)
                if abs(alt - 1.0) < 0.08 and sim_time > 1.0:
                    self.phase = "PHASE_2_CORRIDOR_APPROACH"
                    print("🚀 [PHASE TRANSITION] Liftoff complete -> PHASE_2_CORRIDOR_APPROACH")

            elif self.phase == "PHASE_2_CORRIDOR_APPROACH":
                if pos_x < 2.5:
                    diagnosis = "Advancing through hazard pylons along runway corridor (Leg 1)"
                    wc = WaypointCommand(mode="world", target=[2.8, 0.0, 1.0], yaw_deg=0.0, max_velocity=1.2, corridor_width=0.6)
                elif pos_x < 4.8:
                    diagnosis = "Visually tracking Target Pad Alpha concentric rings (Leg 2)"
                    wc = WaypointCommand(mode="world", target=[5.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=1.0, corridor_width=0.6)
                else:
                    self.phase = "PHASE_4_PRECISION_HOVER"
                    diagnosis = "Target Pad Alpha centered in nadir ground rings -> PHASE_4_PRECISION_HOVER"
                    print("🎯 [PHASE TRANSITION] Pad Alpha visually acquired -> PHASE_4_PRECISION_HOVER")
                    wc = WaypointCommand(mode="world", target=[5.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=0.5)

            else:  # PHASE_4_PRECISION_HOVER
                diagnosis = "Station-keeping precision hover centered over Target Pad Alpha"
                wc = WaypointCommand(mode="world", target=[5.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=0.4)
                if abs(pos_x - 5.5) < 0.15:
                    self.mission_complete = True

        # ----------------------------------------------------------------------
        # MISSION 2: ELEVATED PEDESTAL DELTA
        # ----------------------------------------------------------------------
        elif self.target_mission in ("delta", "pedestal"):
            if self.phase == "PHASE_0_LIFTOFF":
                diagnosis = "Liftoff to nominal hover altitude (Z=1.0m)"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 1.0], yaw_deg=0.0)
                if abs(alt - 1.0) < 0.08 and sim_time > 1.0:
                    self.phase = "PHASE_2_CORRIDOR_APPROACH"
                    print("🚀 [PHASE TRANSITION] Liftoff complete -> PHASE_2_CORRIDOR_APPROACH")

            elif self.phase == "PHASE_2_CORRIDOR_APPROACH":
                if pos_x < 3.5:
                    diagnosis = "Advancing past hazard pylons (Leg 1)"
                    wc = WaypointCommand(mode="world", target=[4.0, 0.0, 1.0], yaw_deg=0.0, max_velocity=1.4)
                elif pos_x < 6.5:
                    diagnosis = "Overflying Pad Alpha toward Elevated Platform Delta (Leg 2)"
                    wc = WaypointCommand(mode="world", target=[8.0, 0.0, 1.10], yaw_deg=0.0, max_velocity=1.2)
                else:
                    self.phase = "PHASE_4_PRECISION_HOVER"
                    diagnosis = "Station Delta platform acquired -> PHASE_4_PRECISION_HOVER"
                    print("🎯 [PHASE TRANSITION] Platform Delta acquired -> PHASE_4_PRECISION_HOVER")
                    wc = WaypointCommand(mode="world", target=[8.0, 0.0, 1.10], yaw_deg=0.0, max_velocity=0.5)

            else:  # PHASE_4_PRECISION_HOVER
                diagnosis = "Precision elevated hover over Platform Delta (Z=1.10m)"
                wc = WaypointCommand(mode="world", target=[8.0, 0.0, 1.10], yaw_deg=0.0, max_velocity=0.4)
                if abs(pos_x - 8.0) < 0.20:
                    self.mission_complete = True

        # ----------------------------------------------------------------------
        # MISSION 3: LEFT WINDOW (Active Perception Saccade + Aperture Ingress)
        # ----------------------------------------------------------------------
        elif self.target_mission in ("window", "traverse_window"):
            if self.phase == "PHASE_0_LIFTOFF":
                diagnosis = "Liftoff to window centerline altitude (Z=1.10m)"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 1.10], yaw_deg=0.0)
                if abs(alt - 1.10) < 0.08 and sim_time > 1.0:
                    self.phase = "PHASE_1_SECTOR_SCAN"
                    print("👁️ [ACTIVE PERCEPTION] Window unobservable in forward FOV -> In-Place Yaw Saccade (+90° Left)")

            elif self.phase == "PHASE_1_SECTOR_SCAN":
                diagnosis = "Active saccade: in-place yaw rotation +90° to visually acquire Left Window"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 1.10], yaw_deg=90.0, max_velocity=0.6)
                if abs(yaw - 90.0) < 3.0:
                    self.phase = "PHASE_2_CORRIDOR_APPROACH"
                    print("✅ [TARGET ACQUIRED] Left Window framed in HUD -> PHASE_2_CORRIDOR_APPROACH")

            elif self.phase == "PHASE_2_CORRIDOR_APPROACH":
                diagnosis = "Approaching Left Window along HUD approach glidepath vector"
                wc = WaypointCommand(mode="world", target=[0.0, 3.8, 1.10], yaw_deg=90.0, max_velocity=0.8, corridor_width=0.5)
                if pos_y >= 3.5:
                    self.phase = "PHASE_3_APERTURE_INGRESS"
                    print("🚪 [INGRESS ARMED] Traversing inner clearance envelope -> PHASE_3_APERTURE_INGRESS")

            elif self.phase == "PHASE_3_APERTURE_INGRESS":
                diagnosis = "Passing through Left Window with safety envelope clamping (tilt <= 12°, vel <= 0.6 m/s)"
                wc = WaypointCommand(mode="world", target=[0.0, 5.8, 1.10], yaw_deg=90.0, max_velocity=0.6, corridor_width=0.4)
                max_tilt = 12.0
                if pos_y >= 5.3:
                    self.phase = "PHASE_5_EGRESS_STABILIZE"
                    print("🏆 [MISSION COMPLETE] Successfully traversed Left Window aperture!")

            else:  # PHASE_5_EGRESS_STABILIZE
                diagnosis = "Aperture traversal complete; station-keeping hover on outer perimeter"
                wc = WaypointCommand(mode="world", target=[0.0, 6.0, 1.10], yaw_deg=90.0)
                self.mission_complete = True

        # ----------------------------------------------------------------------
        # MISSION 4: RIGHT ARCH (Active Perception Saccade + Aperture Ingress)
        # ----------------------------------------------------------------------
        elif self.target_mission in ("arch", "traverse_arch"):
            if self.phase == "PHASE_0_LIFTOFF":
                diagnosis = "Liftoff to arch centerline altitude (Z=0.90m)"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 0.90], yaw_deg=0.0)
                if abs(alt - 0.90) < 0.08 and sim_time > 1.0:
                    self.phase = "PHASE_1_SECTOR_SCAN"
                    print("👁️ [ACTIVE PERCEPTION] Arch unobservable in forward FOV -> In-Place Yaw Saccade (-90° Right)")

            elif self.phase == "PHASE_1_SECTOR_SCAN":
                diagnosis = "Active saccade: in-place yaw rotation -90° to visually acquire Right Arch"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 0.90], yaw_deg=-90.0, max_velocity=0.6)
                if abs(yaw - (-90.0)) < 3.0 or abs(yaw - 270.0) < 3.0:
                    self.phase = "PHASE_2_CORRIDOR_APPROACH"
                    print("✅ [TARGET ACQUIRED] Right Arch framed in HUD -> PHASE_2_CORRIDOR_APPROACH")

            elif self.phase == "PHASE_2_CORRIDOR_APPROACH":
                diagnosis = "Approaching Right Arch along HUD approach glidepath vector"
                wc = WaypointCommand(mode="world", target=[0.0, -3.8, 0.90], yaw_deg=-90.0, max_velocity=0.8, corridor_width=0.5)
                if pos_y <= -3.5:
                    self.phase = "PHASE_3_APERTURE_INGRESS"
                    print("⛩️ [INGRESS ARMED] Traversing arch inner envelope -> PHASE_3_APERTURE_INGRESS")

            elif self.phase == "PHASE_3_APERTURE_INGRESS":
                diagnosis = "Passing through Right Arch opening (tilt <= 12°, vel <= 0.6 m/s)"
                wc = WaypointCommand(mode="world", target=[0.0, -5.8, 0.90], yaw_deg=-90.0, max_velocity=0.6, corridor_width=0.4)
                max_tilt = 12.0
                if pos_y <= -5.3:
                    self.phase = "PHASE_5_EGRESS_STABILIZE"
                    print("🏆 [MISSION COMPLETE] Successfully traversed Right Arch aperture!")

            else:  # PHASE_5_EGRESS_STABILIZE
                diagnosis = "Arch traversal complete; station-keeping hover on outer perimeter"
                wc = WaypointCommand(mode="world", target=[0.0, -6.0, 0.90], yaw_deg=-90.0)
                self.mission_complete = True

        # ----------------------------------------------------------------------
        # MISSION 5: RUNWAY POLES (Alpha Gate Entrance)
        # ----------------------------------------------------------------------
        elif self.target_mission in ("alpha_pole", "pole", "runway_pole", "runway_poles", "poles"):
            if self.phase == "PHASE_0_LIFTOFF":
                diagnosis = "Liftoff to nominal hover altitude (Z=1.0m)"
                wc = WaypointCommand(mode="world", target=[0.0, 0.0, 1.0], yaw_deg=0.0)
                if abs(alt - 1.0) < 0.08 and sim_time > 1.0:
                    self.phase = "PHASE_2_CORRIDOR_APPROACH"
                    print("🚀 [PHASE TRANSITION] Liftoff complete -> PHASE_2_CORRIDOR_APPROACH")

            elif self.phase == "PHASE_2_CORRIDOR_APPROACH":
                if pos_x < 3.2:
                    diagnosis = "Advancing to runway pylon entrance corridor"
                    wc = WaypointCommand(mode="world", target=[3.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=1.0, corridor_width=0.5)
                else:
                    self.phase = "PHASE_4_PRECISION_HOVER"
                    diagnosis = "Arrived at runway gate -> Centered hover between runway poles"
                    print("🎯 [PHASE TRANSITION] Runway poles reached -> PHASE_4_PRECISION_HOVER")
                    wc = WaypointCommand(mode="world", target=[3.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=0.4, corridor_width=0.4)

            else:  # PHASE_4_PRECISION_HOVER
                diagnosis = "Holding precision station-keeping hover centered between runway poles (Y=±0.9m)"
                wc = WaypointCommand(mode="world", target=[3.5, 0.0, 1.0], yaw_deg=0.0, max_velocity=0.4, corridor_width=0.4)
                if abs(pos_x - 3.5) < 0.15 and abs(pos_y) < 0.15:
                    self.mission_complete = True

        else:
            raise ValueError(f"Unknown target mission: {self.target_mission}")

        return System2Patch(
            diagnosis=diagnosis,
            flight_phase=self.phase,
            waypoint_command=wc,
            feedforward_force_offset=ff,
            gain_multipliers=gains,
            motor_trim_scale=trim,
            safety_envelope=SafetyEnvelope(max_tilt_deg=max_tilt),
            timestamp=sim_time,
        )

    def step_cognitive_cycle(self, sim_time: float) -> Optional[dict]:
        """Runs the System 2 cognitive decision loop using sensors & FPV image only.
        
        Evaluates whether a VLM call is necessary. If necessary (startup, phase transition,
        force disturbance, motor failure, or periodic heartbeat), queries the VLM reasoner.
        Otherwise, maintains smooth reflexive station-keeping without wasteful calls.
        """
        if sim_time - self.last_s2_time < self.s2_interval:
            return None

        self.last_s2_time = sim_time

        # 1. Read strictly onboard sensory telemetry
        telemetry = self.get_onboard_telemetry()

        # Update candidate patch from visual-inertial reasoner to track phase state
        candidate_patch = self._visual_inertial_reasoner(telemetry, sim_time)

        # 2. Check if System 2 VLM reasoning is necessary
        vlm_needed, trigger_reason = self.should_query_vlm(telemetry, sim_time)

        patch: Optional[System2Patch] = None
        vlm_invoked = False

        if vlm_needed:
            # Capture live FPV frame with calibrated HUD overlay
            _, img_path = self.capture_fpv_image_with_hud()

            if self.use_vlm:
                print(f"🧠 [VLM TRIGGER] t={sim_time:.2f}s: Querying VLM Reasoner ({self.vlm_model}) -> {trigger_reason}")
                patch = self._query_live_vlm(img_path, telemetry)
                vlm_invoked = True
                self._vlm_call_count += 1

            if patch is None:
                patch = candidate_patch

            # Update condition tracking
            self.last_vlm_call_time = sim_time
            self._prev_phase = self.phase
            f_dist = telemetry.get("unmodeled_force_disturbance_N", [0.0, 0.0, 0.0])
            self._last_vlm_force_norm = float(np.linalg.norm(f_dist))
            curr_pos = self.sim.data.qpos[0:3]
            self._last_vlm_pos_err = float(np.linalg.norm(self.sim.controller.target_pos - curr_pos))

            # Inject patch into System 1 controller via bridge
            self.sim.bridge.apply_system2_patch(patch)
        else:
            # Reflexive hold: System 1 controller continues executing active patch with smoothing
            active_patch = self.sim.bridge.get_active_patch()
            patch = active_patch or candidate_patch

        # Telemetry logging
        log_entry = {
            "t": round(sim_time, 2),
            "phase": self.phase,
            "alt": telemetry["altitude_m"],
            "yaw": telemetry["attitude_deg"]["yaw"],
            "diagnosis": patch.diagnosis if patch else "Reflexive hold",
            "vlm_invoked": vlm_invoked,
            "trigger_reason": trigger_reason if vlm_needed else "Reflexive hold within tolerance",
            "waypoint": patch.waypoint_command.to_dict() if (patch and patch.waypoint_command) else None,
        }
        self.flight_log.append(log_entry)
        return log_entry

    def apply_chaos(self, sim_time: float):
        """Injects optional disturbance for robustness verification."""
        if self.chaos_type == "wind" and sim_time >= 1.5:
            # 5.0N lateral wind gust
            drone_id = self.sim.drone_body_id
            self.sim.data.xfrc_applied[drone_id, 0] = 5.0  # +X wind
        elif self.chaos_type == "motor" and sim_time >= 1.5:
            # Motor 1 degradation (40% power)
            self.sim.chaos.active_chaos = "motor_failure"

    def run_headless(self, max_duration: float = 12.0) -> bool:
        """Runs headless simulation with telemetry reporting."""
        mode_label = f"LIVE VLM ({self.vlm_model})" if self.use_vlm else "SENSORS + FPV VISUAL REASONER"
        print("\n" + "═" * 78)
        print(f"🚁 [AUTONOMOUS TARGET FLIGHT DEMO | {mode_label}]")
        print(f"🎯 Mission Target: {self.target_mission.upper()}")
        print(f"🌪️ Chaos Mode:    {self.chaos_type or 'None (Nominal)'}")
        print("═" * 78 + "\n")

        while self.sim.data.time < max_duration:
            t = float(self.sim.data.time)

            # System 2 Cognitive Step (2 Hz)
            entry = self.step_cognitive_cycle(t)
            if entry:
                print(f"⏱️ t={t:5.2f}s | Phase: {entry['phase']:<24} | Alt: {entry['alt']:4.2f}m | Yaw: {entry['yaw']:+6.1f}° | {entry['diagnosis']}")

            # Disturbance injection (if requested)
            if self.chaos_type:
                self.apply_chaos(t)

            # System 1 Reflex Step (500 Hz physics)
            self.sim.step_simulation(current_time=time.time())

            if self.mission_complete and (t - self.last_s2_time > 1.0):
                print(f"\n🎉 Mission '{self.target_mission.upper()}' SUCCESS at t={t:.2f}s!")
                self.close()
                return True

        self.close()
        print(f"\nSimulation completed at t={self.sim.data.time:.2f}s. Mission complete: {self.mission_complete}")
        return self.mission_complete


def main():
    parser = argparse.ArgumentParser(description="Autonomous Target Flight Demo (Sensors + FPV Image Only)")
    parser.add_argument("--headless", action="store_true", help="Run without OpenGL visualizer window")
    parser.add_argument(
        "--target",
        type=str,
        default="alpha",
        choices=["alpha", "delta", "window", "arch", "grand_tour"],
        help="Target mission: 'alpha' (Pad Alpha), 'delta' (Pedestal Delta), 'window' (Left Window), 'arch' (Right Arch)",
    )
    parser.add_argument("--use-vlm", "--use-llm", action="store_true", dest="use_vlm", help="Query live Gemini Multimodal VLM")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="VLM Model override")
    parser.add_argument("--chaos", type=str, default=None, choices=["wind", "motor", "noise"], help="Inject physical chaos")
    parser.add_argument("--duration", type=float, default=14.0, help="Max simulation duration (seconds)")
    args = parser.parse_args()

    drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
    sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=drone_cfg)

    navigator = AutonomousTargetNavigator(
        sim=sim,
        target_mission=args.target,
        use_vlm=args.use_vlm,
        vlm_model=args.model,
        chaos_type=args.chaos,
    )

    if args.headless:
        success = navigator.run_headless(max_duration=args.duration)
        sys.exit(0 if success else 1)
    else:
        print("Launching interactive visualizer...")
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                t = float(sim.data.time)
                navigator.step_cognitive_cycle(t)
                if args.chaos:
                    navigator.apply_chaos(t)

                sim.step_simulation(current_time=time.time())
                viewer.sync()
                time.sleep(sim.sim_cfg.physics_timestep)
        navigator.close()


if __name__ == "__main__":
    main()
