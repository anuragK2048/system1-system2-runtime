"""MuJoCo Drone Physics Simulator & Viewer Engine.

What does the Simulator do?
---------------------------
This module ties together:
1. The MuJoCo C++ Physics Engine (stepping equations of motion at 500 Hz).
2. The System 1 PID reflex controller (stabilizing the drone every step).
3. The Chaos Manager (injecting hardware and environmental anomalies).
4. The 3D OpenGL Viewer (rendering at 60 FPS with auto-tracking camera).
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import mujoco
import mujoco.viewer

from runtime.sim.config import (
    SCENE_PATH,
    SimConfig,
    DroneConfig,
    PIDConfig,
    L1Config,
    ChaosConfig,
    System2MasterConfig,
)
from runtime.sim.system1_controller import System1Controller
from runtime.sim.chaos_injector import ChaosManager
from runtime.sim.bridge import System1System2Bridge


class DroneSimulator:
    """Manages the full lifecycle of the Skydio X2 MuJoCo simulation.

    Attributes:
        sim_cfg (SimConfig): Timestep, FPS, and camera tracking parameters.
        drone_cfg (DroneConfig): Physical drone mass, weight, and limits.
        controller (System1Controller): 500 Hz reflex controller (L1 Adaptive or PID).
        chaos (ChaosManager): Failure mode injector and keyboard listener.
        bridge (System1System2Bridge): Async coordination bridge with System 2.
    """

    def __init__(
        self,
        scene_path: Optional[Path] = None,
        sim_cfg: Optional[SimConfig] = None,
        drone_cfg: Optional[DroneConfig] = None,
        pid_cfg: Optional[PIDConfig] = None,
        chaos_cfg: Optional[ChaosConfig] = None,
        l1_cfg: Optional[L1Config] = None,
        bridge: Optional[System1System2Bridge] = None,
        controller_mode: str = "l1_adaptive",
        s2_cfg: Optional[System2MasterConfig] = None,
    ) -> None:
        self.scene_path = scene_path or SCENE_PATH
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.scene_path}")

        self.sim_cfg = sim_cfg or SimConfig()
        self.drone_cfg = drone_cfg or DroneConfig()
        self.s2_cfg = s2_cfg or System2MasterConfig()
        effective_bridge = bridge or System1System2Bridge(
            trigger_cfg=self.s2_cfg.trigger,
            telemetry_cfg=self.s2_cfg.telemetry,
            patch_cfg=self.s2_cfg.patch,
        )
        self.controller = System1Controller(
            drone_cfg=self.drone_cfg,
            l1_cfg=l1_cfg,
            pid_cfg=pid_cfg,
            bridge=effective_bridge,
            mode=controller_mode,
            patch_cfg=self.s2_cfg.patch,
        )
        self.bridge = self.controller.bridge
        self.chaos = ChaosManager(cfg=chaos_cfg)

        # ----------------------------------------------------------------------
        # MuJoCo Engine Initialization
        # ----------------------------------------------------------------------
        # Load compiled MJCF model and dynamic simulation state
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        # Set physics integration timestep (500 Hz = 0.002s)
        self.model.opt.timestep = self.sim_cfg.physics_timestep

        # Resolve entity IDs from model
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "x2")
        self.payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.tether_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tether_rod")
        self.bob_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "payload_bob")
        self.payload_mass = float(self.model.body_mass[self.payload_body_id])

        # Initialize simulation state
        self.reset_simulation()

    def reset_simulation(self) -> None:
        """Resets physics state back to initial level hover at spawn position."""
        # Reset to keyframe 0 (defined in x2.xml)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Reset drone coordinates: x=0, y=0, z=0.8m
        self.data.qpos[0:3] = self.drone_cfg.spawn_pos
        # Identity quaternion [w=1, x=0, y=0, z=0] (level attitude)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Zero pendulum swing angles
        self.data.qpos[7:9] = 0.0

        # Zero all velocities and applied forces
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # Reset L1 adaptive observer state and bridge
        self.controller.reset_l1_state()
        self.bridge.reset()

    def step_simulation(self, current_time: float) -> dict:
        """Executes a single 500 Hz simulation time-step.

        Order of operations:
          1. Apply physical perturbations (cargo gravity, wind gust).
          2. Read noisy sensor telemetry.
          3. Compute System 1 L1 Adaptive / PID motor demands.
          4. Apply actuator hardware degradation (prop failure).
          5. Advance MuJoCo multibody physics (`mj_step`).

        Args:
            current_time: Current wall-clock time (for gust duration checks).

        Returns:
            Dictionary containing state telemetry and controller diagnostics.
        """
        # 0. Record fault on/off transitions for trigger timing correlation
        self.bridge.note_fault_state(self.chaos.active_chaos, float(self.data.time))

        # 1. Update payload attachment / gravity neutralization
        self.chaos.apply_payload_physics(
            self.model,
            self.data,
            self.payload_body_id,
            self.tether_geom_id,
            self.bob_geom_id,
            self.payload_mass,
        )

        # 2. Clear previous body forces and apply active wind gust
        self.data.xfrc_applied[self.drone_body_id, :] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.chaos.apply_wind_gust(self.data, self.drone_body_id, current_time)

        # 3. Read drone sensory state
        pos = self.data.qpos[0:3].copy()
        vel = self.data.qvel[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        raw_omega = self.data.sensordata[0:3].copy()

        # 4. Corrupt gyro rates if sensor noise chaos is active
        omega = self.chaos.corrupt_sensor_data(raw_omega)

        # 5. Compute System 1 motor demands (L1 Adaptive + Geometric loop)
        commanded_thrust, diag = self.controller.compute_motor_thrusts(
            pos,
            vel,
            quat,
            omega,
            sim_time=float(self.data.time),
            dt=self.sim_cfg.physics_timestep,
        )

        # 6. Apply actuator degradation if Motor 1 failure is active
        actual_thrust = self.chaos.degrade_actuators(commanded_thrust)

        # Automatic Crash Disarm: Cut motor thrust to 0 upon ground impact/tumble
        qx, qy = quat[1], quat[2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0))))
        if (pos[2] < 0.12) or (pos[2] < 0.35 and tilt_deg > 50.0) or (tilt_deg > 85.0):
            actual_thrust = np.zeros(4, dtype=np.float32)

        self.data.ctrl[:] = actual_thrust

        # 7. Step physics in MuJoCo
        mujoco.mj_step(self.model, self.data)

        diag.update({
            "pos": pos,
            "vel": vel,
            "quat": quat,
            "omega": omega,
            "actual_thrust": actual_thrust,
        })
        return diag

    def run(self) -> None:
        """Launches the interactive 3D visualizer and starts the physics loop."""
        self.chaos.print_menu()

        # Launch passive viewer on the UI thread
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.chaos.handle_key
        ) as viewer:
            # Hide side panels by default for clean split-screen viewing (press [Tab] in viewer to toggle)
            if hasattr(viewer, "_get_sim"):
                try:
                    sim_obj = viewer._get_sim()
                    if hasattr(sim_obj, "ui0_enable"):
                        sim_obj.ui0_enable = 0
                    if hasattr(sim_obj, "ui1_enable"):
                        sim_obj.ui1_enable = 0
                except Exception:
                    pass
            # Configure auto-tracking camera to keep drone centered
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.drone_body_id
            viewer.cam.distance = self.sim_cfg.cam_distance
            viewer.cam.elevation = self.sim_cfg.cam_elevation
            viewer.cam.azimuth = self.sim_cfg.cam_azimuth

            physics_dt = self.sim_cfg.physics_timestep
            render_interval = 1.0 / self.sim_cfg.render_fps
            last_render = time.time()
            last_print = time.time()

            # Main real-time simulation loop
            while viewer.is_running():
                step_start = time.time()
                now = time.time()

                # Handle explicit user reset request (Shift+5 or key '0')
                if self.chaos.reset_requested:
                    self.reset_simulation()
                    self.chaos.reset_requested = False

                # Step 500 Hz physics
                telemetry = self.step_simulation(now)

                # Sync 3D viewer at 60 FPS
                if now - last_render >= render_interval:
                    # Enforce geom groups 0, 1, 2 visible so the drone never disappears
                    viewer.opt.geomgroup[0:3] = 1
                    viewer.sync()
                    last_render = now

                # Telemetry terminal logging
                if now - last_print > self.sim_cfg.telemetry_print_interval:
                    self._print_telemetry(telemetry)
                    last_print = now

                # Pace loop to real-time physics clock
                time_until_next = physics_dt - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

    def _print_telemetry(self, telemetry: dict) -> None:
        """Formats and outputs telemetry to the console."""
        x, y, z = telemetry["pos"]
        quat = telemetry["quat"]
        roll_deg = np.degrees(2.0 * quat[1])
        pitch_deg = np.degrees(2.0 * quat[2])

        active_list = [k for k, v in self.chaos.active_chaos.items() if v]
        chaos_str = ", ".join(active_list) if active_list else "NONE (Stable)"

        # Check L1 disturbance estimates if present
        f_dist = telemetry.get("sigma_f_hat")
        m_dist = telemetry.get("sigma_M_hat")
        l1_str = ""
        if f_dist is not None and m_dist is not None:
            l1_str = f" | L1 Est: F=[{f_dist[0]:+.1f},{f_dist[1]:+.1f},{f_dist[2]:+.1f}]N"

        # Check Active System 2 Patch
        active_patch = self.bridge.get_active_patch()
        llm_str = ""
        if active_patch:
            parts = []
            if active_patch.feedforward_force_offset:
                ff = active_patch.feedforward_force_offset
                parts.append(f"FF=[{ff[0]:+.1f},{ff[1]:+.1f},{ff[2]:+.1f}]N")
            if active_patch.motor_trim_scale:
                tr = active_patch.motor_trim_scale
                parts.append(f"Trim=[{tr[0]:.2f},{tr[1]:.2f},{tr[2]:.2f},{tr[3]:.2f}]")
            llm_str = f" | 🧠 [LLM: {', '.join(parts)}]"

        # Check Tier 2 Trigger
        if telemetry.get("is_triggered") and telemetry.get("trigger_reasons"):
            trigger_msg = telemetry["trigger_reasons"][0]
            print(f"🚨 [TIER 2 TRIGGER] {trigger_msg}")

        # Detect crash
        if z < 0.12 and any(self.chaos.active_chaos.values()):
            print(
                f"💥 [CRASH] Drone crashed on the ground! "
                f"(Alt={z:.2f}m, Roll={roll_deg:+.1f}°). Press [Shift+5] to reset."
            )
        elif self.chaos.active_chaos["swinging_weight"]:
            sw_x = np.degrees(self.data.qpos[7])
            sw_y = np.degrees(self.data.qpos[8])
            print(
                f"Tele: Alt={z:.2f}m | Roll={roll_deg:+.1f}° | "
                f"Payload Swing=({sw_x:+.1f}°, {sw_y:+.1f}°) 📦 [Attached]{l1_str}{llm_str}"
            )
        else:
            print(
                f"Tele: Alt={z:.2f}m | Pos=({x:+.1f},{y:+.1f}) | "
                f"Roll={roll_deg:+.1f}° | Chaos: [{chaos_str}]{l1_str}{llm_str}"
            )
