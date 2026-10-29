"""Dual-Drone MuJoCo Physics Simulator & Comparative Testbed.

What does the Dual Simulator do?
---------------------------------
This module allows running two independent drones in the exact same MuJoCo physics
world simultaneously for real-time, side-by-side visual and quantitative comparison:
1. Drone 2 (Left / Cyan Accents): Configurable System 1 & System 2 setup.
2. Drone 1 (Right / Orange Accents): Configurable System 1 & System 2 setup.
3. Synchronized Chaos: Both drones experience the exact same failure events at the
   exact same instant (motor cuts, wind gusts, sensor noise, swinging payload).
4. Auto-Tracking Camera: Keeps both drones framed in the 3D OpenGL viewport.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import numpy as np
import mujoco
import mujoco.viewer

from runtime.sim.config import (
    DUAL_SCENE_PATH,
    SimConfig,
    DroneConfig,
    PIDConfig,
    L1Config,
    ChaosConfig,
    get_system_naming,
)
from runtime.sim.system1_controller import System1Controller
from runtime.sim.chaos_injector import ChaosManager
from runtime.sim.bridge import System1System2Bridge


class DualDroneSimulator:
    """Manages the full lifecycle of a dual-drone comparative MuJoCo simulation.

    Attributes:
        drone1_mode (str): 'pid', 'l1_adaptive', 'pid+system2', or 'l1_adaptive+system2'.
        drone2_mode (str): 'pid', 'l1_adaptive', 'pid+system2', or 'l1_adaptive+system2'.
        controller_1 (System1Controller): Drone 1 reflex controller.
        controller_2 (System1Controller): Drone 2 reflex controller.
        bridge_1 (System1System2Bridge): Drone 1 coordination bridge.
        bridge_2 (System1System2Bridge): Drone 2 coordination bridge.
        worker_1 (Optional[System2Worker]): Drone 1 LLM meta-controller worker.
        worker_2 (Optional[System2Worker]): Drone 2 LLM meta-controller worker.
        chaos (ChaosManager): Shared failure mode injector and hotkey handler.
    """

    SUPPORTED_MODES = ("pid", "l1_adaptive", "pid+system2", "l1_adaptive+system2")

    def __init__(
        self,
        drone1_mode: str = "pid",
        drone2_mode: str = "l1_adaptive+system2",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        scene_path: Optional[Path] = None,
        sim_cfg: Optional[SimConfig] = None,
        drone_cfg: Optional[DroneConfig] = None,
        pid_cfg: Optional[PIDConfig] = None,
        l1_cfg: Optional[L1Config] = None,
        chaos_cfg: Optional[ChaosConfig] = None,
    ) -> None:
        self.drone1_mode = drone1_mode.lower().strip()
        self.drone2_mode = drone2_mode.lower().strip()
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        if self.drone1_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid drone1_mode '{drone1_mode}'. Must be one of: {self.SUPPORTED_MODES}"
            )
        if self.drone2_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid drone2_mode '{drone2_mode}'. Must be one of: {self.SUPPORTED_MODES}"
            )

        self.scene_path = scene_path or DUAL_SCENE_PATH
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Dual scene XML not found at: {self.scene_path}")

        self.sim_cfg = sim_cfg or SimConfig()
        self.drone_cfg = drone_cfg or DroneConfig()
        self.pid_cfg = pid_cfg or PIDConfig()
        self.l1_cfg = l1_cfg or L1Config()
        self.chaos = ChaosManager(cfg=chaos_cfg)

        # Register with server state context for live WebSocket & REST control
        try:
            from runtime.server.state import sim_context
            sim_context.attach_simulator(self)
        except Exception:
            pass

        # ----------------------------------------------------------------------
        # Target Setpoints: Drone 2 (Left: Y = +1.2m), Drone 1 (Right: Y = -1.2m)
        # ----------------------------------------------------------------------
        target_z = self.drone_cfg.target_z
        self.d1_target = np.array([0.0, -1.2, target_z], dtype=np.float32)
        self.d2_target = np.array([0.0, +1.2, target_z], dtype=np.float32)

        self.d1_spawn = np.array([0.0, -1.2, 0.8], dtype=np.float32)
        self.d2_spawn = np.array([0.0, +1.2, 0.8], dtype=np.float32)

        # ----------------------------------------------------------------------
        # Drone 1 Controller & Bridge
        # ----------------------------------------------------------------------
        d1_sys1_mode = "pid" if "pid" in self.drone1_mode else "l1_adaptive"
        self.bridge_1 = System1System2Bridge(
            pos_error_threshold=self.l1_cfg.pos_error_threshold,
            tilt_deg_threshold=self.l1_cfg.tilt_error_threshold_deg,
            gyro_norm_threshold=self.l1_cfg.gyro_rms_threshold,
            force_residual_threshold=self.l1_cfg.force_disturbance_threshold,
        )
        self.controller_1 = System1Controller(
            drone_cfg=self.drone_cfg,
            l1_cfg=self.l1_cfg,
            pid_cfg=self.pid_cfg,
            bridge=self.bridge_1,
            mode=d1_sys1_mode,
            target_pos=self.d1_target,
        )

        # ----------------------------------------------------------------------
        # Drone 2 Controller & Bridge
        # ----------------------------------------------------------------------
        d2_sys1_mode = "pid" if "pid" in self.drone2_mode else "l1_adaptive"
        self.bridge_2 = System1System2Bridge(
            pos_error_threshold=self.l1_cfg.pos_error_threshold,
            tilt_deg_threshold=self.l1_cfg.tilt_error_threshold_deg,
            gyro_norm_threshold=self.l1_cfg.gyro_rms_threshold,
            force_residual_threshold=self.l1_cfg.force_disturbance_threshold,
        )
        self.controller_2 = System1Controller(
            drone_cfg=self.drone_cfg,
            l1_cfg=self.l1_cfg,
            pid_cfg=self.pid_cfg,
            bridge=self.bridge_2,
            mode=d2_sys1_mode,
            target_pos=self.d2_target,
        )

        # ----------------------------------------------------------------------
        # System 2 Workers (Background LLM reasoning threads)
        # ----------------------------------------------------------------------
        self.d1_enable_system2 = "system2" in self.drone1_mode
        self.d2_enable_system2 = "system2" in self.drone2_mode

        self.worker_1 = None
        self.worker_2 = None

        if self.d1_enable_system2 or self.d2_enable_system2:
            from runtime.brain.llm import System2Worker

            if self.d1_enable_system2:
                self.worker_1 = System2Worker(
                    bridge=self.bridge_1,
                    provider=self.llm_provider,
                    model=self.llm_model,
                    drone_name="Drone 1 (Right / Orange)",
                )
            if self.d2_enable_system2:
                self.worker_2 = System2Worker(
                    bridge=self.bridge_2,
                    provider=self.llm_provider,
                    model=self.llm_model,
                    drone_name="Drone 2 (Left / Cyan)",
                )

        # ----------------------------------------------------------------------
        # MuJoCo Engine Initialization
        # ----------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_cfg.physics_timestep

        # Resolve Model Entity IDs
        self.anchor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "camera_anchor"
        )

        # Drone 1 IDs
        self.d1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "d1_x2")
        self.d1_root_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "d1_root")
        self.d1_qpos_adr = self.model.jnt_qposadr[self.d1_root_jnt]
        self.d1_dof_adr = self.model.jnt_dofadr[self.d1_root_jnt]

        self.d1_payload_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "d1_payload")
        self.d1_tether_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "d1_tether_rod")
        self.d1_bob_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "d1_payload_bob")
        self.d1_swing_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "d1_swing_x")
        self.d1_payload_qpos_adr = self.model.jnt_qposadr[self.d1_swing_x]
        self.d1_payload_dof_adr = self.model.jnt_dofadr[self.d1_swing_x]
        self.d1_payload_mass = float(self.model.body_mass[self.d1_payload_id])

        d1_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "d1_body_gyro")
        self.d1_gyro_adr = self.model.sensor_adr[d1_gyro_id]
        self.d1_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"d1_thrust{i}")
            for i in range(1, 5)
        ]

        # Drone 2 IDs
        self.d2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "d2_x2")
        self.d2_root_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "d2_root")
        self.d2_qpos_adr = self.model.jnt_qposadr[self.d2_root_jnt]
        self.d2_dof_adr = self.model.jnt_dofadr[self.d2_root_jnt]

        self.d2_payload_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "d2_payload")
        self.d2_tether_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "d2_tether_rod")
        self.d2_bob_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "d2_payload_bob")
        self.d2_swing_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "d2_swing_x")
        self.d2_payload_qpos_adr = self.model.jnt_qposadr[self.d2_swing_x]
        self.d2_payload_dof_adr = self.model.jnt_dofadr[self.d2_swing_x]
        self.d2_payload_mass = float(self.model.body_mass[self.d2_payload_id])

        d2_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "d2_body_gyro")
        self.d2_gyro_adr = self.model.sensor_adr[d2_gyro_id]
        self.d2_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"d2_thrust{i}")
            for i in range(1, 5)
        ]

        # Initialize physics state
        self.reset_simulation()

    def reset_simulation(self) -> None:
        """Resets both drones to their initial level hover positions."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Drone 1 coordinates
        self.data.qpos[self.d1_qpos_adr : self.d1_qpos_adr + 3] = self.d1_spawn
        self.data.qpos[self.d1_qpos_adr + 3 : self.d1_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.d1_payload_qpos_adr : self.d1_payload_qpos_adr + 2] = 0.0

        # Drone 2 coordinates
        self.data.qpos[self.d2_qpos_adr : self.d2_qpos_adr + 3] = self.d2_spawn
        self.data.qpos[self.d2_qpos_adr + 3 : self.d2_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.d2_payload_qpos_adr : self.d2_payload_qpos_adr + 2] = 0.0

        # Zero velocities and forces
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        # Reset controllers & bridges
        self.controller_1.reset_l1_state()
        self.controller_2.reset_l1_state()
        self.bridge_1.reset()
        self.bridge_2.reset()
        if self.worker_1:
            self.worker_1.worker_state = "IDLE"
        if self.worker_2:
            self.worker_2.worker_state = "IDLE"

    def step_simulation(self, current_time: float) -> Tuple[dict, dict]:
        """Executes a single 500 Hz simulation step for both drones.

        Returns:
            Tuple of (telemetry_drone1, telemetry_drone2).
        """
        # 1. Reset applied body forces and generalized forces
        self.data.xfrc_applied[self.d1_body_id, :] = 0.0
        self.data.xfrc_applied[self.d2_body_id, :] = 0.0
        self.data.qfrc_applied[:] = 0.0

        sim_time = float(self.data.time)
        dt = self.sim_cfg.physics_timestep

        # 2. Apply physics perturbations (3D wind + payload aerodynamic drag) for Drone 1 & 2
        wind_f1 = self.chaos.apply_drone_physics(
            drone_idx=1,
            model=self.model,
            data=self.data,
            drone_body_id=self.d1_body_id,
            payload_body_id=self.d1_payload_id,
            tether_geom_id=self.d1_tether_id,
            bob_geom_id=self.d1_bob_id,
            sim_time=sim_time,
            payload_qpos_adr=self.d1_payload_qpos_adr,
            payload_qvel_adr=self.d1_payload_dof_adr,
        )
        wind_f2 = self.chaos.apply_drone_physics(
            drone_idx=2,
            model=self.model,
            data=self.data,
            drone_body_id=self.d2_body_id,
            payload_body_id=self.d2_payload_id,
            tether_geom_id=self.d2_tether_id,
            bob_geom_id=self.d2_bob_id,
            sim_time=sim_time,
            payload_qpos_adr=self.d2_payload_qpos_adr,
            payload_qvel_adr=self.d2_payload_dof_adr,
        )

        # 3. Read sensory state for Drone 1 (with sensor noise/vibrations)
        pos_1 = self.data.qpos[self.d1_qpos_adr : self.d1_qpos_adr + 3].copy()
        vel_1 = self.data.qvel[self.d1_dof_adr : self.d1_dof_adr + 3].copy()
        quat_1 = self.data.qpos[self.d1_qpos_adr + 3 : self.d1_qpos_adr + 7].copy()
        raw_omega_1 = self.data.sensordata[self.d1_gyro_adr : self.d1_gyro_adr + 3].copy()
        omega_1 = self.chaos.corrupt_sensors_for_drone(1, raw_omega_1, sim_time=sim_time)

        # 4. Read sensory state for Drone 2 (with sensor noise/vibrations)
        pos_2 = self.data.qpos[self.d2_qpos_adr : self.d2_qpos_adr + 3].copy()
        vel_2 = self.data.qvel[self.d2_dof_adr : self.d2_dof_adr + 3].copy()
        quat_2 = self.data.qpos[self.d2_qpos_adr + 3 : self.d2_qpos_adr + 7].copy()
        raw_omega_2 = self.data.sensordata[self.d2_gyro_adr : self.d2_gyro_adr + 3].copy()
        omega_2 = self.chaos.corrupt_sensors_for_drone(2, raw_omega_2, sim_time=sim_time)

        # 5. Compute System 1 motor demands independently
        thrust_1, diag_1 = self.controller_1.compute_motor_thrusts(
            pos_1, vel_1, quat_1, omega_1, sim_time=sim_time, dt=dt, target_pos=self.d1_target
        )
        thrust_2, diag_2 = self.controller_2.compute_motor_thrusts(
            pos_2, vel_2, quat_2, omega_2, sim_time=sim_time, dt=dt, target_pos=self.d2_target
        )

        # 6. Apply hardware failure / motor degradation per drone
        actual_thrust_1 = self.chaos.degrade_actuators_for_drone(1, thrust_1, sim_time=sim_time)
        actual_thrust_2 = self.chaos.degrade_actuators_for_drone(2, thrust_2, sim_time=sim_time)

        self.data.ctrl[self.d1_act_ids] = actual_thrust_1
        self.data.ctrl[self.d2_act_ids] = actual_thrust_2

        # 7. Step physics
        mujoco.mj_step(self.model, self.data)

        # 8. Dynamic Camera Tracking Anchor (Midpoint between both drones)
        self.data.xpos[self.anchor_body_id, :] = 0.5 * (pos_1 + pos_2)

        diag_1.update({
            "pos": pos_1,
            "vel": vel_1,
            "quat": quat_1,
            "omega": omega_1,
            "actual_thrust": actual_thrust_1,
            "target": self.d1_target,
            "injected_wind_force": wind_f1,
            "chaos_hud": self.chaos.get_hud_text(1),
        })
        diag_2.update({
            "pos": pos_2,
            "vel": vel_2,
            "quat": quat_2,
            "omega": omega_2,
            "actual_thrust": actual_thrust_2,
            "target": self.d2_target,
            "injected_wind_force": wind_f2,
            "chaos_hud": self.chaos.get_hud_text(2),
        })

        # Record latest telemetry into server context for WebSocket streaming
        try:
            from runtime.server.state import sim_context
            sim_context.record_telemetry(diag_1, diag_2)
        except Exception:
            pass

        return diag_1, diag_2

    def run(self) -> None:
        """Launches the interactive 3D comparative visualizer."""
        self._print_header()
        self.chaos.print_menu()

        # Start System 2 workers if configured
        if self.worker_1:
            try:
                self.worker_1.start()
            except Exception as e:
                print(f"⚠️ [SYSTEM 2 - DRONE 1] Could not start worker ({e}). Running System 1 only.")
        if self.worker_2:
            try:
                self.worker_2.start()
            except Exception as e:
                print(f"⚠️ [SYSTEM 2 - DRONE 2] Could not start worker ({e}). Running System 1 only.")

        try:
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

                # Camera tracks the anchor between both drones (Azimuth 0.0 frames Left=Cyan, Right=Orange side-by-side)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = self.anchor_body_id
                viewer.cam.distance = 4.5
                viewer.cam.elevation = -12.0
                viewer.cam.azimuth = 0.0

                physics_dt = self.sim_cfg.physics_timestep
                render_interval = 1.0 / self.sim_cfg.render_fps
                last_render = time.time()
                last_print = time.time()
                sim_wall_start = time.time()
                sim_start_time = float(self.data.time)

                tele_1 = {}
                tele_2 = {}

                while viewer.is_running():
                    now = time.time()

                    if self.chaos.reset_requested:
                        self.reset_simulation()
                        self.chaos.reset_requested = False
                        sim_wall_start = time.time()
                        sim_start_time = float(self.data.time)

                    # Real-time wall clock synchronization:
                    # Advance 500 Hz physics steps to match elapsed real-world time at 1.0x real speed
                    target_sim_time = sim_start_time + (now - sim_wall_start)
                    substeps = 0
                    while float(self.data.time) < target_sim_time and substeps < 25:
                        tele_1, tele_2 = self.step_simulation(float(self.data.time))
                        substeps += 1

                    # Soft resync if physics lagged behind >60ms (e.g. window moved or OS sleep)
                    if (target_sim_time - float(self.data.time)) > 0.060:
                        sim_wall_start = time.time()
                        sim_start_time = float(self.data.time)

                    # 60 FPS Render Sync & In-Viewer HUD Update
                    if now - last_render >= render_interval:
                        viewer.opt.geomgroup[0:3] = 1

                        # Render in-viewer HUD text overlays (One on TOP, one on BOTTOM to prevent collision on split-screen)
                        d2_name = get_system_naming(self.drone2_mode)["system_short"]
                        d1_name = get_system_naming(self.drone1_mode)["system_short"]
                        hud_d2 = f"Left [Cyan - {d2_name}]: {tele_2.get('chaos_hud', '')}"
                        hud_d1 = f"Right [Orange - {d1_name}]: {tele_1.get('chaos_hud', '')}"
                        viewer.set_texts([
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, hud_d2, ""),
                            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, hud_d1, ""),
                        ])

                        viewer.sync()
                        last_render = now

                    # Telemetry Logging
                    if now - last_print > self.sim_cfg.telemetry_print_interval:
                        if tele_1 and tele_2:
                            self._print_comparative_telemetry(tele_1, tele_2)
                        last_print = now

                    # Yield CPU briefly to OS and background Web server
                    time_ahead = float(self.data.time) - (time.time() - sim_wall_start + sim_start_time)
                    if time_ahead > 0:
                        time.sleep(min(time_ahead, 0.002))
                    else:
                        time.sleep(0.0005)

        finally:
            if self.worker_1:
                self.worker_1.stop()
            if self.worker_2:
                self.worker_2.stop()

    def _print_header(self) -> None:
        """Prints testbed setup info."""
        print("\n" + "═" * 78)
        print("🚁 SKYDIO X2 DUAL-DRONE SIDE-BY-SIDE COMPARATIVE SIMULATION")
        print("═" * 78)
        print(f"🟠 LEFT  (Drone 1, Y=-1.2m):  Mode = [{self._format_mode_label(self.drone1_mode)}]")
        print(f"🔵 RIGHT (Drone 2, Y=+1.2m):  Mode = [{self._format_mode_label(self.drone2_mode)}]")
        print("═" * 78)

    @staticmethod
    def _format_mode_label(mode: str) -> str:
        labels = {
            "pid": "System-1 (Classical PID)",
            "l1_adaptive": "System-1 (L1 Adaptive - QuadLib)",
            "pid+system2": "System-1 (PID) + System-2 LLM Meta-Controller",
            "l1_adaptive+system2": "System-1 (L1 Adaptive) + System-2 LLM Meta-Controller",
        }
        return labels.get(mode, mode)

    def _print_comparative_telemetry(self, t1: dict, t2: dict) -> None:
        """Prints synchronized side-by-side telemetry for both drones."""
        pos1, quat1 = t1["pos"], t1["quat"]
        pos2, quat2 = t2["pos"], t2["quat"]

        roll1 = np.degrees(2.0 * quat1[1])
        roll2 = np.degrees(2.0 * quat2[1])

        err1 = np.linalg.norm(pos1 - t1["target"])
        err2 = np.linalg.norm(pos2 - t2["target"])

        # Drone 1 Patch status
        patch1 = self.bridge_1.get_active_patch()
        p1_str = ""
        if patch1 and patch1.feedforward_force_offset:
            ff = patch1.feedforward_force_offset
            p1_str = f" | 🧠 FF=[{ff[0]:+.1f},{ff[1]:+.1f}]N"
        elif patch1 and patch1.motor_trim_scale:
            p1_str = f" | 🧠 Trim=[{patch1.motor_trim_scale[0]:.2f}]"

        # Drone 2 Patch status
        patch2 = self.bridge_2.get_active_patch()
        p2_str = ""
        if patch2 and patch2.feedforward_force_offset:
            ff = patch2.feedforward_force_offset
            p2_str = f" | 🧠 FF=[{ff[0]:+.1f},{ff[1]:+.1f}]N"
        elif patch2 and patch2.motor_trim_scale:
            p2_str = f" | 🧠 Trim=[{patch2.motor_trim_scale[0]:.2f}]"

        # Disturbance force estimates
        f_dist1 = t1.get("sigma_f_hat")
        f_dist2 = t2.get("sigma_f_hat")
        l1_str1 = f" F_est=[{f_dist1[0]:+.1f}]N" if (f_dist1 is not None and any(abs(x) > 0.1 for x in f_dist1)) else ""
        l1_str2 = f" F_est=[{f_dist2[0]:+.1f}]N" if (f_dist2 is not None and any(abs(x) > 0.1 for x in f_dist2)) else ""

        # Status tags
        stat1 = "💥 CRASH" if pos1[2] < 0.15 else ("🚨 DRIFT" if err1 > 0.40 else "🟢 HOVER")
        stat2 = "💥 CRASH" if pos2[2] < 0.15 else ("🚨 DRIFT" if err2 > 0.40 else "🟢 HOVER")

        d2_naming = get_system_naming(self.drone2_mode)
        d1_naming = get_system_naming(self.drone1_mode)
        d2_label = f"Left [Cyan / D2: {d2_naming['system_short']}]"
        d1_label = f"Right [Orange / D1: {d1_naming['system_short']}]"

        print(
            f"[{stat2}] {d2_label:<35} | Pos=({pos2[0]:+.2f},{pos2[1]:+.2f}) Alt={pos2[2]:.2f}m Err={err2:.2f}m Roll={roll2:+.1f}°{l1_str2}{p2_str}\n"
            f"[{stat1}] {d1_label:<35} | Pos=({pos1[0]:+.2f},{pos1[1]:+.2f}) Alt={pos1[2]:.2f}m Err={err1:.2f}m Roll={roll1:+.1f}°{l1_str1}{p1_str}"
        )
        print("-" * 88)
