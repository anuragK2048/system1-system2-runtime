"""System 1 Reflex Controller (500 Hz L1 Adaptive + Geometric Loop).

What is System 1?
-----------------
In autonomous robotic architectures:
- System 1 is the fast 'spinal reflex' (running at 500 Hz). It computes instantaneous
  motor thrusts to stabilize the drone against extreme disturbances (wind gusts,
  payload pendulum dynamics, motor/propeller failures) with theoretical performance
  guarantees.
- System 2 (the asynchronous LLM foundation model, running at 1-5 Hz) acts as the
  'conscious meta-controller' that monitors System 1 telemetry, diagnoses anomalies,
  and updates System 1's control law via feedforward force offsets, gain multipliers,
  motor trims, and safety envelopes.

Control Theory Foundation:
--------------------------
This controller integrates Mini-QuadLib, implementing:
1. SE(3) Geometric Controller: Non-linear tracking on the SE(3) manifold for large
   attitude excursions and agile quadrotor maneuvers.
2. L1 Adaptive Control Augmentation: Fast adaptation with proven stability guarantees
   that estimates matched and unmatched disturbances (wind, swinging load, actuator
   degradation) and rejects them within milliseconds.
   Reference: 'L1Quad: L1 Adaptive Augmentation of Geometric Control for Agile
   Quadrotors With Performance Guarantees' (IEEE TCST, 2025).
"""

import time
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from runtime.sim.config import DroneConfig, PIDConfig, L1Config
from runtime.sim.bridge import System1System2Bridge
from runtime.schemas.tier2 import System2Patch
from runtime.config.system2_config import PatchApplicationConfig
from runtime.quadlib import (
    Quaternion,
    RotationMatrix,
    QuadrotorState,
    QuadrotorSetpoint,
    L1AdaptiveState,
    geometric_control,
    l1_adaptive_control,
    rotation_matrix_to_quaternion,
)


# F7: default max tilt for the L1 / geometric path. Matches the SafetyEnvelope
# default (25 deg) and is enforced even when no System 2 patch is active.
# The PID path keeps its own 0.15 rad limit (PIDConfig.max_tilt_rad).
L1_DEFAULT_MAX_TILT_DEG: float = 25.0


class System1Controller:
    """Robust 500 Hz Reflex Controller with L1 Adaptive Augmentation for Skydio X2.

    Attributes:
        drone_cfg (DroneConfig): Physical drone constants (hover thrust, mixer matrix).
        l1_cfg (L1Config): Gains for geometric baseline and L1 adaptive observer.
        pid_cfg (PIDConfig): Fallback classical PID gains (if mode='pid').
        bridge (System1System2Bridge): Thread-safe non-blocking telemetry & command bridge.
        mode (str): 'l1_adaptive' (default) or 'pid'.
    """

    def __init__(
        self,
        drone_cfg: Optional[DroneConfig] = None,
        l1_cfg: Optional[L1Config] = None,
        pid_cfg: Optional[PIDConfig] = None,
        bridge: Optional[System1System2Bridge] = None,
        mode: str = "l1_adaptive",
        target_pos: Optional[np.ndarray] = None,
        patch_cfg: Optional[PatchApplicationConfig] = None,
    ) -> None:
        self.drone_cfg = drone_cfg or DroneConfig()
        self.l1_cfg = l1_cfg or L1Config()
        self.pid_cfg = pid_cfg or PIDConfig()
        self.bridge = bridge or System1System2Bridge(
            pos_error_threshold=self.l1_cfg.pos_error_threshold,
            tilt_deg_threshold=self.l1_cfg.tilt_error_threshold_deg,
            gyro_norm_threshold=self.l1_cfg.gyro_rms_threshold,
            force_residual_threshold=self.l1_cfg.force_disturbance_threshold,
        )
        self.patch_cfg = patch_cfg or getattr(self.bridge, "patch_cfg", None) or PatchApplicationConfig()
        self.mode = mode
        self.target_pos = (
            np.array(target_pos, dtype=np.float32)
            if target_pos is not None
            else np.array([0.0, 0.0, self.drone_cfg.target_z], dtype=np.float32)
        )

        # Physical constants
        self.mass = 1.325  # kg
        # F8: inertia about body axes [Ixx, Iyy, Izz] in kg*m^2. No datasheet
        # source is recorded; plausibility basis is a point-mass hand check from
        # the XML geometry (4 x 0.25 kg rotors at x +/-0.14 m, y +/-0.18 m, plus
        # the 0.325 kg body ellipsoid), which reproduces these three numbers
        # within ~13% but in permuted axis order (physics gives Izz largest for
        # a planar rotor layout, here Ixx is largest). The frame convention is
        # therefore UNVERIFIED: do not change these values without a test.
        # See tests/test_system1_l1.py::test_controller_inertia_matches_mujoco,
        # which compares them against MuJoCo's computed body inertia.
        self.inertia = np.array([0.06071, 0.03647, 0.02541], dtype=np.float32)  # [Ixx, Iyy, Izz]

        # Inverse control allocation matrix (4x4)
        self.B_INV = self.drone_cfg.b_inv

        # Coordinate transformation between MuJoCo (Forward=X, Left=Y, Up=Z)
        # and mini_quadlib NED (Forward=X, Right=Y, Down=Z)
        # T = diag(1, -1, -1) -> 180 deg roll rotation
        self._T = np.diag([1.0, -1.0, -1.0])

        # Persistent L1 Adaptive State for online disturbance estimation
        self.l1_state = L1AdaptiveState()

    def reset_l1_state(self) -> None:
        """Resets the L1 adaptive observer state back to zero."""
        self.l1_state = L1AdaptiveState()

    def _resolve_active_patch(
        self,
        sim_time: float,
        explicit_ff: Optional[np.ndarray],
        explicit_trim: Optional[np.ndarray],
        target_pos: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, float, float, np.ndarray, Optional[float]]:
        """Resolves active patch with hot-swap ramp-in apply time, confidence scaling, and safety clamps."""
        active_patch = self.bridge.get_active_patch()

        # Compute hot-swap ramp factor (lambda in [0.0, 1.0])
        ramp = 1.0
        if active_patch is not None and self.patch_cfg.ramp_duration_s > 0.0:
            patch_applied_t = self.bridge.get_patch_applied_time()
            if patch_applied_t is not None:
                elapsed = max(0.0, time.time() - patch_applied_t)
                ramp = min(1.0, elapsed / self.patch_cfg.ramp_duration_s)

        # Scale by patch confidence if enabled
        if active_patch is not None and self.patch_cfg.confidence_scaling:
            ramp *= float(np.clip(active_patch.confidence, 0.0, 1.0))

        # Extract feedforward force offset [Fx, Fy, Fz]
        active_ff = explicit_ff
        if active_ff is None and active_patch and active_patch.feedforward_force_offset:
            raw_ff = np.array(active_patch.feedforward_force_offset, dtype=np.float32)
            active_ff = np.clip(
                raw_ff * ramp,
                self.patch_cfg.feedforward_clamp_min,
                self.patch_cfg.feedforward_clamp_max,
            )

        # Extract motor trim [s1, s2, s3, s4]
        active_trim = explicit_trim
        if active_trim is None and active_patch and active_patch.motor_trim_scale:
            raw_trim = np.array(active_patch.motor_trim_scale, dtype=np.float32)
            active_trim = 1.0 + (raw_trim - 1.0) * ramp
            active_trim = np.clip(
                active_trim,
                self.patch_cfg.motor_trim_clamp_min,
                self.patch_cfg.motor_trim_clamp_max,
            )

        # Extract gain multipliers
        pos_kp_mult = 1.0
        pos_kd_mult = 1.0
        att_kp_mult = 1.0
        att_kd_mult = 1.0
        if active_patch and active_patch.gain_multipliers:
            pos_kp_mult = 1.0 + (active_patch.gain_multipliers.pos_kp - 1.0) * ramp
            pos_kd_mult = 1.0 + (active_patch.gain_multipliers.pos_kd - 1.0) * ramp
            att_kp_mult = 1.0 + (active_patch.gain_multipliers.att_kp - 1.0) * ramp
            att_kd_mult = 1.0 + (active_patch.gain_multipliers.att_kd - 1.0) * ramp

        pos_kp_mult = float(np.clip(pos_kp_mult, self.patch_cfg.gain_multiplier_clamp_min, self.patch_cfg.gain_multiplier_clamp_max))
        pos_kd_mult = float(np.clip(pos_kd_mult, self.patch_cfg.gain_multiplier_clamp_min, self.patch_cfg.gain_multiplier_clamp_max))
        att_kp_mult = float(np.clip(att_kp_mult, self.patch_cfg.gain_multiplier_clamp_min, self.patch_cfg.gain_multiplier_clamp_max))
        att_kd_mult = float(np.clip(att_kd_mult, self.patch_cfg.gain_multiplier_clamp_min, self.patch_cfg.gain_multiplier_clamp_max))

        # Target setpoint in MuJoCo frame
        current_target_pos = (target_pos if target_pos is not None else self.target_pos).copy()
        if active_patch and active_patch.safety_envelope and active_patch.safety_envelope.temp_setpoint_offset:
            offset = np.array(active_patch.safety_envelope.temp_setpoint_offset, dtype=np.float32) * ramp
            current_target_pos += offset

        # Max tilt deg clamp
        max_tilt_deg = None
        if active_patch and active_patch.safety_envelope and active_patch.safety_envelope.max_tilt_deg:
            max_tilt_deg = float(np.clip(
                active_patch.safety_envelope.max_tilt_deg,
                self.patch_cfg.max_tilt_deg_clamp_min,
                self.patch_cfg.max_tilt_deg_clamp_max,
            ))

        return active_ff, active_trim, pos_kp_mult, pos_kd_mult, att_kp_mult, att_kd_mult, current_target_pos, max_tilt_deg

    def compute_motor_thrusts(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        feedforward_force: Optional[np.ndarray] = None,
        motor_trim: Optional[np.ndarray] = None,
        sim_time: float = 0.0,
        dt: float = 0.002,
        target_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Calculates 4 motor thrust commands from current drone state at 500 Hz.

        Args:
            pos: 3D position [x, y, z] in world frame (meters).
            vel: 3D linear velocity [vx, vy, vz] in world frame (m/s).
            quat: Orientation quaternion [w, x, y, z] in world frame.
            omega: Angular velocity [wx, wy, wz] around body axes (rad/s).
            feedforward_force: Optional explicit [Fx, Fy, Fz] force offset.
            motor_trim: Optional explicit [s1, s2, s3, s4] per-motor scaling.
            sim_time: Current simulation timestamp in seconds.
            dt: Simulation timestep (default 0.002s = 500 Hz).
            target_pos: Optional [x, y, z] position setpoint (defaults to self.target_pos).

        Returns:
            Tuple of:
              - motor_thrusts: 4-element array [u1, u2, u3, u4] in Newtons, clamped to [0, 13] N.
              - diagnostics: Comprehensive dictionary with state errors, L1 disturbance estimates,
                             trigger statuses, and control allocation demands.
        """
        if self.mode == "pid":
            return self._compute_pid_thrusts(
                pos, vel, quat, omega, feedforward_force, motor_trim, sim_time, target_pos
            )

        return self._compute_l1_thrusts(
            pos, vel, quat, omega, feedforward_force, motor_trim, sim_time, dt, target_pos
        )

    def _compute_l1_thrusts(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        feedforward_force: Optional[np.ndarray],
        motor_trim: Optional[np.ndarray],
        sim_time: float,
        dt: float,
        target_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Core L1 Adaptive + Geometric controller execution."""
        # ----------------------------------------------------------------------
        # 1. Resolve Active System 2 Patch (with ramp-in apply time, confidence, clamps)
        # ----------------------------------------------------------------------
        (
            active_ff,
            active_trim,
            pos_kp_mult,
            pos_kd_mult,
            att_kp_mult,
            att_kd_mult,
            current_target_pos,
            _,
        ) = self._resolve_active_patch(sim_time, feedforward_force, motor_trim, target_pos)

        # ----------------------------------------------------------------------
        # 2. Coordinate Transformation (MuJoCo ENU/NWU -> Mini-QuadLib NED)
        # ----------------------------------------------------------------------
        pos_ned = self._T @ pos
        vel_ned = self._T @ vel
        omega_ned = self._T @ omega
        desired_pos_ned = self._T @ current_target_pos

        # Compute rotation matrix in MuJoCo from quaternion [w, x, y, z]
        w, qx, qy, qz = quat
        R_mujoco = np.array([
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * w),       2.0 * (qx * qz + qy * w)],
            [2.0 * (qx * qy + qz * w),       1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * w)],
            [2.0 * (qx * qz - qy * w),       2.0 * (qy * qz + qx * w),       1.0 - 2.0 * (qx**2 + qy**2)]
        ], dtype=np.float32)

        # R_ned = T @ R_mujoco @ T
        R_ned = self._T @ R_mujoco @ self._T
        rot_mat_ned = RotationMatrix(R_ned)
        quat_ned = rotation_matrix_to_quaternion(rot_mat_ned)

        # Instantiate Mini-QuadLib state & setpoint
        current_state = QuadrotorState(
            position=pos_ned,
            velocity=vel_ned,
            attitude=quat_ned,
            angular_velocity=omega_ned,
        )
        # Convert feedforward force into feedforward acceleration in NED
        acc_ff_ned = np.zeros(3, dtype=np.float32)
        if active_ff is not None:
            acc_ff_ned = self._T @ (np.array(active_ff, dtype=np.float32) / self.mass)

        effective_pos_gains = self.l1_cfg.pos_gains * np.array([pos_kp_mult, pos_kp_mult, pos_kp_mult], dtype=np.float32)
        effective_vel_gains = self.l1_cfg.vel_gains * np.array([pos_kd_mult, pos_kd_mult, pos_kd_mult], dtype=np.float32)
        effective_att_gains = self.l1_cfg.att_gains * np.array([att_kp_mult, att_kp_mult, att_kp_mult], dtype=np.float32)
        effective_ang_gains = self.l1_cfg.ang_gains * np.array([att_kd_mult, att_kd_mult, att_kd_mult], dtype=np.float32)

        # ----------------------------------------------------------------------
        # F7. L1 tilt limit: clamp the horizontal position demand so the
        # steady-state tilt stays within the safety envelope. In steady state
        # the geometric controller tilts atan(a_horiz / g) with
        # a_horiz ~= kp_xy * err_xy, so |err_xy| <= g * tan(max_tilt) / kp_xy.
        # Uses the patch envelope when present, else the 25 deg default above.
        # (Velocity-error terms can add transient tilt beyond this bound.)
        # ----------------------------------------------------------------------
        max_tilt_deg = L1_DEFAULT_MAX_TILT_DEG
        if (active_patch is not None and active_patch.safety_envelope is not None
                and active_patch.safety_envelope.max_tilt_deg is not None):
            max_tilt_deg = float(active_patch.safety_envelope.max_tilt_deg)
        kp_xy = float(min(effective_pos_gains[0], effective_pos_gains[1]))
        if kp_xy > 1e-6 and max_tilt_deg > 0.0:
            max_err_xy = 9.81 * float(np.tan(np.radians(max_tilt_deg))) / kp_xy
            err_xy = desired_pos_ned[:2] - pos_ned[:2]
            err_xy_norm = float(np.linalg.norm(err_xy))
            if err_xy_norm > max_err_xy:
                desired_pos_ned[:2] = pos_ned[:2] + err_xy * (max_err_xy / err_xy_norm)

        setpoint = QuadrotorSetpoint(position=desired_pos_ned, acceleration=acc_ff_ned)

        # ----------------------------------------------------------------------
        # 3. Geometric Baseline Controller (SE(3) tracking)
        # ----------------------------------------------------------------------
        u_b = geometric_control(
            current_state=current_state,
            desired_setpoint=setpoint,
            position_gains=effective_pos_gains,
            velocity_gains=effective_vel_gains,
            attitude_gains=effective_att_gains,
            angular_velocity_gains=effective_ang_gains,
            mass=self.mass,
            inertia=self.inertia,
        )

        # ----------------------------------------------------------------------
        # 4. L1 Adaptive Controller Augmentation (Disturbance Observer)
        # ----------------------------------------------------------------------
        self.l1_state = l1_adaptive_control(
            previous_state=self.l1_state,
            current_velocity=vel_ned,
            current_angular_velocity=omega_ned,
            current_attitude=quat_ned,
            baseline_control=u_b,
            dt=float(dt),
            vtilde_xy_scale=float(self.l1_cfg.vtilde_xy_scale),
            As_v=float(self.l1_cfg.as_v),
            As_W=float(self.l1_cfg.as_w),
            lpf1_cutoff_freq_force=float(self.l1_cfg.lpf1_cutoff_freq_force),
            lpf1_cutoff_freq_moment=float(self.l1_cfg.lpf1_cutoff_freq_moment),
            lpf2_cutoff_freq_moment=float(self.l1_cfg.lpf2_cutoff_freq_moment),
            mass=self.mass,
            inertia=self.inertia,
        )

        # Total augmented control in NED: u_total = u_b + u_ad
        u_total = self.l1_state.get_total_control()
        u_ad = self.l1_state.adaptive_control

        # Disturbance estimates from L1 observer
        # sigma_f_hat is in NED frame -> map to MuJoCo via T
        sigma_f_hat_ned = self.l1_state.force_disturbance_estimate
        sigma_M_hat_ned = self.l1_state.moment_disturbance_estimate
        sigma_f_hat_mujoco = self._T @ sigma_f_hat_ned
        sigma_M_hat_mujoco = self._T @ sigma_M_hat_ned

        # ----------------------------------------------------------------------
        # 5. Map Demands back to MuJoCo Frame & Invert Allocation
        # ----------------------------------------------------------------------
        # u_total is [F_thrust, M_x, M_y, M_z] in NED
        F_z = float(u_total[0])
        M_mujoco = self._T @ u_total[1:4]

        demands = np.array([F_z, M_mujoco[0], M_mujoco[1], M_mujoco[2]], dtype=np.float32)

        # Convert demands [F_z, tau_x, tau_y, tau_z] into 4 motor thrusts
        motor_thrusts = self.B_INV @ demands

        # Apply per-motor trim scaling (if specified by System 2)
        if active_trim is not None:
            motor_thrusts = motor_thrusts * active_trim

        # Clamp motor thrusts to physical limits [0, 13] N
        motor_thrusts = np.clip(
            motor_thrusts,
            self.drone_cfg.min_thrust,
            self.drone_cfg.max_thrust,
        )

        # ----------------------------------------------------------------------
        # 6. Telemetry & Fast Trigger Notification
        # ----------------------------------------------------------------------
        is_triggered, trigger_reasons = self.bridge.push_telemetry(
            sim_time=sim_time,
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            motor_ctrl=motor_thrusts,
            target_pos=current_target_pos,
            sigma_f_hat=sigma_f_hat_mujoco,
            sigma_M_hat=sigma_M_hat_mujoco,
        )

        # Compute Euler angles for legacy diagnostics
        roll_deg = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
        yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))

        diagnostics: Dict[str, Any] = {
            # Legacy fields for complete backward compatibility
            "F_z": float(F_z),
            "tau_x": float(M_mujoco[0]),
            "tau_y": float(M_mujoco[1]),
            "tau_z": float(M_mujoco[2]),
            "roll_err": float(0.0 - np.radians(roll_deg)),
            "pitch_err": float(0.0 - np.radians(pitch_deg)),
            "target_roll": 0.0,
            "target_pitch": 0.0,
            "l1_max_tilt_deg": float(max_tilt_deg),
            # L1 Adaptive Controller Metrics
            "u_b": [float(x) for x in u_b],
            "u_ad": [float(x) for x in u_ad],
            "u_total": [float(x) for x in u_total],
            "sigma_f_hat": [float(x) for x in sigma_f_hat_mujoco],
            "sigma_M_hat": [float(x) for x in sigma_M_hat_mujoco],
            "euler_deg": [roll_deg, pitch_deg, yaw_deg],
            "is_triggered": is_triggered,
            "trigger_reasons": trigger_reasons,
            "pos_error": [float(x) for x in (current_target_pos - pos)],
        }

        return motor_thrusts, diagnostics

    def _compute_pid_thrusts(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        feedforward_force: Optional[np.ndarray],
        motor_trim: Optional[np.ndarray],
        sim_time: float,
        target_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """PID reflex controller with optional System 2 patch adaptation."""
        x, y, z = pos
        vx, vy, vz = vel

        # ----------------------------------------------------------------------
        # 1. Resolve Active System 2 Patch (with ramp-in apply time, confidence, clamps)
        # ----------------------------------------------------------------------
        (
            active_ff,
            active_trim,
            pos_kp_mult,
            pos_kd_mult,
            att_kp_mult,
            att_kd_mult,
            current_target_pos,
            custom_max_tilt_deg,
        ) = self._resolve_active_patch(sim_time, feedforward_force, motor_trim, target_pos)

        # Compute position error relative to setpoint
        err_x = x - current_target_pos[0]
        err_y = y - current_target_pos[1]
        alt_error = current_target_pos[2] - z

        # Altitude demand (collective thrust F_z)
        F_z = self.drone_cfg.total_hover_weight + (self.pid_cfg.kp_z * pos_kp_mult) * alt_error - (self.pid_cfg.kd_z * pos_kd_mult) * vz
        if active_ff is not None:
            F_z += active_ff[2]

        # Horizontal Position Hold with feedforward tilt bias
        ff_pitch = 0.0
        ff_roll = 0.0
        if active_ff is not None:
            # -Fx requires negative pitch (nose down / forward acceleration)
            ff_pitch = -active_ff[0] / self.drone_cfg.total_hover_weight
            # +Fy requires positive roll (right acceleration in NWU)
            ff_roll = active_ff[1] / self.drone_cfg.total_hover_weight

        max_tilt = np.radians(custom_max_tilt_deg) if custom_max_tilt_deg is not None else self.pid_cfg.max_tilt_rad

        target_pitch = np.clip(
            -(self.pid_cfg.kp_xy * pos_kp_mult) * err_x - (self.pid_cfg.kd_xy * pos_kd_mult) * vx + ff_pitch,
            -max_tilt,
            max_tilt,
        )
        target_roll = np.clip(
            (self.pid_cfg.kp_xy * pos_kp_mult) * err_y + (self.pid_cfg.kd_xy * pos_kd_mult) * vy + ff_roll,
            -max_tilt,
            max_tilt,
        )

        # Attitude & Heading
        roll_err = target_roll - (2.0 * quat[1])
        pitch_err = target_pitch - (2.0 * quat[2])
        yaw_err = 0.0 - (2.0 * quat[3])

        tau_x = (self.pid_cfg.kp_att * att_kp_mult) * roll_err - (self.pid_cfg.kd_att * att_kd_mult) * omega[0]
        tau_y = (self.pid_cfg.kp_att * att_kp_mult) * pitch_err - (self.pid_cfg.kd_att * att_kd_mult) * omega[1]
        tau_z = self.pid_cfg.kp_yaw * yaw_err - self.pid_cfg.kd_yaw * omega[2]

        demands = np.array([F_z, tau_x, tau_y, tau_z])
        motor_thrusts = self.B_INV @ demands

        if active_trim is not None:
            motor_thrusts = motor_thrusts * active_trim

        motor_thrusts = np.clip(
            motor_thrusts,
            self.drone_cfg.min_thrust,
            self.drone_cfg.max_thrust,
        )

        is_triggered, trigger_reasons = self.bridge.push_telemetry(
            sim_time=sim_time,
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            motor_ctrl=motor_thrusts,
            target_pos=current_target_pos,
            sigma_f_hat=np.zeros(3),
            sigma_M_hat=np.zeros(3),
        )

        diagnostics = {
            "F_z": float(F_z),
            "tau_x": float(tau_x),
            "tau_y": float(tau_y),
            "tau_z": float(tau_z),
            "roll_err": float(roll_err),
            "pitch_err": float(pitch_err),
            "target_roll": float(target_roll),
            "target_pitch": float(target_pitch),
            "u_b": [float(F_z), float(tau_x), float(tau_y), float(tau_z)],
            "u_ad": [0.0, 0.0, 0.0, 0.0],
            "sigma_f_hat": [0.0, 0.0, 0.0],
            "sigma_M_hat": [0.0, 0.0, 0.0],
            "is_triggered": is_triggered,
            "trigger_reasons": trigger_reasons,
            "pos_error": [float(val) for val in (current_target_pos - pos)],
        }
        return motor_thrusts, diagnostics
