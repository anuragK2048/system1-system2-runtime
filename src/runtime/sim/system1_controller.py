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

from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from runtime.sim.config import DroneConfig, PIDConfig, L1Config
from runtime.sim.bridge import System1System2Bridge
from runtime.schemas.tier2 import System2Patch
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
        self.mode = mode

        # Physical constants
        self.mass = 1.325  # kg
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

        Returns:
            Tuple of:
              - motor_thrusts: 4-element array [u1, u2, u3, u4] in Newtons, clamped to [0, 13] N.
              - diagnostics: Comprehensive dictionary with state errors, L1 disturbance estimates,
                             trigger statuses, and control allocation demands.
        """
        if self.mode == "pid":
            return self._compute_pid_thrusts(
                pos, vel, quat, omega, feedforward_force, motor_trim, sim_time
            )

        return self._compute_l1_thrusts(
            pos, vel, quat, omega, feedforward_force, motor_trim, sim_time, dt
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
    ) -> Tuple[np.ndarray, dict]:
        """Core L1 Adaptive + Geometric controller execution."""
        # ----------------------------------------------------------------------
        # 1. Check for Active System 2 Patch from Async Bridge
        # ----------------------------------------------------------------------
        active_patch = self.bridge.get_active_patch()

        # Extract feedforward force offset [Fx, Fy, Fz]
        active_ff = feedforward_force
        if active_ff is None and active_patch and active_patch.feedforward_force_offset:
            active_ff = np.array(active_patch.feedforward_force_offset, dtype=np.float32)

        # Extract motor trim [s1, s2, s3, s4]
        active_trim = motor_trim
        if active_trim is None and active_patch and active_patch.motor_trim_scale:
            active_trim = np.array(active_patch.motor_trim_scale, dtype=np.float32)

        # Extract gain multipliers
        pos_kp_mult = 1.0
        pos_kd_mult = 1.0
        att_kp_mult = 1.0
        att_kd_mult = 1.0
        if active_patch and active_patch.gain_multipliers:
            pos_kp_mult = active_patch.gain_multipliers.pos_kp
            pos_kd_mult = active_patch.gain_multipliers.pos_kd
            att_kp_mult = active_patch.gain_multipliers.att_kp
            att_kd_mult = active_patch.gain_multipliers.att_kd

        # Target setpoint in MuJoCo frame
        target_z = self.drone_cfg.target_z
        target_pos = np.array([0.0, 0.0, target_z], dtype=np.float32)
        if active_patch and active_patch.safety_envelope and active_patch.safety_envelope.temp_setpoint_offset:
            target_pos += np.array(active_patch.safety_envelope.temp_setpoint_offset, dtype=np.float32)

        # ----------------------------------------------------------------------
        # 2. Coordinate Transformation (MuJoCo ENU/NWU -> Mini-QuadLib NED)
        # ----------------------------------------------------------------------
        pos_ned = self._T @ pos
        vel_ned = self._T @ vel
        omega_ned = self._T @ omega
        desired_pos_ned = self._T @ target_pos

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
        setpoint = QuadrotorSetpoint(position=desired_pos_ned)

        # ----------------------------------------------------------------------
        # 3. Geometric Baseline Controller (SE(3) tracking)
        # ----------------------------------------------------------------------
        effective_pos_gains = self.l1_cfg.pos_gains * np.array([pos_kp_mult, pos_kp_mult, pos_kp_mult], dtype=np.float32)
        effective_vel_gains = self.l1_cfg.vel_gains * np.array([pos_kd_mult, pos_kd_mult, pos_kd_mult], dtype=np.float32)
        effective_att_gains = self.l1_cfg.att_gains * np.array([att_kp_mult, att_kp_mult, att_kp_mult], dtype=np.float32)
        effective_ang_gains = self.l1_cfg.ang_gains * np.array([att_kd_mult, att_kd_mult, att_kd_mult], dtype=np.float32)

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

        # Apply System 2 feedforward force offset if present
        if active_ff is not None:
            # Vertical offset directly adds to collective thrust
            F_z += float(active_ff[2])
            # Horizontal force offset creates lean moments
            # Fx > 0 commands pitch down (-My); Fy > 0 commands roll right (+Mx)
            lever_arm = 0.05
            M_mujoco[1] -= float(active_ff[0]) * lever_arm
            M_mujoco[0] += float(active_ff[1]) * lever_arm

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
            target_pos=target_pos,
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
            # L1 Adaptive Controller Metrics
            "u_b": [float(x) for x in u_b],
            "u_ad": [float(x) for x in u_ad],
            "u_total": [float(x) for x in u_total],
            "sigma_f_hat": [float(x) for x in sigma_f_hat_mujoco],
            "sigma_M_hat": [float(x) for x in sigma_M_hat_mujoco],
            "euler_deg": [roll_deg, pitch_deg, yaw_deg],
            "is_triggered": is_triggered,
            "trigger_reasons": trigger_reasons,
            "pos_error": [float(x) for x in (target_pos - pos)],
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
    ) -> Tuple[np.ndarray, dict]:
        """Legacy PID reflex controller fallback."""
        x, y, z = pos
        vx, vy, vz = vel

        # Altitude
        alt_error = self.drone_cfg.target_z - z
        F_z = self.drone_cfg.total_hover_weight + self.pid_cfg.kp_z * alt_error - self.pid_cfg.kd_z * vz
        if feedforward_force is not None:
            F_z += feedforward_force[2]

        # Horizontal Position Hold
        target_pitch = np.clip(
            -self.pid_cfg.kp_xy * x - self.pid_cfg.kd_xy * vx,
            -self.pid_cfg.max_tilt_rad,
            self.pid_cfg.max_tilt_rad,
        )
        target_roll = np.clip(
            self.pid_cfg.kp_xy * y + self.pid_cfg.kd_xy * vy,
            -self.pid_cfg.max_tilt_rad,
            self.pid_cfg.max_tilt_rad,
        )

        # Attitude & Heading
        roll_err = target_roll - (2.0 * quat[1])
        pitch_err = target_pitch - (2.0 * quat[2])
        yaw_err = 0.0 - (2.0 * quat[3])

        tau_x = self.pid_cfg.kp_att * roll_err - self.pid_cfg.kd_att * omega[0]
        tau_y = self.pid_cfg.kp_att * pitch_err - self.pid_cfg.kd_att * omega[1]
        tau_z = self.pid_cfg.kp_yaw * yaw_err - self.pid_cfg.kd_yaw * omega[2]

        demands = np.array([F_z, tau_x, tau_y, tau_z])
        motor_thrusts = self.B_INV @ demands

        if motor_trim is not None:
            motor_thrusts = motor_thrusts * motor_trim

        motor_thrusts = np.clip(
            motor_thrusts,
            self.drone_cfg.min_thrust,
            self.drone_cfg.max_thrust,
        )

        target_pos = np.array([0.0, 0.0, self.drone_cfg.target_z])
        self.bridge.push_telemetry(
            sim_time=sim_time,
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            motor_ctrl=motor_thrusts,
            target_pos=target_pos,
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
            "is_triggered": False,
            "trigger_reasons": [],
            "pos_error": [float(x) for x in (target_pos - pos)],
        }
        return motor_thrusts, diagnostics
