"""System 1 Reflex Controller (500 Hz PID Loop).

What is System 1?
-----------------
In human cognition and autonomous robotics:
- System 1 is the involuntary, fast 'spinal reflex' (running at 500 Hz).
  It doesn't think or plan; its sole job is to keep the drone from flipping over
  and crashing by computing motor thrusts based on instant sensor readings.
- System 2 (our LLM foundation model, running at 1-5 Hz) will later act as the
  'conscious brain' that observes when System 1 is struggling (due to chipped
  propellers, heavy swinging cargo, or high wind) and adjusts System 1's gains
  or applies compensatory force offsets.

How Drone Flight Physics Works:
-------------------------------
A quadrotor cannot push sideways directly. To move forward or backward, it must
tilt its nose down (pitch). To move left or right, it must tilt its body (roll).
To gain or lose altitude, it increases or decreases all 4 motors simultaneously.
"""

from typing import Optional, Tuple
import numpy as np
from runtime.sim.config import DroneConfig, PIDConfig


class System1Controller:
    """High-frequency Proportional-Derivative (PID) reflex controller for Skydio X2.
    
    Attributes:
        drone_cfg (DroneConfig): Physical drone constants (hover thrust, mixer matrix).
        pid_cfg (PIDConfig): Proportional & Derivative gains for altitude, attitude, and position.
    """

    def __init__(
        self,
        drone_cfg: Optional[DroneConfig] = None,
        pid_cfg: Optional[PIDConfig] = None
    ) -> None:
        self.drone_cfg = drone_cfg or DroneConfig()
        self.pid_cfg = pid_cfg or PIDConfig()

        # Cache allocation matrix inverse for high-performance matrix multiplication
        self.B_INV = self.drone_cfg.b_inv

    def compute_motor_thrusts(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        feedforward_force: Optional[np.ndarray] = None,
        motor_trim: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Calculates 4 motor thrust commands from current drone state.

        Args:
            pos: 3D position [x, y, z] in world frame (meters).
            vel: 3D linear velocity [vx, vy, vz] in world frame (m/s).
            quat: Orientation quaternion [w, x, y, z] in world frame.
            omega: Angular velocity [wx, wy, wz] around body axes (rad/s).
            feedforward_force: Optional [Fx, Fy, Fz] force offset from System 2.
            motor_trim: Optional [s1, s2, s3, s4] per-motor scaling from System 2.

        Returns:
            Tuple of:
              - motor_thrusts: 4-element array [u1, u2, u3, u4] in Newtons, clamped to [0, 13] N.
              - telemetry_diagnostics: Dictionary containing computed forces, torques, and errors.
        """
        x, y, z = pos
        vx, vy, vz = vel

        # ----------------------------------------------------------------------
        # STEP 1: Altitude Control (Vertical Z Axis)
        # ----------------------------------------------------------------------
        # Target: Stay at target_z (1.0 meter).
        # We start with the base hover thrust (total weight of drone = ~13 N).
        # - If z < target_z: push harder upward (+Kp_z * error).
        # - If vz > 0 (climbing fast): apply damping braking (-Kd_z * vz).
        alt_error = self.drone_cfg.target_z - z
        F_z = self.drone_cfg.total_hover_weight + self.pid_cfg.kp_z * alt_error - self.pid_cfg.kd_z * vz

        # If System 2 provided a vertical feedforward offset, add it here
        if feedforward_force is not None:
            F_z += feedforward_force[2]

        # ----------------------------------------------------------------------
        # STEP 2: Horizontal Position Hold (X and Y Coordinates)
        # ----------------------------------------------------------------------
        # To stay at (x=0, y=0), the drone tilts slightly in the opposite direction of drift:
        # - Drifting in +X -> tilt nose down (pitch < 0) to push back.
        # - Drifting in +Y -> tilt right (roll > 0) to push back.
        # Clamped to max_tilt_rad (~8.6 degrees) so the drone never flips over.
        target_pitch = np.clip(
            -self.pid_cfg.kp_xy * x - self.pid_cfg.kd_xy * vx,
            -self.pid_cfg.max_tilt_rad,
            self.pid_cfg.max_tilt_rad
        )
        target_roll = np.clip(
            self.pid_cfg.kp_xy * y + self.pid_cfg.kd_xy * vy,
            -self.pid_cfg.max_tilt_rad,
            self.pid_cfg.max_tilt_rad
        )

        # ----------------------------------------------------------------------
        # STEP 3: Attitude & Heading Control (Roll, Pitch, Yaw)
        # ----------------------------------------------------------------------
        # For small angles, a quaternion [w, x, y, z] maps to Euler angles via:
        #   current_roll  ≈ 2 * quat[1]
        #   current_pitch ≈ 2 * quat[2]
        #   current_yaw   ≈ 2 * quat[3]
        roll_err = target_roll - (2.0 * quat[1])
        pitch_err = target_pitch - (2.0 * quat[2])
        yaw_err = 0.0 - (2.0 * quat[3])  # Keep heading at 0 degrees

        # Compute desired corrective torques around the 3 body axes:
        # tau = Kp * angle_error - Kd * angular_velocity
        tau_x = self.pid_cfg.kp_att * roll_err - self.pid_cfg.kd_att * omega[0]
        tau_y = self.pid_cfg.kp_att * pitch_err - self.pid_cfg.kd_att * omega[1]
        tau_z = self.pid_cfg.kp_yaw * yaw_err - self.pid_cfg.kd_yaw * omega[2]

        # ----------------------------------------------------------------------
        # STEP 4: Control Allocation / Mixer
        # ----------------------------------------------------------------------
        # Convert virtual demands [F_z, tau_x, tau_y, tau_z] into 4 motor thrusts.
        # Matrix multiplication: [u1, u2, u3, u4] = B_INV @ [F_z, tau_x, tau_y, tau_z]
        demands = np.array([F_z, tau_x, tau_y, tau_z])
        motor_thrusts = self.B_INV @ demands

        # Optional System 2 Motor Trim scaling (e.g., compensating for damaged prop)
        if motor_trim is not None:
            motor_thrusts = motor_thrusts * motor_trim

        # Clamp motor commands to physically allowable bounds [0.0 N, 13.0 N]
        motor_thrusts = np.clip(
            motor_thrusts,
            self.drone_cfg.min_thrust,
            self.drone_cfg.max_thrust
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
        }

        return motor_thrusts, diagnostics
