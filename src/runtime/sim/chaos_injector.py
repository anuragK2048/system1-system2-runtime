"""Chaos Injector and Failure Mode Simulator.

What is Chaos in this System?
------------------------------
Real-world drones face unpredictable environmental and mechanical anomalies:
1. Motor 1 Failure: A chipped or cracked propeller blade, or an ESC (Electronic Speed
   Controller) losing power. The drone commands full thrust, but the damaged rotor
   produces only 40%, creating severe roll/pitch instability.
2. Swinging Payload: Autonomous delivery drones carrying tethered cargo. The sloshing/
   pendulum mass acts as a non-linear oscillator that pulls the drone around.
3. Sensor Noise: Motor vibrations or loose mounting cause gyro chatter in the IMU,
   tricking standard PID into aggressive, destabilizing twitching.
4. Wind Gusts: Sudden 5.0 N lateral gusts testing whether the reflex controller can
   recover and maintain station.

This module encapsulates all chaos states, keyboard listeners (preventing MuJoCo
collisions), and physical perturbations applied to MuJoCo data.
"""

import time
import ctypes
from typing import Dict, Optional
import numpy as np
import mujoco
from runtime.sim.config import ChaosConfig


# ==============================================================================
# MACOS SHIFT KEY MODIFIER DETECTION VIA COREGRAPHICS
# ==============================================================================
try:
    _core_graphics = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
    )
    _core_graphics.CGEventSourceFlagsState.restype = ctypes.c_uint64
    _core_graphics.CGEventSourceFlagsState.argtypes = [ctypes.c_int32]

    def is_shift_down() -> bool:
        """Returns True if the physical Shift key is held down on macOS."""
        # kCGEventSourceStateCombinedSessionState = 0, kCGEventFlagMaskShift = 0x00020000
        return bool(_core_graphics.CGEventSourceFlagsState(0) & 0x00020000)
except Exception:
    def is_shift_down() -> bool:
        """Fallback for non-macOS platforms."""
        return False


# ==============================================================================
# CHAOS MANAGER CLASS
# ==============================================================================
class ChaosManager:
    """Manages failure modes, user hotkeys, and physical disturbances.

    Attributes:
        cfg (ChaosConfig): Parameters for failure intensities and durations.
        active_chaos (dict): Boolean flags for each active anomaly.
        gust_end_time (float): Timestamp when the active wind gust should expire.
        reset_requested (bool): Flag indicating the user pressed the reset key.
    """

    def __init__(self, cfg: Optional[ChaosConfig] = None) -> None:
        self.cfg = cfg or ChaosConfig()

        # Dictionary of active anomaly states
        self.active_chaos: Dict[str, bool] = {
            "motor_failure": False,    # Chaos 1: Motor 1 thrust reduced by 60%
            "swinging_weight": False,  # Chaos 2: True physical pendulum payload
            "sensor_noise": False,     # Chaos 3: IMU gyro sensor noise
            "wind_gust": False,        # Chaos 4: 5.0N lateral wind gust
        }

        self.gust_end_time: float = 0.0
        self.reset_requested: bool = False
        self._was_payload_active: bool = False

    def print_menu(self) -> None:
        """Displays the interactive testbed hotkey menu in the terminal."""
        print("\n" + "=" * 70)
        print("🚁 SKYDIO X2 - INTERACTIVE MULTIBODY CHAOS TESTBED")
        print("=" * 70)
        print("Press the following keys inside the 3D Viewer window:")
        print("  [Shift + 1] (or 6) 💥 CUT MOTOR 1 POWER BY 60%   (Chipped prop)")
        print("  [Shift + 2] (or 7) 📦 ATTACH SWINGING WEIGHT     (0.35kg red ball)")
        print("  [Shift + 3] (or 8) ⚡ INJECT SENSOR NOISE        (Gyro chatter)")
        print("  [Shift + 4] (or 9) 💨 5.0N LATERAL WIND          (Toggle On/Off)")
        print("  [Shift + 5] (or 0) 🔄 RESET DRONE                (Restore stable hover)")
        print("  [Spacebar]         ⏸  PAUSE / RESUME")
        print("-" * 70)
        print("🛡️  Anti-Collision Protection: Drone visibility locked permanently ON.")
        print("Camera: AUTO-TRACKING ENABLED (Drone will never leave screen)")
        print("  • 1-Finger Drag: Orbit around drone | 2-Finger Swipe: Zoom")
        print("=" * 70 + "\n")

    def handle_key(self, keycode: int) -> None:
        """Callback triggered whenever a keyboard key is pressed inside MuJoCo viewer.

        Args:
            keycode: Raw integer key code passed by GLFW / MuJoCo.
        """
        shift = is_shift_down()

        # ----------------------------------------------------------------------
        # Chaos 1: Motor 1 Failure
        # Triggered by: Shift + 1 ('!' = 33, or key 49 with Shift) OR key '6' (54)
        # ----------------------------------------------------------------------
        if keycode == 33 or (keycode == 49 and shift) or keycode == 54:
            self.active_chaos["motor_failure"] = not self.active_chaos["motor_failure"]
            status = "🔴 ACTIVE (Motor 1 at 40% power)" if self.active_chaos["motor_failure"] else "🟢 OFF (Full Power)"
            print(f"\n🚨 [CHAOS 1] Motor 1 Failure: {status}")

        # ----------------------------------------------------------------------
        # Chaos 2: Physical Swinging Weight
        # Triggered by: Shift + 2 ('@' = 64, or key 50 with Shift) OR key '7' (55)
        # ----------------------------------------------------------------------
        elif keycode == 64 or (keycode == 50 and shift) or keycode == 55:
            self.active_chaos["swinging_weight"] = not self.active_chaos["swinging_weight"]
            if self.active_chaos["swinging_weight"]:
                print("\n🚨 [CHAOS 2] Swinging Weight: 🔴 ATTACHED! (Look directly under drone: red ball swinging!)")
            else:
                print("\n🚨 [CHAOS 2] Swinging Weight: 🟢 DETACHED (Weightless)")

        # ----------------------------------------------------------------------
        # Chaos 3: Sensor Noise
        # Triggered by: Shift + 3 ('#' = 35, or key 51 with Shift) OR key '8' (56)
        # ----------------------------------------------------------------------
        elif keycode == 35 or (keycode == 51 and shift) or keycode == 56:
            self.active_chaos["sensor_noise"] = not self.active_chaos["sensor_noise"]
            status = "🔴 ACTIVE (IMU vibrating with noise)" if self.active_chaos["sensor_noise"] else "🟢 OFF (Clean sensors)"
            print(f"\n🚨 [CHAOS 3] Sensor Noise: {status}")

        # ----------------------------------------------------------------------
        # Chaos 4: 5.0N Sustained Wind
        # Triggered by: Shift + 4 ('$' = 36, or key 52 with Shift) OR key '9' (57)
        # ----------------------------------------------------------------------
        elif keycode == 36 or (keycode == 52 and shift) or keycode == 57:
            self.active_chaos["wind_gust"] = not self.active_chaos["wind_gust"]
            status = "🔴 ACTIVE (5.0N Crosswind pushing +X)" if self.active_chaos["wind_gust"] else "🟢 OFF (Calm air)"
            print(f"\n💨 [CHAOS 4] Sustained Wind: {status}")

        # ----------------------------------------------------------------------
        # Simulation Reset: Restore level hover and clear all anomalies
        # Triggered by: Shift + 5 ('%' = 37, key 53 with Shift, key '0'=48, or Shift+R/'R')
        # ----------------------------------------------------------------------
        elif keycode == 37 or (keycode in (53, 48, 82, 114) and shift) or keycode == 48:
            self.clear_all()
            self.reset_requested = True
            print("\n🔄 [RESET] All chaos modes CLEARED! Restoring drone to level hover.")

        # ----------------------------------------------------------------------
        # Helpful Reminder for accidental plain number key presses
        # ----------------------------------------------------------------------
        elif keycode in (49, 50, 51, 52, 53) and not shift:
            num = keycode - 48
            alt_key = num + 5
            print(f"\n💡 [TIP] Hold Shift! Press [Shift + {num}] (or single key [{alt_key}]) to trigger Chaos {num}.")

    def clear_all(self) -> None:
        """Clears all active failure modes back to safe defaults."""
        for k in self.active_chaos:
            self.active_chaos[k] = False
        self._was_payload_active = False

    # ==========================================================================
    # PHYSICS PERTURBATION APPLIERS
    # ==========================================================================
    def apply_payload_physics(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        payload_body_id: int,
        tether_geom_id: int,
        bob_geom_id: int,
        payload_mass: float
    ) -> None:
        """Simulates physical coupling or unhooking of the swinging cargo.

        When payload is attached:
          - Makes the tether (golden rod) and bob (red sphere) visible.
          - Allows full gravity (m*g) to pull down on the drone.
          - Gives an initial pendulum kick so the user sees the swing.

        When payload is detached:
          - Hides the tether and bob (sets visual alpha = 0.0).
          - Applies an exact upward counter-force (mp * 9.81) to neutralize its mass.
          - Resets swing angles to zero.
        """
        if self.active_chaos["swinging_weight"]:
            # Newly attached: impart an initial swing angle and velocity
            if not self._was_payload_active:
                data.qpos[7] = self.cfg.payload_kick_angle_rad  # ~17 degrees
                data.qvel[6] = self.cfg.payload_kick_velocity   # 1.5 rad/s
                self._was_payload_active = True

            # Make tether (gold) and bob (red) visible
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 1.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 1.0]

            # Clear counter-force so full gravity pulls down on the drone
            data.xfrc_applied[payload_body_id, :] = 0.0
        else:
            self._was_payload_active = False

            # Make tether and bob invisible
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 0.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 0.0]

            # Neutralize payload weight so the drone feels exactly 0.00 kg
            data.xfrc_applied[payload_body_id, 2] = payload_mass * 9.81
            data.qpos[7:9] = 0.0
            data.qvel[6:8] = 0.0

    def apply_wind_gust(
        self,
        data: mujoco.MjData,
        drone_body_id: int,
        current_time: float
    ) -> None:
        """Applies external wind force to the drone body if active."""
        if self.active_chaos["wind_gust"]:
            # Apply continuous lateral force along the world X axis
            data.xfrc_applied[drone_body_id, 0] = self.cfg.wind_gust_force_newtons

    def corrupt_sensor_data(self, omega: np.ndarray) -> np.ndarray:
        """Injects high-frequency Gaussian vibration noise into gyro rates."""
        if self.active_chaos["sensor_noise"]:
            noise = np.random.normal(0.0, self.cfg.sensor_noise_std, size=3)
            return omega + noise
        return omega

    def degrade_actuators(self, commanded_thrust: np.ndarray) -> np.ndarray:
        """Simulates physical thrust degradation (e.g., Motor 1 prop chipped)."""
        actual_thrust = commanded_thrust.copy()
        if self.active_chaos["motor_failure"]:
            actual_thrust[0] *= self.cfg.motor_failure_multiplier
        return actual_thrust
