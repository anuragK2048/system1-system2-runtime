"""Chaos Injector and Failure Mode Simulator.

NOTE: The chaos injection engine has been refactored and modernized into the
dedicated `runtime.chaos` package for modular real-time continuous control,
aerodynamic wind coupling, and dual-drone targeting.

This module re-exports `ChaosManager` from `runtime.chaos` to maintain 100%
backward compatibility with existing simulation runners and unit tests.
"""

from runtime.chaos.manager import ChaosManager, is_shift_down
from runtime.chaos.models import (
    ChaosTarget,
    MotorChaosConfig,
    WindChaosConfig,
    PayloadChaosConfig,
    SensorChaosConfig,
    DroneChaosState,
)
from runtime.chaos.presets import CHAOS_PRESETS, get_preset_state

<<<<<<< HEAD

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

        # F3: 0.0 = no gust latched; otherwise the time (same clock as the
        # current_time passed to apply_wind_gust) when the gust expires.
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
        print("  [Shift + 4] (or 9) 💨 5.0N WIND GUST (0.8s timed, press again to cancel)")
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
        # Chaos 4: 5.0N Timed Wind Gust
        # Triggered by: Shift + 4 ('$' = 36, or key 52 with Shift) OR key '9' (57)
        # ----------------------------------------------------------------------
        elif keycode == 36 or (keycode == 52 and shift) or keycode == 57:
            if self.active_chaos["wind_gust"]:
                # Cancel the timed gust early (F3)
                self.active_chaos["wind_gust"] = False
                self.gust_end_time = 0.0
                print("\n💨 [CHAOS 4] Wind Gust: 🟢 CANCELLED early (Calm air)")
            else:
                # Start a timed gust; end time latches on next apply_wind_gust (F3)
                self.active_chaos["wind_gust"] = True
                self.gust_end_time = 0.0
                print(f"\n💨 [CHAOS 4] Wind Gust: 🔴 ACTIVE (5.0N pushing +X for {self.cfg.wind_gust_duration_seconds:.2f}s)")

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
        self.gust_end_time = 0.0
        self._was_payload_active = False

    def trigger_wind_gust(self, current_time: float) -> None:
        """Starts a timed wind gust immediately (F3).

        The gust stays active for `wind_gust_duration_seconds` and then turns
        itself off inside apply_wind_gust. Toggling the flag off early (hotkey
        or direct assignment + gust_end_time reset) still cancels it.

        Args:
            current_time: Now, on the same clock passed to apply_wind_gust
                (wall time in the viewer loop, sim time in headless runs).
        """
        self.active_chaos["wind_gust"] = True
        self.gust_end_time = current_time + self.cfg.wind_gust_duration_seconds
        print(
            f"\n💨 [CHAOS 4] Wind Gust: 🔴 ACTIVE "
            f"(5.0N pushing +X for {self.cfg.wind_gust_duration_seconds:.2f}s)"
        )

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
        payload_mass: float,
        qpos_adr: int = 7,
        qvel_adr: int = 6,
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
        if not hasattr(self, "_kicked_payloads"):
            self._kicked_payloads = set()

        if self.active_chaos["swinging_weight"]:
            # Newly attached: impart an initial swing angle and velocity
            if payload_body_id not in self._kicked_payloads:
                data.qpos[qpos_adr] = self.cfg.payload_kick_angle_rad  # ~17 degrees
                data.qvel[qvel_adr] = self.cfg.payload_kick_velocity   # 1.5 rad/s
                self._kicked_payloads.add(payload_body_id)

            # Make tether (gold) and bob (red) visible
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 1.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 1.0]

            # Clear counter-force so full gravity pulls down on the drone
            data.xfrc_applied[payload_body_id, :] = 0.0
        else:
            self._kicked_payloads.discard(payload_body_id)

            # Make tether and bob invisible
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 0.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 0.0]

            # Neutralize payload weight so the drone feels exactly 0.00 kg
            data.xfrc_applied[payload_body_id, 2] = payload_mass * 9.81
            data.qpos[qpos_adr : qpos_adr + 2] = 0.0
            data.qvel[qvel_adr : qvel_adr + 2] = 0.0

    def apply_wind_gust(
        self,
        data: mujoco.MjData,
        drone_body_id: int,
        current_time: float
    ) -> None:
        """Applies the timed wind gust to the drone body while it is active (F3).

        When the gust flag is on but no end time is latched yet
        (gust_end_time <= 0, e.g. hotkey trigger or direct flag assignment),
        the end time latches here as current_time + duration. Once
        current_time passes the end time, the gust turns itself off.
        """
        if self.active_chaos["wind_gust"]:
            if self.gust_end_time <= 0.0:
                # Just triggered: latch the expiry time on this clock.
                self.gust_end_time = current_time + self.cfg.wind_gust_duration_seconds
            elif current_time >= self.gust_end_time:
                # Gust expired: switch it off (hotkey-off earlier also lands here
                # cleanly since it clears both the flag and the end time).
                self.active_chaos["wind_gust"] = False
                self.gust_end_time = 0.0
                print(f"\n💨 [CHAOS 4] Wind gust expired after {self.cfg.wind_gust_duration_seconds:.2f}s. Calm air.")
                return
            # Gust still active: apply lateral force along the world X axis
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
=======
__all__ = [
    "ChaosManager",
    "is_shift_down",
    "ChaosTarget",
    "MotorChaosConfig",
    "WindChaosConfig",
    "PayloadChaosConfig",
    "SensorChaosConfig",
    "DroneChaosState",
    "CHAOS_PRESETS",
    "get_preset_state",
]
>>>>>>> origin/main
