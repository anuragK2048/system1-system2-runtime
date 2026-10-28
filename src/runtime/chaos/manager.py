"""Thread-Safe Real-Time Chaos Manager.

Coordinates parameterized failures across both drones with:
1. Real-time parameter tweaking (motor thrusts, wind 3D vector, cargo drag, sensor chatter).
2. Targeting modes: Synchronized ('both') for fair A/B comparisons or Targeted ('drone1'/'drone2').
3. Instant repeatable preset application.
4. Active chaos state reporting for HUD overlays, REST APIs, and WebSocket telemetry.
5. 100% backward compatibility with legacy keyboard listeners and unit tests.
"""

import time
import ctypes
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import mujoco

from runtime.chaos.models import (
    ChaosConfig,
    ChaosTarget,
    DroneChaosState,
    MotorChaosConfig,
    WindChaosConfig,
    PayloadChaosConfig,
    SensorChaosConfig,
)
from runtime.chaos.physics import (
    apply_actuator_degradation,
    compute_wind_force,
    apply_wind_to_body,
    apply_payload_physics as phys_apply_payload,
    corrupt_imu_sensors,
)
from runtime.chaos.presets import get_preset_state, CHAOS_PRESETS


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
        """Returns True if physical Shift key is held down on macOS."""
        return bool(_core_graphics.CGEventSourceFlagsState(0) & 0x00020000)
except Exception:
    def is_shift_down() -> bool:
        return False


# ==============================================================================
# BACKWARD-COMPATIBLE ACTIVE CHAOS DICTIONARY PROXY
# ==============================================================================
class _ActiveChaosDict(dict):
    """Proxy dictionary allowing legacy code like `active_chaos['wind_gust'] = True`

    to automatically mutate the underlying modern DroneChaosState parameters.
    """

    def __init__(self, manager: "ChaosManager") -> None:
        super().__init__()
        self._manager = manager

    def __getitem__(self, key: str) -> bool:
        d1 = self._manager.drone1
        d2 = self._manager.drone2
        if key == "motor_failure":
            return d1.motor.is_active() or d2.motor.is_active()
        elif key == "swinging_weight":
            return d1.payload.is_active() or d2.payload.is_active()
        elif key == "sensor_noise":
            return d1.sensor.is_active() or d2.sensor.is_active()
        elif key == "wind_gust":
            return d1.wind.is_active() or d2.wind.is_active()
        return super().get(key, False)

    def __setitem__(self, key: str, value: bool) -> None:
        super().__setitem__(key, bool(value))
        if key == "motor_failure":
            m_val = self._manager.cfg.motor_failure_multiplier if value else 1.0
            self._manager.set_motor_multiplier(0, m_val, target=ChaosTarget.BOTH)
        elif key == "swinging_weight":
            self._manager.set_payload_attached(bool(value), target=ChaosTarget.BOTH)
        elif key == "sensor_noise":
            noise_std = self._manager.cfg.sensor_noise_std if value else 0.0
            self._manager.set_sensor_noise(noise_std, target=ChaosTarget.BOTH)
        elif key == "wind_gust":
            if value:
                self._manager.set_wind_force(
                    self._manager.cfg.wind_gust_force_newtons, 0.0, 0.0, target=ChaosTarget.BOTH
                )
            else:
                self._manager.set_wind_force(0.0, 0.0, 0.0, target=ChaosTarget.BOTH)


# ==============================================================================
# CHAOS MANAGER CLASS
# ==============================================================================
class ChaosManager:
    """Central thread-safe manager for chaos injection across dual drones.

    Attributes:
        cfg: Baseline ChaosConfig parameters.
        drone2: Chaos state for Drone 2 (Left / Cyan).
        drone1: Chaos state for Drone 1 (Right / Orange).
        target_mode: ChaosTarget ('both', 'drone1', 'drone2').
        reset_requested: Boolean flag when reset is requested.
        active_chaos: Backward-compatible dictionary of active failure modes.
    """

    def __init__(self, cfg: Optional[ChaosConfig] = None) -> None:
        self.cfg = cfg or ChaosConfig()
        self._lock = threading.RLock()

        # Independent chaos state per drone
        self.drone1 = DroneChaosState()
        self.drone2 = DroneChaosState()
        self.target_mode: ChaosTarget = ChaosTarget.BOTH

        # Set of payload body IDs that have already received their initial attachment kick
        self._kicked_payloads: Set[int] = set()

        # Reset flag
        self.reset_requested: bool = False

        # Backward compatibility proxy
        self.active_chaos = _ActiveChaosDict(self)
        self.active_chaos["motor_failure"] = False
        self.active_chaos["swinging_weight"] = False
        self.active_chaos["sensor_noise"] = False
        self.active_chaos["wind_gust"] = False

    # ==========================================================================
    # TARGETING HELPERS
    # ==========================================================================
    def _get_target_states(self, target: Optional[ChaosTarget] = None) -> List[DroneChaosState]:
        """Resolves target into list of DroneChaosState objects to modify."""
        t = target or self.target_mode
        if t == ChaosTarget.DRONE1:
            return [self.drone1]
        elif t == ChaosTarget.DRONE2:
            return [self.drone2]
        return [self.drone1, self.drone2]

    # ==========================================================================
    # REAL-TIME PARAMETER MUTATORS (Thread-Safe)
    # ==========================================================================
    def set_target_mode(self, mode: ChaosTarget) -> None:
        """Sets the active targeting mode ('both', 'drone1', or 'drone2')."""
        with self._lock:
            self.target_mode = mode

    def set_motor_multiplier(
        self,
        motor_idx: int,
        multiplier: float,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Sets the thrust multiplier (0.0 to 1.0) for a specific motor (0..3)."""
        with self._lock:
            clamped = float(np.clip(multiplier, 0.0, 1.5))
            for state in self._get_target_states(target):
                state.motor.multipliers[motor_idx] = clamped

    def set_chipped_propeller(
        self,
        motor_idx: Optional[int] = 0,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Enables high-frequency aerodynamic flutter for a chipped propeller blade."""
        with self._lock:
            for state in self._get_target_states(target):
                state.motor.chipped_motor_idx = motor_idx

    def set_motor_multipliers(
        self,
        multipliers: List[float],
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Sets all 4 motor thrust multipliers at once."""
        with self._lock:
            for state in self._get_target_states(target):
                for i in range(min(4, len(multipliers))):
                    state.motor.multipliers[i] = float(np.clip(multipliers[i], 0.0, 1.5))

    def set_wind_force(
        self,
        fx: float,
        fy: float = 0.0,
        fz: float = 0.0,
        turbulence_std: float = 0.0,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Sets continuous 3D wind disturbance force vector [Fx, Fy, Fz] in Newtons."""
        with self._lock:
            enabled = bool(abs(fx) > 0.01 or abs(fy) > 0.01 or abs(fz) > 0.01)
            for state in self._get_target_states(target):
                state.wind.enabled = enabled
                state.wind.steady_force = [float(fx), float(fy), float(fz)]
                state.wind.turbulence_std = float(max(0.0, turbulence_std))

    def trigger_wind_gust(
        self,
        force: float = 5.0,
        direction: Optional[List[float]] = None,
        duration_s: float = 0.8,
        sim_time: float = 0.0,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Triggers a transient half-sine wind gust burst."""
        with self._lock:
            dir_vec = direction or [1.0, 0.0, 0.0]
            for state in self._get_target_states(target):
                state.wind.gust_active = True
                state.wind.gust_force = float(force)
                state.wind.gust_direction = list(dir_vec)
                state.wind.gust_duration_s = float(duration_s)
                state.wind.gust_start_time = float(sim_time)

    def set_payload_attached(
        self,
        attached: bool,
        mass: Optional[float] = None,
        aerodynamic_drag: bool = True,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Attaches or detaches the tethered cargo pendulum."""
        with self._lock:
            for state in self._get_target_states(target):
                state.payload.attached = bool(attached)
                if mass is not None:
                    state.payload.mass = float(max(0.05, mass))
                state.payload.aerodynamic_drag_enabled = bool(aerodynamic_drag)

    def set_sensor_noise(
        self,
        gyro_std: float,
        accel_std: float = 0.0,
        target: Optional[ChaosTarget] = None,
    ) -> None:
        """Sets IMU sensor vibration noise amplitude."""
        with self._lock:
            enabled = bool(gyro_std > 0.001 or accel_std > 0.001)
            for state in self._get_target_states(target):
                state.sensor.enabled = enabled
                state.sensor.gyro_noise_std = float(max(0.0, gyro_std))
                state.sensor.accel_noise_std = float(max(0.0, accel_std))

    def apply_preset(
        self,
        preset_key: str,
        additive: bool = False,
        target: Optional[ChaosTarget] = None,
    ) -> str:
        """Applies a named preset scenario to the target drone(s).

        If additive is True, layers the failure mode onto currently active chaos.
        If additive is False, replaces the entire state (legacy behavior).
        """
        with self._lock:
            target_mode = target or self.target_mode
            targets = self._get_target_states(target_mode)

            if str(preset_key) in ("0", "clean"):
                self.clear_all(target=target_mode)
                return f"Clean nominal hover restored on {target_mode.value}"

            if additive:
                self._activate_preset_channels(preset_key, targets)
                return f"Preset '{preset_key}' added to {target_mode.value}"

            for s in targets:
                new_state = get_preset_state(preset_key)
                s.motor = new_state.motor
                s.wind = new_state.wind
                s.payload = new_state.payload
                s.sensor = new_state.sensor
            return f"Preset '{preset_key}' applied to {target_mode.value}"

    def toggle_preset(
        self,
        preset_key: str,
        target: Optional[ChaosTarget] = None,
    ) -> Dict[str, Any]:
        """Toggles a specific chaos channel (motor, wind, cargo, sensor) on or off.

        Enables composing multiple independent failures simultaneously on both drones.
        """
        with self._lock:
            target_mode = target or self.target_mode
            targets = self._get_target_states(target_mode)

            if str(preset_key) in ("0", "clean"):
                self.clear_all(target=target_mode)
                return {
                    "action": "cleared",
                    "preset_id": "0",
                    "active_presets": self.get_active_preset_keys(target_mode),
                    "summary": self.get_chaos_summary(),
                }

            is_active = self._is_preset_active_on_states(preset_key, targets)
            if is_active:
                self._deactivate_preset_channels(preset_key, targets)
                action = "deactivated"
            else:
                self._activate_preset_channels(preset_key, targets)
                action = "activated"

            return {
                "action": action,
                "preset_id": preset_key,
                "active_presets": self.get_active_preset_keys(target_mode),
                "summary": self.get_chaos_summary(),
            }

    def _is_preset_active_on_states(self, preset_key: str, states: List[DroneChaosState]) -> bool:
        """Determines if a given preset is currently active across target states."""
        key = str(preset_key).lower().strip()
        if key in ("1", "chipped_propeller", "m1"):
            return any(s.motor.multipliers[0] < 0.95 or s.motor.chipped_motor_idx is not None for s in states)
        elif key in ("2", "microburst_wind", "wind"):
            return any(s.wind.enabled and abs(s.wind.steady_force[0]) > 0.5 for s in states)
        elif key in ("3", "tethered_cargo_wind", "payload", "cargo"):
            return any(s.payload.attached for s in states)
        elif key in ("4", "sensor_chatter", "sensor", "imu"):
            return any(s.sensor.enabled and s.sensor.gyro_noise_std > 0.05 for s in states)
        elif key in ("5", "compound_stress", "compound"):
            return all(
                s.motor.multipliers[0] < 0.95 and s.wind.enabled and s.payload.attached
                for s in states
            )
        elif key in ("0", "clean"):
            return not any(s.is_any_active() for s in states)
        return False

    def _activate_preset_channels(self, preset_key: str, states: List[DroneChaosState]) -> None:
        """Activates only the specific failure channels targeted by this preset."""
        key = str(preset_key).lower().strip()
        if key in ("1", "chipped_propeller", "m1"):
            for s in states:
                s.motor.multipliers = [0.40, 1.0, 1.0, 1.0]
                s.motor.chipped_motor_idx = 0
        elif key in ("2", "microburst_wind", "wind"):
            for s in states:
                s.wind.enabled = True
                s.wind.steady_force = [6.5, 0.0, 0.0]
                s.wind.turbulence_std = 0.8
        elif key in ("3", "tethered_cargo_wind", "payload", "cargo"):
            for s in states:
                s.payload.attached = True
                s.payload.mass = 0.35
                s.payload.aerodynamic_drag_enabled = True
                if not s.wind.enabled:
                    s.wind.enabled = True
                    s.wind.steady_force = [4.0, 1.0, 0.0]
        elif key in ("4", "sensor_chatter", "sensor", "imu"):
            for s in states:
                s.sensor.enabled = True
                s.sensor.gyro_noise_std = 0.85
                s.sensor.harmonic_freq_hz = 85.0
                s.sensor.harmonic_amplitude = 0.3
        elif key in ("5", "compound_stress", "compound"):
            for s in states:
                s.motor.multipliers = [0.50, 1.0, 1.0, 1.0]
                s.motor.chipped_motor_idx = 0
                s.wind.enabled = True
                s.wind.steady_force = [5.0, 0.0, 0.0]
                s.wind.turbulence_std = 0.5
                s.payload.attached = True
                s.payload.mass = 0.35
                s.payload.aerodynamic_drag_enabled = True

    def _deactivate_preset_channels(self, preset_key: str, states: List[DroneChaosState]) -> None:
        """Deactivates only the specific failure channels targeted by this preset."""
        key = str(preset_key).lower().strip()
        if key in ("1", "chipped_propeller", "m1"):
            for s in states:
                s.motor.reset()
        elif key in ("2", "microburst_wind", "wind"):
            for s in states:
                s.wind.reset()
        elif key in ("3", "tethered_cargo_wind", "payload", "cargo"):
            for s in states:
                s.payload.reset()
        elif key in ("4", "sensor_chatter", "sensor", "imu"):
            for s in states:
                s.sensor.reset()
        elif key in ("5", "compound_stress", "compound"):
            for s in states:
                s.motor.reset()
                s.wind.reset()
                s.payload.reset()

    def get_active_preset_keys(self, target: Optional[ChaosTarget] = None) -> List[str]:
        """Returns list of preset IDs currently active on target drone(s)."""
        states = self._get_target_states(target or self.target_mode)
        active = []
        for k in ("1", "2", "3", "4", "5"):
            if self._is_preset_active_on_states(k, states):
                active.append(k)
        return active

    def clear_all(self, target: Optional[ChaosTarget] = None) -> None:
        """Restores target drone(s) back to pristine unperturbed nominal flight."""
        with self._lock:
            for state in self._get_target_states(target):
                state.reset()
            self._kicked_payloads.clear()

    # ==========================================================================
    # REAL-TIME MONITORING & TELEMETRY EXPORT
    # ==========================================================================
    def get_chaos_summary(self) -> Dict[str, Any]:
        """Returns a complete dictionary of current chaos state for API / WebSockets."""
        with self._lock:
            return {
                "target_mode": self.target_mode.value,
                "drone1": self.drone1.to_dict(),
                "drone2": self.drone2.to_dict(),
                "active_presets": self.get_active_preset_keys(),
                "hud_summary": {
                    "drone1": self.get_hud_text(1),
                    "drone2": self.get_hud_text(2),
                },
                "presets_available": list(CHAOS_PRESETS.keys()),
            }

    def get_hud_text(self, drone_idx: int) -> str:
        """Returns a concise, one-line status string suitable for MuJoCo HUD overlay."""
        state = self.drone1 if drone_idx == 1 else self.drone2
        if not state.is_any_active():
            return "CLEAN (Nominal Hover)"

        active_tags = []
        if state.motor.is_active():
            active_tags.append(f"MTR:[{state.motor.summary_str()}]")
        if state.wind.is_active():
            active_tags.append(f"WIND:[{state.wind.summary_str()}]")
        if state.payload.is_active():
            active_tags.append(f"CARGO:[{state.payload.summary_str()}]")
        if state.sensor.is_active():
            active_tags.append(f"IMU:[{state.sensor.summary_str()}]")

        return " | ".join(active_tags)

    # ==========================================================================
    # SIMULATION PHYSICS HOOKS (Called at 500 Hz from DualDroneSimulator)
    # ==========================================================================
    def apply_drone_physics(
        self,
        drone_idx: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        drone_body_id: int,
        payload_body_id: int,
        tether_geom_id: int,
        bob_geom_id: int,
        sim_time: float,
        payload_qpos_adr: int,
        payload_qvel_adr: int,
    ) -> np.ndarray:
        """Applies 3D wind and payload coupling for a specific drone at 500 Hz.

        Returns:
            The total 3D wind disturbance force vector applied to the drone body.
        """
        state = self.drone1 if drone_idx == 1 else self.drone2

        # 1. Compute 3D wind disturbance force
        wind_force = compute_wind_force(state.wind, sim_time)

        # 2. Apply wind force directly to drone airframe COM
        if np.any(np.abs(wind_force) > 0.001):
            apply_wind_to_body(data, drone_body_id, wind_force)

        # 3. Apply swinging payload coupling and wind aerodynamic drag on cargo bob
        phys_apply_payload(
            model=model,
            data=data,
            payload_body_id=payload_body_id,
            tether_geom_id=tether_geom_id,
            bob_geom_id=bob_geom_id,
            cfg=state.payload,
            wind_force=wind_force,
            kicked_set=self._kicked_payloads,
            qpos_adr=payload_qpos_adr,
            qvel_adr=payload_qvel_adr,
        )

        return wind_force

    def degrade_actuators_for_drone(
        self,
        drone_idx: int,
        commanded_thrust: np.ndarray,
        sim_time: float,
    ) -> np.ndarray:
        """Degrades actuator thrusts for a specific drone."""
        state = self.drone1 if drone_idx == 1 else self.drone2
        return apply_actuator_degradation(commanded_thrust, state.motor, sim_time)

    def corrupt_sensors_for_drone(
        self,
        drone_idx: int,
        raw_omega: np.ndarray,
        sim_time: float,
    ) -> np.ndarray:
        """Corrupts gyro angular velocity for a specific drone."""
        state = self.drone1 if drone_idx == 1 else self.drone2
        corrupted_omega, _ = corrupt_imu_sensors(raw_omega, state.sensor, sim_time)
        return corrupted_omega

    # ==========================================================================
    # BACKWARD COMPATIBILITY INTERFACES
    # ==========================================================================
    def degrade_actuators(self, commanded_thrust: np.ndarray) -> np.ndarray:
        """Legacy helper: degrades actuators for Drone 1 or synchronized both."""
        return self.degrade_actuators_for_drone(1, commanded_thrust, sim_time=0.0)

    def corrupt_sensor_data(self, omega: np.ndarray) -> np.ndarray:
        """Legacy helper: corrupts gyro readings for Drone 1 or synchronized both."""
        return self.corrupt_sensors_for_drone(1, omega, sim_time=0.0)

    def apply_wind_gust(
        self,
        data: mujoco.MjData,
        drone_body_id: int,
        current_time: float,
    ) -> None:
        """Legacy helper: applies lateral wind gust to body."""
        # Find which drone this body_id belongs to
        state = self.drone1
        if state.wind.is_active():
            wind_force = compute_wind_force(state.wind, current_time)
            apply_wind_to_body(data, drone_body_id, wind_force)

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
        """Legacy helper: executes payload physics."""
        state = self.drone1
        state.payload.mass = payload_mass
        wind_force = compute_wind_force(state.wind, sim_time=0.0)
        phys_apply_payload(
            model=model,
            data=data,
            payload_body_id=payload_body_id,
            tether_geom_id=tether_geom_id,
            bob_geom_id=bob_geom_id,
            cfg=state.payload,
            wind_force=wind_force,
            kicked_set=self._kicked_payloads,
            qpos_adr=qpos_adr,
            qvel_adr=qvel_adr,
        )

    def print_menu(self) -> None:
        """Displays the interactive testbed hotkey menu in the terminal."""
        print("\n" + "=" * 74)
        print("🚁 SKYDIO X2 - INTERACTIVE MULTIBODY CHAOS TESTBED (ENHANCED)")
        print("=" * 74)
        print("Keyboard Controls (Inside 3D Viewer):")
        print("  [Shift + 1] (or 6) 💥 PRESET 1: CUT MOTOR 1 BY 60%  (Chipped Propeller)")
        print("  [Shift + 2] (or 7) 📦 PRESET 2: ATTACH CARGO WEIGHT (0.35kg Tethered Bob)")
        print("  [Shift + 3] (or 8) ⚡ PRESET 3: INJECT SENSOR NOISE (IMU Gyro Chatter)")
        print("  [Shift + 4] (or 9) 💨 PRESET 4: 6.5N LATERAL WIND    (Crosswind + Turbulence)")
        print("  [Shift + 5] (or 0) 🔄 PRESET 0: RESET DRONES        (Restore Clean Hover)")
        print("  [Spacebar]         ⏸  PAUSE / RESUME SIMULATION")
        print("-" * 74)
        print("Real-time Web & API Control available on port 8000 via FastAPI.")
        print("Camera: AUTO-TRACKING ENABLED (Midpoint tracking between both drones)")
        print("=" * 74 + "\n")

    def handle_key(self, keycode: int) -> None:
        """Callback triggered whenever a keyboard key is pressed inside MuJoCo viewer."""
        shift = is_shift_down()

        # Preset 1: Motor 1 Failure (Shift + 1 / '!' / '6')
        if keycode == 33 or (keycode == 49 and shift) or keycode == 54:
            is_active = self.drone1.motor.is_active()
            new_multiplier = 1.0 if is_active else self.cfg.motor_failure_multiplier
            self.set_motor_multiplier(0, new_multiplier, target=ChaosTarget.BOTH)
            status = f"🔴 ACTIVE (M1 @ {new_multiplier*100:.0f}%)" if not is_active else "🟢 OFF (Nominal 100%)"
            print(f"\n🚨 [CHAOS 1] Motor 1 Failure: {status}")

        # Preset 2: Swinging Payload (Shift + 2 / '@' / '7')
        elif keycode == 64 or (keycode == 50 and shift) or keycode == 55:
            new_attached = not self.drone1.payload.attached
            self.set_payload_attached(new_attached, target=ChaosTarget.BOTH)
            status = "🔴 ATTACHED (0.35kg with Aerodynamic Drag)" if new_attached else "🟢 DETACHED (Weightless)"
            print(f"\n📦 [CHAOS 2] Tethered Cargo: {status}")

        # Preset 3: Sensor Noise (Shift + 3 / '#' / '8')
        elif keycode == 35 or (keycode == 51 and shift) or keycode == 56:
            new_noise = not self.drone1.sensor.enabled
            noise_val = self.cfg.sensor_noise_std if new_noise else 0.0
            self.set_sensor_noise(noise_val, target=ChaosTarget.BOTH)
            status = f"🔴 ACTIVE (±{noise_val:.2f} rad/s vibration)" if new_noise else "🟢 OFF (Clean)"
            print(f"\n⚡ [CHAOS 3] IMU Sensor Noise: {status}")

        # Preset 4: Lateral Wind (Shift + 4 / '$' / '9')
        elif keycode == 36 or (keycode == 52 and shift) or keycode == 57:
            new_wind = not self.drone1.wind.enabled
            f_wind = self.cfg.wind_gust_force_newtons if new_wind else 0.0
            self.set_wind_force(f_wind, 0.0, 0.0, turbulence_std=0.5 if new_wind else 0.0, target=ChaosTarget.BOTH)
            status = f"🔴 ACTIVE ({f_wind:.1f}N Crosswind +X)" if new_wind else "🟢 OFF (Calm)"
            print(f"\n💨 [CHAOS 4] Sustained Wind: {status}")

        # Preset 0 / Reset: Clear All (Shift + 5 / '%' / '0' / 'R')
        elif keycode == 37 or (keycode in (53, 48, 82, 114) and shift) or keycode == 48:
            self.clear_all(target=ChaosTarget.BOTH)
            self.reset_requested = True
            print("\n🔄 [RESET] All chaos modes CLEARED! Restoring both drones to level hover.")
