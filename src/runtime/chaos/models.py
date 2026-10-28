"""Data models and schemas for the Chaos Injection Engine.

Defines typed configurations for:
- Motor degradation and physical failures
- 3D wind vectors, gusts, and turbulence
- Tethered swinging payload with aerodynamic wind coupling
- Noisy IMU sensors (gyro chatter, accel noise, bias drift)
- Drone-specific and global synchronized chaos states
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class ChaosTarget(str, Enum):
    """Specifies which drone(s) receive chaos injection."""
    BOTH = "both"
    DRONE1 = "drone1"  # Left drone (Orange)
    DRONE2 = "drone2"  # Right drone (Cyan)


@dataclass
class MotorChaosConfig:
    """Configuration for quadrotor motor degradation and failure modes."""
    # Thrust scale multipliers for motors 1..4 (1.0 = nominal 100%, 0.4 = 60% loss, 0.0 = dead)
    multipliers: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Which motor has a chipped propeller blade (induces high-frequency vibration torque)
    chipped_motor_idx: Optional[int] = None
    # Intermittent electrical / ESC cutout probability per step (0.0 = none)
    dropout_prob: float = 0.0

    def is_active(self) -> bool:
        """Returns True if any motor is degraded or faulted."""
        return any(m < 0.999 for m in self.multipliers) or (self.chipped_motor_idx is not None) or (self.dropout_prob > 0.0)

    def summary_str(self) -> str:
        """Compact status string for HUD / telemetry."""
        if not self.is_active():
            return "100% (Clean)"
        degraded = [f"M{i+1}:{self.multipliers[i]*100:.0f}%" for i in range(4) if self.multipliers[i] < 0.999]
        if self.chipped_motor_idx is not None:
            degraded.append(f"M{self.chipped_motor_idx+1}:CHIPPED")
        return ", ".join(degraded) if degraded else "Faulted"


@dataclass
class WindChaosConfig:
    """Configuration for 3D wind, lateral gusts, and atmospheric turbulence."""
    enabled: bool = False
    # Steady 3D wind force vector [Fx, Fy, Fz] in Newtons applied directly to the drone body
    steady_force: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Turbulence intensity (standard deviation of Dryden-like stochastic buffeting in Newtons)
    turbulence_std: float = 0.0
    # Dynamic gust burst settings
    gust_active: bool = False
    gust_force: float = 5.0                # Gust peak magnitude in Newtons
    gust_direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])  # Unit vector
    gust_start_time: float = 0.0
    gust_duration_s: float = 0.80

    def is_active(self) -> bool:
        """Returns True if any wind or gust is currently active."""
        has_force = any(abs(f) > 0.01 for f in self.steady_force)
        return (self.enabled and has_force) or self.gust_active or (self.turbulence_std > 0.0)

    def summary_str(self) -> str:
        """Compact status string for HUD / telemetry."""
        if not self.is_active():
            return "Calm (0.0N)"
        fx, fy, fz = self.steady_force
        parts = []
        if any(abs(f) > 0.05 for f in (fx, fy, fz)):
            parts.append(f"[{fx:+.1f},{fy:+.1f},{fz:+.1f}]N")
        if self.gust_active:
            parts.append(f"GUST:{self.gust_force:.1f}N")
        if self.turbulence_std > 0.0:
            parts.append(f"Turb:{self.turbulence_std:.1f}")
        return " ".join(parts) if parts else "Windy"


@dataclass
class PayloadChaosConfig:
    """Configuration for tethered sloshing/pendulum cargo attached under the drone."""
    attached: bool = False
    mass: float = 0.35                      # Mass of payload bob in kg (~26% of drone weight)
    kick_angle_rad: float = 0.30            # Initial displacement kick (~17 degrees)
    kick_velocity: float = 1.5              # Initial swing velocity (rad/s)
    # Physical wind aerodynamic coupling: wind blowing on the payload bob induces realistic swinging
    aerodynamic_drag_enabled: bool = True
    drag_coeff: float = 0.47                # Drag coefficient of spherical bob
    projected_area: float = 0.0045          # Cross-sectional area of sphere (radius ~3.8cm)

    def is_active(self) -> bool:
        """Returns True if payload cargo is attached."""
        return self.attached

    def summary_str(self) -> str:
        """Compact status string for HUD / telemetry."""
        if not self.attached:
            return "Detached (0.00kg)"
        drag_str = "+WindDrag" if self.aerodynamic_drag_enabled else ""
        return f"{self.mass:.2f}kg Attached{drag_str}"


@dataclass
class SensorChaosConfig:
    """Configuration for IMU sensor noise, vibrations, and bias drift."""
    enabled: bool = False
    # High-frequency Gaussian vibration noise added to angular velocity (rad/s std)
    gyro_noise_std: float = 0.0
    # Vibration noise added to linear accelerometer (m/s^2 std)
    accel_noise_std: float = 0.0
    # Gyroscope constant bias drift [bx, by, bz] in rad/s
    gyro_bias: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Loose motor mount harmonic vibration frequency (Hz) and amplitude (rad/s)
    harmonic_freq_hz: float = 0.0
    harmonic_amplitude: float = 0.0

    def is_active(self) -> bool:
        """Returns True if any sensor corruption is enabled."""
        return (
            self.enabled and (
                self.gyro_noise_std > 0.001 or
                self.accel_noise_std > 0.001 or
                any(abs(b) > 0.001 for b in self.gyro_bias) or
                self.harmonic_amplitude > 0.001
            )
        )

    def summary_str(self) -> str:
        """Compact status string for HUD / telemetry."""
        if not self.is_active():
            return "Clean Sensors"
        items = []
        if self.gyro_noise_std > 0.0:
            items.append(f"GyroNoise:±{self.gyro_noise_std:.2f}")
        if any(abs(b) > 0.001 for b in self.gyro_bias):
            items.append("BiasDrift")
        if self.harmonic_amplitude > 0.0:
            items.append(f"Chirp:{self.harmonic_freq_hz:.0f}Hz")
        return ", ".join(items) if items else "Noisy"


@dataclass
class DroneChaosState:
    """Encapsulates all real-time chaos parameters applied to a single drone."""
    motor: MotorChaosConfig = field(default_factory=MotorChaosConfig)
    wind: WindChaosConfig = field(default_factory=WindChaosConfig)
    payload: PayloadChaosConfig = field(default_factory=PayloadChaosConfig)
    sensor: SensorChaosConfig = field(default_factory=SensorChaosConfig)

    def is_any_active(self) -> bool:
        """Returns True if any physical failure or disturbance is active."""
        return (
            self.motor.is_active() or
            self.wind.is_active() or
            self.payload.is_active() or
            self.sensor.is_active()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current chaos state for REST/WebSocket transmission."""
        return {
            "is_active": self.is_any_active(),
            "motor": {
                "active": self.motor.is_active(),
                "multipliers": list(self.motor.multipliers),
                "chipped_motor": self.motor.chipped_motor_idx,
                "summary": self.motor.summary_str(),
            },
            "wind": {
                "active": self.wind.is_active(),
                "steady_force": list(self.wind.steady_force),
                "gust_active": self.wind.gust_active,
                "turbulence_std": self.wind.turbulence_std,
                "summary": self.wind.summary_str(),
            },
            "payload": {
                "active": self.payload.is_active(),
                "mass": self.payload.mass,
                "aerodynamic_drag": self.payload.aerodynamic_drag_enabled,
                "summary": self.payload.summary_str(),
            },
            "sensor": {
                "active": self.sensor.is_active(),
                "gyro_noise_std": self.sensor.gyro_noise_std,
                "summary": self.sensor.summary_str(),
            },
        }

    def reset(self) -> None:
        """Restores all parameters to pristine, unperturbed nominal flight."""
        self.motor = MotorChaosConfig()
        self.wind = WindChaosConfig()
        self.payload = PayloadChaosConfig()
        self.sensor = SensorChaosConfig()
