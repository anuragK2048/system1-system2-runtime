"""Repeatable benchmark presets for chaos injection and controller resilience testing.

Provides pre-calibrated failure scenarios for reproducible evaluation:
- Preset 1: Chipped Propeller (Motor 1 at 40% thrust + rotor flutter)
- Preset 2: Severe 6.5N Crosswind Microburst
- Preset 3: Tethered Cargo in Lateral Wind (Swinging 0.35kg pendulum + aerodynamic drag)
- Preset 4: Loose IMU High-Frequency Sensor Chatter
- Preset 5: Compound Failure Stress Test (Motor degradation + Crosswind + Cargo)
- Preset 0: Clear All / Pristine Hover
"""

from typing import Dict, Any
from runtime.chaos.models import (
    DroneChaosState,
    MotorChaosConfig,
    WindChaosConfig,
    PayloadChaosConfig,
    SensorChaosConfig,
)


CHAOS_PRESETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "id": "chipped_propeller",
        "name": "Chipped Propeller (M1 40% Thrust)",
        "description": "Motor 1 produces only 40% power with aerodynamic flutter, simulating chipped blade or damaged ESC.",
        "motor": MotorChaosConfig(multipliers=[0.40, 1.0, 1.0, 1.0], chipped_motor_idx=0),
        "wind": WindChaosConfig(enabled=False),
        "payload": PayloadChaosConfig(attached=False),
        "sensor": SensorChaosConfig(enabled=False),
    },
    "2": {
        "id": "microburst_wind",
        "name": "6.5N Crosswind Microburst",
        "description": "Continuous 6.5N lateral wind pushing +X with 0.8N stochastic Dryden turbulence.",
        "motor": MotorChaosConfig(multipliers=[1.0, 1.0, 1.0, 1.0]),
        "wind": WindChaosConfig(enabled=True, steady_force=[6.5, 0.0, 0.0], turbulence_std=0.8),
        "payload": PayloadChaosConfig(attached=False),
        "sensor": SensorChaosConfig(enabled=False),
    },
    "3": {
        "id": "tethered_cargo_wind",
        "name": "Tethered Cargo in 4.0N Wind",
        "description": "0.35kg swinging pendulum mass with wind drag naturally exciting chaotic pendulum oscillations.",
        "motor": MotorChaosConfig(multipliers=[1.0, 1.0, 1.0, 1.0]),
        "wind": WindChaosConfig(enabled=True, steady_force=[4.0, 1.0, 0.0]),
        "payload": PayloadChaosConfig(attached=True, mass=0.35, aerodynamic_drag_enabled=True),
        "sensor": SensorChaosConfig(enabled=False),
    },
    "4": {
        "id": "sensor_chatter",
        "name": "Severe IMU Sensor Chatter",
        "description": "0.85 rad/s gyro noise with motor harmonic vibration resonance.",
        "motor": MotorChaosConfig(multipliers=[1.0, 1.0, 1.0, 1.0]),
        "wind": WindChaosConfig(enabled=False),
        "payload": PayloadChaosConfig(attached=False),
        "sensor": SensorChaosConfig(enabled=True, gyro_noise_std=0.85, harmonic_freq_hz=85.0, harmonic_amplitude=0.3),
    },
    "5": {
        "id": "compound_stress",
        "name": "Compound Failure Stress Test",
        "description": "Tri-fold stress test: Motor 1 at 50%, 5.0N crosswind, and swinging tethered cargo simultaneously.",
        "motor": MotorChaosConfig(multipliers=[0.50, 1.0, 1.0, 1.0], chipped_motor_idx=0),
        "wind": WindChaosConfig(enabled=True, steady_force=[5.0, 0.0, 0.0], turbulence_std=0.5),
        "payload": PayloadChaosConfig(attached=True, mass=0.35, aerodynamic_drag_enabled=True),
        "sensor": SensorChaosConfig(enabled=False),
    },
    "0": {
        "id": "clean",
        "name": "Clean Nominal Hover",
        "description": "All failure modes and environmental perturbations cleared.",
        "motor": MotorChaosConfig(multipliers=[1.0, 1.0, 1.0, 1.0]),
        "wind": WindChaosConfig(enabled=False),
        "payload": PayloadChaosConfig(attached=False),
        "sensor": SensorChaosConfig(enabled=False),
    },
}

# Alias dictionary for friendly lookup by key, name, or id
PRESET_ALIASES: Dict[str, str] = {
    "1": "1",
    "chipped_propeller": "1",
    "motor1": "1",
    "m1": "1",
    "2": "2",
    "wind": "2",
    "microburst": "2",
    "crosswind": "2",
    "3": "3",
    "payload": "3",
    "cargo": "3",
    "pendulum": "3",
    "4": "4",
    "sensor": "4",
    "noise": "4",
    "chatter": "4",
    "imu": "4",
    "5": "5",
    "compound": "5",
    "stress": "5",
    "extreme": "5",
    "0": "0",
    "clean": "0",
    "reset": "0",
    "clear": "0",
}


def get_preset_state(preset_key: str) -> DroneChaosState:
    """Returns a new DroneChaosState configured according to the requested preset."""
    canonical_key = PRESET_ALIASES.get(str(preset_key).lower().strip(), "0")
    p = CHAOS_PRESETS[canonical_key]
    return DroneChaosState(
        motor=MotorChaosConfig(
            multipliers=list(p["motor"].multipliers),
            chipped_motor_idx=p["motor"].chipped_motor_idx,
            dropout_prob=p["motor"].dropout_prob,
        ),
        wind=WindChaosConfig(
            enabled=p["wind"].enabled,
            steady_force=list(p["wind"].steady_force),
            turbulence_std=p["wind"].turbulence_std,
            gust_active=p["wind"].gust_active,
            gust_force=p["wind"].gust_force,
            gust_direction=list(p["wind"].gust_direction),
            gust_duration_s=p["wind"].gust_duration_s,
        ),
        payload=PayloadChaosConfig(
            attached=p["payload"].attached,
            mass=p["payload"].mass,
            kick_angle_rad=p["payload"].kick_angle_rad,
            kick_velocity=p["payload"].kick_velocity,
            aerodynamic_drag_enabled=p["payload"].aerodynamic_drag_enabled,
        ),
        sensor=SensorChaosConfig(
            enabled=p["sensor"].enabled,
            gyro_noise_std=p["sensor"].gyro_noise_std,
            accel_noise_std=p["sensor"].accel_noise_std,
            gyro_bias=list(p["sensor"].gyro_bias),
            harmonic_freq_hz=p["sensor"].harmonic_freq_hz,
            harmonic_amplitude=p["sensor"].harmonic_amplitude,
        ),
    )
