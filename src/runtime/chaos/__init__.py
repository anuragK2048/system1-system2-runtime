"""Chaos Injection Subsystem for Quadrotor Flight Resilience Benchmarking.

Provides:
- ChaosManager: Thread-safe runtime manager for physical faults and disturbances.
- ChaosTarget: Enumeration ('both', 'drone1', 'drone2').
- Models: MotorChaosConfig, WindChaosConfig, PayloadChaosConfig, SensorChaosConfig, DroneChaosState.
- Presets: Standard repeatable test scenarios (CHAOS_PRESETS, get_preset_state).
"""

from runtime.chaos.models import (
    ChaosTarget,
    MotorChaosConfig,
    WindChaosConfig,
    PayloadChaosConfig,
    SensorChaosConfig,
    DroneChaosState,
)
from runtime.chaos.presets import (
    CHAOS_PRESETS,
    PRESET_ALIASES,
    get_preset_state,
    get_preset_definition,
)
from runtime.chaos.manager import ChaosManager

__all__ = [
    "ChaosManager",
    "ChaosTarget",
    "MotorChaosConfig",
    "WindChaosConfig",
    "PayloadChaosConfig",
    "SensorChaosConfig",
    "DroneChaosState",
    "CHAOS_PRESETS",
    "PRESET_ALIASES",
    "get_preset_state",
    "get_preset_definition",
]
