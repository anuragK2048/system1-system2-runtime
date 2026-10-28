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
