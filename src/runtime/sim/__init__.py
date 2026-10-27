"""Simulation package for the Skydio X2 multibody chaos testbed.

Exports:
    DroneSimulator: Full MuJoCo simulation environment & 3D visualizer.
    System1Controller: High-frequency (500 Hz) PID reflex stabilizer.
    ChaosManager: Anomaly injector for prop cuts, payload, noise, and wind.
    SimConfig, DroneConfig, PIDConfig, ChaosConfig: Configuration dataclasses.
"""

from runtime.sim.config import (
    SimConfig,
    DroneConfig,
    PIDConfig,
    L1Config,
    ChaosConfig,
    SCENE_PATH,
)
from runtime.sim.system1_controller import System1Controller
from runtime.sim.chaos_injector import ChaosManager
from runtime.sim.simulator import DroneSimulator
from runtime.sim.bridge import System1System2Bridge

__all__ = [
    "DroneSimulator",
    "System1Controller",
    "ChaosManager",
    "System1System2Bridge",
    "SimConfig",
    "DroneConfig",
    "PIDConfig",
    "L1Config",
    "ChaosConfig",
    "SCENE_PATH",
]
