"""Unified Configuration Hub for System-1 and System-2 Runtime.

This package provides centralized access to all tunable parameters across the
entire multibody simulation and fault-tolerance architecture:

1. System-2 & Coordination Configurations:
   - FastTriggerConfig: High-frequency thresholds to wake up System-2
   - TelemetrySummaryConfig: 1.0s sliding-window and instant metric settings
   - PatchApplicationConfig: Hot-swap ramp-in apply time, TTL, and clamping limits
   - System2InferenceConfig: LLM provider, model, temperature, token limits, and prompt
   - System2WorkerConfig: Cooldown, attention thresholds, calm-hover, and crash limits
   - System2MasterConfig: Central container uniting all System-2 parameters

2. System-1 & Physics Configurations:
   - DroneConfig: Physical drone mass, inertia, rotor thrust bounds, mixer matrix
   - PIDConfig: Classical PID reflex gains for altitude, attitude, and horizontal hold
   - L1Config: SE(3) geometric gains, L1 Hurwitz adaptation, and filter bandwidths
   - SimConfig: MuJoCo physics timestep (500 Hz), render FPS, camera tracking
   - ChaosConfig: Baseline motor cuts, crosswinds, cargo payload, and sensor chatter

Usage:
------
    from runtime.config import System2MasterConfig, L1Config, DroneConfig

    # 1. Inspect or tweak System-2 defaults from one place:
    s2_cfg = System2MasterConfig()
    s2_cfg.trigger.pos_error_threshold = 0.18
    s2_cfg.patch.ramp_duration_s = 0.3
    s2_cfg.save_json("my_tuned_s2_config.json")

    # 2. Load custom tuned parameters for benchmarks:
    tuned_s2 = System2MasterConfig.load_json("my_tuned_s2_config.json")
"""

from runtime.config.system2_config import (
    FastTriggerConfig,
    TelemetrySummaryConfig,
    PatchApplicationConfig,
    System2InferenceConfig,
    System2WorkerConfig,
    System2MasterConfig,
)
from runtime.sim.config import (
    DroneConfig,
    PIDConfig,
    L1Config,
    SimConfig,
    ChaosConfig,
    SCENE_PATH,
    DUAL_SCENE_PATH,
    get_system_naming,
    get_drone_orientation,
)

__all__ = [
    # System 2 Configs
    "FastTriggerConfig",
    "TelemetrySummaryConfig",
    "PatchApplicationConfig",
    "System2InferenceConfig",
    "System2WorkerConfig",
    "System2MasterConfig",
    # System 1 & Physics Configs
    "DroneConfig",
    "PIDConfig",
    "L1Config",
    "SimConfig",
    "ChaosConfig",
    "SCENE_PATH",
    "DUAL_SCENE_PATH",
    "get_system_naming",
    "get_drone_orientation",
]

