"""System-2 (LLM Meta-Controller) Configuration Architecture.

This module provides a unified, cohesive parameter specification for all
components involved in System-2 fault-tolerance control and its coordination
with System-1:

1. FastTriggerConfig:
   Controls when 500 Hz reflex loops wake up the asynchronous LLM reasoner.
2. TelemetrySummaryConfig:
   Controls how raw 500 Hz physics state is aggregated into semantic sliding-window summaries.
3. PatchApplicationConfig:
   Controls how and when LLM patches are injected into System-1 (hot-swap ramp-in apply time,
   patch TTL expiration, and safety clamping bounds).
4. System2InferenceConfig:
   Controls LLM provider, model, temperature, token limits, system prompt, and reasoning parameters.
5. System2WorkerConfig:
   Controls the background reasoning thread lifecycle, attention thresholds, calm-hover detection,
   and crash guards.
6. System2MasterConfig:
   Master aggregator combining all sub-configs for one-stop tweaking, automated grid search,
   and benchmark evaluation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


# ==============================================================================
# 1. FAST TRIGGER CONFIGURATION
# ==============================================================================
@dataclass
class FastTriggerConfig:
    """Thresholds evaluated at 500 Hz by the bridge to alert System-2.

    When any metric exceeds its threshold, a FastTriggerEvent is generated and
    the System-2 worker is signaled.
    """
    # Position tracking error threshold in meters (||p_target - p_actual||)
    pos_error_threshold: float = 0.30

    # Total inclination tilt angle threshold in degrees
    tilt_deg_threshold: float = 25.0

    # Angular rate norm threshold in rad/s (||omega||)
    gyro_norm_threshold: float = 3.5

    # L1 adaptive observer disturbance force norm threshold in Newtons (||sigma_f_hat||)
    force_residual_threshold: float = 3.0

    # Minimum wall-clock interval (seconds) between stored trigger events in bridge
    trigger_cooldown_s: float = 0.5

    # Minimum simulation time (seconds) before triggers are allowed (skips initial spawn)
    min_sim_time: float = 1.0

    # Maximum 500 Hz telemetry sample frames stored in the ring buffer (1500 = 3.0s)
    max_buffer_size: int = 1500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastTriggerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 2. TELEMETRY SUMMARY CONFIGURATION
# ==============================================================================
@dataclass
class TelemetrySummaryConfig:
    """Parameters for aggregating high-frequency 500 Hz frames into semantic summaries.

    These parameters dictate what physical insights are fed to the LLM.
    """
    # Sliding window length in seconds over which statistics are computed
    window_seconds: float = 1.0

    # Number of most recent samples used for instantaneous force calculations
    # (25 samples at 500 Hz = 50 ms window, useful for detecting rapid gust onset)
    recent_instant_samples: int = 25

    # Actuator lower saturation threshold in Newtons (<= 0.05 N considered saturated off)
    motor_saturation_min: float = 0.05

    # Actuator upper saturation threshold in Newtons (>= 12.95 N considered saturated max)
    motor_saturation_max: float = 12.95

    # Frequency threshold in Hz for sign flips in pitch/roll rate to qualify as oscillation
    oscillation_min_freq_hz: float = 3.0

    # Gyro RMS threshold in rad/s required alongside frequency to flag attitude oscillation
    oscillation_min_gyro_rms: float = 2.0

    # Whether to format instantaneous vs 1.0s mean force comparisons in LLM prompt text
    include_instantaneous_force: bool = True

    # Whether to include active patch details in the prompt text
    include_active_patch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetrySummaryConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 3. PATCH APPLICATION & HOT-SWAP CONFIGURATION
# ==============================================================================
@dataclass
class PatchApplicationConfig:
    """Controls how and when System-2 control patches are injected into System-1.

    Supports hot-swap ramp-in transitions, time-to-live expiration, and strict
    clamping bounds to prevent hallucinated extreme values from destabilizing the drone.
    """
    # Hot-swap ramp duration in seconds:
    #   0.0 = instant step injection (current default)
    #   >0.0 = smooth linear fade-in from 0% to 100% over ramp_duration_s
    ramp_duration_s: float = 0.0

    # Patch Time-To-Live in seconds: if set, patch auto-expires back to nominal
    # after this duration if no refresh arrives from System-2 (None = no expiration)
    patch_ttl_s: Optional[float] = None

    # Whether to scale feedforward force and motor trims by patch.confidence (0.0 to 1.0)
    confidence_scaling: bool = False

    # --------------------------------------------------------------------------
    # Output Parameter Bounds & Safety Clamping
    # --------------------------------------------------------------------------
    # Feedforward force offset bounds [min, max] in Newtons per axis
    feedforward_clamp_min: float = -10.0
    feedforward_clamp_max: float = 10.0

    # Per-motor trim multiplier bounds [min, max] (1.0 = nominal, 0.4 = degraded)
    motor_trim_clamp_min: float = 0.0
    motor_trim_clamp_max: float = 2.0

    # Gain multiplier bounds [min, max] (applied to pos_kp, pos_kd, att_kp, att_kd)
    gain_multiplier_clamp_min: float = 0.1
    gain_multiplier_clamp_max: float = 3.0

    # Commanded maximum tilt limit bounds [min, max] in degrees
    max_tilt_deg_clamp_min: float = 5.0
    max_tilt_deg_clamp_max: float = 45.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchApplicationConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 4. SYSTEM 2 LLM INFERENCE CONFIGURATION
# ==============================================================================
@dataclass
class System2InferenceConfig:
    """LLM provider, model, generation, and prompt parameters."""

    # Provider override: 'groq', 'baseten', 'fireworks', 'openai' (None = env default)
    provider: Optional[str] = None

    # Model identifier override (None = env default for provider)
    model: Optional[str] = None

    # Sampling temperature (lower = more deterministic, 0.0 for pure greedy)
    temperature: float = 0.2

    # Maximum completion tokens for Groq endpoint
    groq_max_completion_tokens: int = 2048

    # Maximum tokens for general OpenAI-compatible endpoints
    general_max_tokens: int = 4096

    # API call timeout in seconds
    timeout_s: float = 10.0

    # Random seed for reproducible completions (where supported)
    seed: Optional[int] = None

    # Custom system prompt override (None = use canonical default SYSTEM_PROMPT)
    system_prompt: Optional[str] = None

    # Enforce JSON object response format
    enable_json_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "System2InferenceConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 5. SYSTEM 2 WORKER LIFECYCLE CONFIGURATION
# ==============================================================================
@dataclass
class System2WorkerConfig:
    """Parameters governing the background System-2 reasoning thread.

    Dictates when to consult the LLM, when to hold an active patch, when to re-consult
    for revisions, and when to halt reasoning upon crash.
    """
    # Minimum cooldown interval (seconds) between successive LLM reasoning calls
    cooldown_s: float = 2.0

    # Polling sleep interval in seconds for the background worker thread
    loop_poll_interval_s: float = 0.05

    # Minimum simulation timestamp (seconds) before processing events
    warmup_min_timestamp: float = 1.0

    # Minimum ring-buffer sample count required before processing events
    warmup_min_samples: int = 200

    # --------------------------------------------------------------------------
    # CASE 1: Attention Thresholds (No active patch; deciding to wake up LLM)
    # --------------------------------------------------------------------------
    attention_pos_error: float = 0.30
    attention_force_disturbance: float = 2.5
    attention_moment_disturbance: float = 0.35
    attention_oscillation: bool = True

    # --------------------------------------------------------------------------
    # CASE 2: Calm Hover Thresholds (Active patch; deciding to clear patch)
    # --------------------------------------------------------------------------
    calm_pos_error: float = 0.15
    calm_force_disturbance: float = 1.0
    calm_moment_disturbance: float = 0.20
    calm_min_settle_time_s: float = 2.5

    # --------------------------------------------------------------------------
    # CASE 2: Patch Revision Thresholds (Active patch insufficient; re-calling LLM)
    # --------------------------------------------------------------------------
    reconsult_pos_error: float = 0.75
    reconsult_oscillation_age_s: float = 4.0
    reconsult_min_patch_age_s: float = 6.0
    stable_pos_error: float = 0.30

    # --------------------------------------------------------------------------
    # Crash Guard Limits (Halting LLM calls when ground contact / tumble detected)
    # --------------------------------------------------------------------------
    crash_alt_min: float = 0.20
    crash_tilt_near_ground_alt: float = 0.40
    crash_tilt_near_ground_deg: float = 55.0
    crash_tilt_extreme_deg: float = 80.0
    crash_runaway_pos_error: float = 4.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "System2WorkerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ==============================================================================
# 6. SYSTEM 2 MASTER CONFIGURATION (Central Hub)
# ==============================================================================
@dataclass
class System2MasterConfig:
    """Master configuration container aggregating all System-2 and coordination parameters.

    Use this class to:
    1. Inspect all System-2 parameters from one central point.
    2. Save and load experiment configurations to/from JSON.
    3. Programmatically iterate over parameter combinations for automated fine-tuning.
    """
    trigger: FastTriggerConfig = field(default_factory=FastTriggerConfig)
    telemetry: TelemetrySummaryConfig = field(default_factory=TelemetrySummaryConfig)
    patch: PatchApplicationConfig = field(default_factory=PatchApplicationConfig)
    inference: System2InferenceConfig = field(default_factory=System2InferenceConfig)
    worker: System2WorkerConfig = field(default_factory=System2WorkerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entire System-2 configuration to nested dictionary."""
        return {
            "trigger": self.trigger.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "patch": self.patch.to_dict(),
            "inference": self.inference.to_dict(),
            "worker": self.worker.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "System2MasterConfig":
        """Instantiates complete configuration hierarchy from dictionary."""
        cfg = cls()
        if "trigger" in data and isinstance(data["trigger"], dict):
            cfg.trigger = FastTriggerConfig.from_dict(data["trigger"])
        if "telemetry" in data and isinstance(data["telemetry"], dict):
            cfg.telemetry = TelemetrySummaryConfig.from_dict(data["telemetry"])
        if "patch" in data and isinstance(data["patch"], dict):
            cfg.patch = PatchApplicationConfig.from_dict(data["patch"])
        if "inference" in data and isinstance(data["inference"], dict):
            cfg.inference = System2InferenceConfig.from_dict(data["inference"])
        if "worker" in data and isinstance(data["worker"], dict):
            cfg.worker = System2WorkerConfig.from_dict(data["worker"])
        return cfg

    def save_json(self, path: Path | str) -> None:
        """Saves current configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> "System2MasterConfig":
        """Loads configuration from a JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

