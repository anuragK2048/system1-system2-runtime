"""Extensible Evaluation Scenarios and Auto-Discovery Registry.

Provides a unified protocol for defining and retrieving chaos scenarios for benchmarking.
Any preset defined in `runtime.chaos.presets` is automatically discovered and registered
with dynamic severity scaling support. New or custom chaos models can also be registered
directly with zero modifications to the evaluation harness or benchmark suite.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import copy

from runtime.chaos.models import DroneChaosState
from runtime.chaos.presets import CHAOS_PRESETS, get_preset_state, PRESET_ALIASES


@dataclass
class EvalScenario:
    """Self-contained description of one evaluation scenario.

    Attributes:
        id: Unique identifier (e.g., 'microburst_wind', 'chipped_propeller').
        display_name: Human-readable label for terminal and reports.
        description: One-sentence explanation of what physical faults are tested.
        make_state: Callable taking a `severity_scale: float` (default 1.0) and returning
            a fully configured `DroneChaosState`.
        duration_s: Recommended evaluation duration under disturbance (seconds).
        warmup_s: Warmup duration to allow clean takeoff before disturbance onset.
        tags: Categorization tags (e.g., ['wind', 'actuator', 'aerodynamics', 'compound']).
        metric_weights: Optional custom metric weights for composite scoring.
    """

    id: str
    display_name: str
    description: str
    make_state: Callable[[float], DroneChaosState]
    duration_s: float = 12.0
    warmup_s: float = 1.5
    tags: List[str] = field(default_factory=list)
    metric_weights: Dict[str, float] = field(default_factory=dict)

    def instantiate(self, severity: float = 1.0) -> DroneChaosState:
        """Instantiates the chaos state for this scenario scaled by severity."""
        return self.make_state(float(severity))


# ==============================================================================
# GLOBAL SCENARIO REGISTRY
# ==============================================================================
class ScenarioRegistry:
    """Central registry holding all known evaluation scenarios."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, EvalScenario] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, scenario: EvalScenario, aliases: Optional[List[str]] = None) -> None:
        """Registers a new evaluation scenario."""
        canon_id = scenario.id.lower().strip()
        self._scenarios[canon_id] = scenario
        self._aliases[canon_id] = canon_id
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower().strip()] = canon_id

    def get(self, identifier: str) -> EvalScenario:
        """Retrieves a scenario by ID or alias."""
        clean_id = str(identifier).lower().strip()
        canon_id = self._aliases.get(clean_id, clean_id)
        if canon_id not in self._scenarios:
            known = list(self._scenarios.keys())
            raise KeyError(f"Unknown scenario '{identifier}'. Registered scenarios: {known}")
        return self._scenarios[canon_id]

    def list_all(self, tags: Optional[List[str]] = None) -> List[EvalScenario]:
        """Returns all registered scenarios, optionally filtered by tag."""
        if not tags:
            return list(self._scenarios.values())
        tag_set = {t.lower().strip() for t in tags}
        return [
            s for s in self._scenarios.values()
            if any(t.lower() in tag_set for t in s.tags)
        ]

    def contains(self, identifier: str) -> bool:
        clean_id = str(identifier).lower().strip()
        return clean_id in self._aliases or clean_id in self._scenarios


_GLOBAL_REGISTRY = ScenarioRegistry()


def register_scenario(scenario: EvalScenario, aliases: Optional[List[str]] = None) -> None:
    """Convenience helper to register an evaluation scenario into the global registry."""
    _GLOBAL_REGISTRY.register(scenario, aliases=aliases)


def get_scenario(identifier: str) -> EvalScenario:
    """Retrieves an EvalScenario from the global registry."""
    return _GLOBAL_REGISTRY.get(identifier)


def list_scenarios(tags: Optional[List[str]] = None) -> List[EvalScenario]:
    """Lists all registered EvalScenarios."""
    return _GLOBAL_REGISTRY.list_all(tags=tags)


# ==============================================================================
# SEVERITY SCALING ENGINE (Works on any current & future DroneChaosState)
# ==============================================================================
def create_preset_severity_scaler(preset_key: str) -> Callable[[float], DroneChaosState]:
    """Builds a dynamic state generator that scales physical severity.

    Severity scaling mechanics:
      - severity = 1.0: Exact baseline preset parameters.
      - severity < 1.0: Proportionally softened perturbations (finds sensitivity onset).
      - severity > 1.0: Linearly amplified disturbances (discovers failure/breaking points).

    Scaling covers all 4 standard physical disturbance channels:
      1. Actuators: Scales the thrust degradation deficit (1.0 - multiplier) * severity.
      2. Wind: Scales steady wind vector norm and Dryden turbulence standard deviation.
      3. Payload: Scales pendulum bob mass and kick velocity.
      4. Sensor: Scales IMU gyro/accel noise standard deviation and harmonic resonance.
    """

    def make_state(severity: float) -> DroneChaosState:
        s = max(0.0, float(severity))
        state = get_preset_state(preset_key)

        # 1. Motor Degradation Scaling
        if state.motor.is_active():
            new_mults = []
            for m in state.motor.multipliers:
                if m < 0.999:  # Degraded motor
                    deficit = 1.0 - m
                    # Higher severity increases deficit (lowering available thrust)
                    scaled_thrust = max(0.05, 1.0 - deficit * s)
                    new_mults.append(float(scaled_thrust))
                else:
                    new_mults.append(1.0)
            state.motor.multipliers = new_mults

        # 2. Wind Force Scaling
        if state.wind.enabled:
            state.wind.steady_force = [float(f * s) for f in state.wind.steady_force]
            state.wind.turbulence_std = float(state.wind.turbulence_std * s)
            if state.wind.gust_active:
                state.wind.gust_force = float(state.wind.gust_force * s)

        # 3. Payload Mass & Kick Scaling
        if state.payload.attached:
            state.payload.mass = float(max(0.05, state.payload.mass * s))
            if state.payload.kick_velocity > 0:
                state.payload.kick_velocity = float(state.payload.kick_velocity * s)

        # 4. Sensor Chatter Scaling
        if state.sensor.enabled:
            state.sensor.gyro_noise_std = float(state.sensor.gyro_noise_std * s)
            state.sensor.accel_noise_std = float(state.sensor.accel_noise_std * s)
            state.sensor.harmonic_amplitude = float(state.sensor.harmonic_amplitude * s)

        return state

    return make_state


# ==============================================================================
# AUTO-DISCOVERY FROM runtime.chaos.presets
# ==============================================================================
def _auto_discover_presets() -> None:
    """Scans all presets in CHAOS_PRESETS and registers them automatically."""
    for key, p in CHAOS_PRESETS.items():
        if key in ("0", "clean", "reset"):
            continue

        preset_id = str(p["id"])
        name = str(p["name"])
        desc = str(p["description"])

        # Tag heuristics based on preset structure
        tags: List[str] = []
        if p["motor"].is_active():
            tags.append("motor")
        if p["wind"].enabled:
            tags.append("wind")
        if p["payload"].attached:
            tags.append("payload")
        if p["sensor"].enabled:
            tags.append("sensor")
        if len(tags) > 1:
            tags.append("compound")

        scenario = EvalScenario(
            id=preset_id,
            display_name=name,
            description=desc,
            make_state=create_preset_severity_scaler(key),
            duration_s=12.0,
            warmup_s=1.5,
            tags=tags,
        )

        # Collect aliases for this preset
        aliases = [key]
        for alias, canon in PRESET_ALIASES.items():
            if canon == key and alias != key:
                aliases.append(alias)

        register_scenario(scenario, aliases=aliases)


_auto_discover_presets()
