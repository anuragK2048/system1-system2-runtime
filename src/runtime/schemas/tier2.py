"""Schemas for System 2 (LLM Meta-Controller) Coordination.

Based on the System 1 / System 2 architecture specification in:
.anurag/tier_2_lifecycle.md
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import math
import numpy as np


# ==============================================================================
# F2. LLM PATCH SAFETY LIMITS
# ==============================================================================
# Bounds applied by validate_patch() before any LLM patch reaches System 1.
# Confirmed against the existing prompt examples (sensor-noise att_kp 0.40,
# att_kd 1.6; motor-failure trim 2.50; wind feedforward +/-5 N) so the advised
# patches pass through unchanged.
FEEDFORWARD_MAX_PER_AXIS_N: float = 6.0  # |Fx|, |Fy|, |Fz| each <= 6 N
GAIN_MULT_MIN: float = 0.3
GAIN_MULT_MAX: float = 2.0
MOTOR_TRIM_MIN: float = 0.5
MOTOR_TRIM_MAX: float = 2.5
SAFETY_MAX_TILT_DEG: float = 25.0  # upper bound; must also be positive
WAYPOINT_MAX_VELOCITY: float = 5.0
WAYPOINT_MIN_VELOCITY: float = 0.1
WAYPOINT_MAX_CORRIDOR: float = 5.0
WAYPOINT_MIN_CORRIDOR: float = 0.1


@dataclass
class WaypointCommand:
    """System 2 autonomous waypoint guidance command."""
    mode: str = "relative_body"  # "relative_body" or "world"
    target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [dx, dy, dz] or [x, y, z]
    yaw_deg: float = 0.0
    max_velocity: float = 1.0
    corridor_width: float = 0.6

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "target": [float(x) for x in self.target],
            "yaw_deg": float(self.yaw_deg),
            "max_velocity": float(self.max_velocity),
            "corridor_width": float(self.corridor_width),
        }


@dataclass
class GainMultipliers:
    """Gain multipliers output by System 2 to dynamically retune System 1 loops."""
    pos_kp: float = 1.0
    pos_kd: float = 1.0
    att_kp: float = 1.0
    att_kd: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "pos_kp": float(self.pos_kp),
            "pos_kd": float(self.pos_kd),
            "att_kp": float(self.att_kp),
            "att_kd": float(self.att_kd),
        }


@dataclass
class SafetyEnvelope:
    """Safety constraints commanded by System 2 during recovery."""
    max_tilt_deg: float = 25.0
    temp_setpoint_offset: Optional[List[float]] = None  # [dx, dy, dz] temp offset

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_tilt_deg": float(self.max_tilt_deg),
            "temp_setpoint_offset": self.temp_setpoint_offset,
        }


@dataclass
class System2Patch:
    """Structured patch emitted by System 2 to adjust System 1.

    Corresponds to the 5 control levers:
    1. feedforward_force_offset: [dFx, dFy, dFz] in Newtons
    2. gain_multipliers: {pos_kp, pos_kd, att_kp, att_kd}
    3. motor_trim_scale: [s1, s2, s3, s4] (e.g. [1.4, 1.0, 1.0, 1.0] for chipped prop)
    4. safety_envelope: {max_tilt_deg, temp_setpoint_offset}
    5. waypoint_command: {mode, target, yaw_deg, max_velocity, corridor_width}
    """
    diagnosis: str = "Nominal"
    feedforward_force_offset: Optional[List[float]] = None
    gain_multipliers: Optional[GainMultipliers] = None
    motor_trim_scale: Optional[List[float]] = None
    safety_envelope: Optional[SafetyEnvelope] = None
    waypoint_command: Optional[WaypointCommand] = None
    confidence: float = 1.0
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "System2Patch":
        """Instantiate patch from parsed LLM JSON dictionary."""
        gains = None
        if "gain_multipliers" in data and isinstance(data["gain_multipliers"], dict):
            gm = data["gain_multipliers"]
            gains = GainMultipliers(
                pos_kp=float(gm.get("pos_kp", 1.0)),
                pos_kd=float(gm.get("pos_kd", 1.0)),
                att_kp=float(gm.get("att_kp", 1.0)),
                att_kd=float(gm.get("att_kd", 1.0)),
            )

        envelope = None
        if "safety_envelope" in data and isinstance(data["safety_envelope"], dict):
            se = data["safety_envelope"]
            envelope = SafetyEnvelope(
                max_tilt_deg=float(se.get("max_tilt_deg", 25.0)),
                temp_setpoint_offset=se.get("temp_setpoint_offset"),
            )

        wp = None
        if "waypoint_command" in data and isinstance(data["waypoint_command"], dict):
            wc_data = data["waypoint_command"]
            t_raw = wc_data.get("target", [0.0, 0.0, 0.0])
            if not isinstance(t_raw, list):
                t_raw = list(t_raw)
            wp = WaypointCommand(
                mode=str(wc_data.get("mode", "relative_body")),
                target=[float(x) for x in t_raw],
                yaw_deg=float(wc_data.get("yaw_deg", 0.0)),
                max_velocity=float(wc_data.get("max_velocity", 1.0)),
                corridor_width=float(wc_data.get("corridor_width", 0.6)),
            )

        ff = data.get("feedforward_force_offset")
        if ff is not None:
            ff = [float(x) for x in ff]

        trim = data.get("motor_trim_scale")
        if trim is not None:
            trim = [float(x) for x in trim]

        patch = cls(
            diagnosis=str(data.get("diagnosis", "")),
            feedforward_force_offset=ff,
            gain_multipliers=gains,
            motor_trim_scale=trim,
            safety_envelope=envelope,
            waypoint_command=wp,
            confidence=float(data.get("confidence", 1.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )
        return validate_patch(patch)

    def to_dict(self) -> Dict[str, object]:
        return {
            "diagnosis": self.diagnosis,
            "feedforward_force_offset": self.feedforward_force_offset,
            "gain_multipliers": self.gain_multipliers.to_dict() if self.gain_multipliers else None,
            "motor_trim_scale": self.motor_trim_scale,
            "safety_envelope": self.safety_envelope.to_dict() if self.safety_envelope else None,
            "waypoint_command": self.waypoint_command.to_dict() if self.waypoint_command else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


def _require_number(value: object, name: str) -> float:
    """Coerces an LLM-provided value to a finite float, rejecting garbage.

    Raises:
        ValueError: If the value is non-numeric or non-finite (NaN/inf).
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"System2Patch rejected: '{name}' is non-numeric ({value!r})")
    if not math.isfinite(number):
        raise ValueError(f"System2Patch rejected: '{name}' is not finite ({value!r})")
    return number


def _clamp(number: float, lo: float, hi: float) -> float:
    """Clamps a number into [lo, hi]."""
    return max(lo, min(hi, number))


def validate_patch(patch: System2Patch) -> System2Patch:
    """Validates an LLM patch before it is applied to System 1 (F2).

    Out-of-range values are clamped to the F2 safety limits and the clamping
    is logged to the terminal. Patches with no diagnosis, non-numeric values,
    or wrong-length vectors are rejected with ValueError (callers treat that
    as "do not apply": System2Brain.analyze_and_patch catches it and returns
    None, System1System2Bridge.apply_system2_patch lets it propagate).

    Args:
        patch: The candidate System2Patch (validated in place, then returned).

    Returns:
        The same patch with all values finite and within limits.

    Raises:
        ValueError: If the patch must be rejected outright.
    """
    if not patch.diagnosis or not str(patch.diagnosis).strip():
        raise ValueError("System2Patch rejected: missing or empty 'diagnosis'")

    clamped_notes: List[str] = []

    if patch.feedforward_force_offset is not None:
        ff = patch.feedforward_force_offset
        if len(ff) != 3:
            raise ValueError(
                f"System2Patch rejected: 'feedforward_force_offset' must have 3 elements, got {len(ff)}"
            )
        for i in range(3):
            checked = _require_number(ff[i], f"feedforward_force_offset[{i}]")
            limited = _clamp(checked, -FEEDFORWARD_MAX_PER_AXIS_N, FEEDFORWARD_MAX_PER_AXIS_N)
            if limited != checked:
                clamped_notes.append(f"feedforward[{i}] {checked:+.2f} -> {limited:+.2f} N")
            ff[i] = limited

    if patch.gain_multipliers is not None:
        gm = patch.gain_multipliers
        for name in ("pos_kp", "pos_kd", "att_kp", "att_kd"):
            checked = _require_number(getattr(gm, name), f"gain_multipliers.{name}")
            limited = _clamp(checked, GAIN_MULT_MIN, GAIN_MULT_MAX)
            if limited != checked:
                clamped_notes.append(f"gain {name} {checked:.2f} -> {limited:.2f}")
            setattr(gm, name, limited)

    if patch.motor_trim_scale is not None:
        trim = patch.motor_trim_scale
        if len(trim) != 4:
            raise ValueError(
                f"System2Patch rejected: 'motor_trim_scale' must have 4 elements, got {len(trim)}"
            )
        for i in range(4):
            checked = _require_number(trim[i], f"motor_trim_scale[{i}]")
            limited = _clamp(checked, MOTOR_TRIM_MIN, MOTOR_TRIM_MAX)
            if limited != checked:
                clamped_notes.append(f"motor_trim[{i}] {checked:.2f} -> {limited:.2f}")
            trim[i] = limited

    if patch.safety_envelope is not None:
        se = patch.safety_envelope
        tilt = _require_number(se.max_tilt_deg, "safety_envelope.max_tilt_deg")
        if tilt <= 0.0:
            raise ValueError(
                f"System2Patch rejected: 'safety_envelope.max_tilt_deg' must be positive, got {tilt}"
            )
        if tilt > SAFETY_MAX_TILT_DEG:
            clamped_notes.append(f"max_tilt {tilt:.1f} -> {SAFETY_MAX_TILT_DEG:.1f} deg")
            tilt = SAFETY_MAX_TILT_DEG
        se.max_tilt_deg = tilt
        if se.temp_setpoint_offset is not None:
            off = se.temp_setpoint_offset
            if len(off) != 3:
                raise ValueError(
                    "System2Patch rejected: 'safety_envelope.temp_setpoint_offset' must have "
                    f"3 elements, got {len(off)}"
                )
            se.temp_setpoint_offset = [
                _require_number(v, f"safety_envelope.temp_setpoint_offset[{i}]")
                for i, v in enumerate(off)
            ]

    if patch.waypoint_command is not None:
        wc = patch.waypoint_command
        if wc.mode not in ("relative_body", "world"):
            raise ValueError(
                f"System2Patch rejected: 'waypoint_command.mode' must be 'relative_body' or 'world', got {wc.mode!r}"
            )
        if len(wc.target) != 3:
            raise ValueError(
                f"System2Patch rejected: 'waypoint_command.target' must have 3 elements, got {len(wc.target)}"
            )
        wc.target = [
            _require_number(wc.target[i], f"waypoint_command.target[{i}]")
            for i in range(3)
        ]
        wc.yaw_deg = _require_number(wc.yaw_deg, "waypoint_command.yaw_deg")
        raw_vel = _require_number(wc.max_velocity, "waypoint_command.max_velocity")
        limited_vel = _clamp(raw_vel, WAYPOINT_MIN_VELOCITY, WAYPOINT_MAX_VELOCITY)
        if limited_vel != raw_vel:
            clamped_notes.append(f"waypoint max_velocity {raw_vel:.2f} -> {limited_vel:.2f} m/s")
        wc.max_velocity = limited_vel

        raw_corr = _require_number(wc.corridor_width, "waypoint_command.corridor_width")
        limited_corr = _clamp(raw_corr, WAYPOINT_MIN_CORRIDOR, WAYPOINT_MAX_CORRIDOR)
        if limited_corr != raw_corr:
            clamped_notes.append(f"waypoint corridor_width {raw_corr:.2f} -> {limited_corr:.2f} m")
        wc.corridor_width = limited_corr

    if clamped_notes:
        print(f"⚠️ [SYSTEM 2 VALIDATE] Clamped out-of-range patch values ({patch.diagnosis}):")
        for note in clamped_notes:
            print(f"   • {note}")

    return patch
