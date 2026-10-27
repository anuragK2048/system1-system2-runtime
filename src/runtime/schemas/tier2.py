"""Schemas for System 2 (LLM Meta-Controller) Coordination.

Based on the System 1 / System 2 architecture specification in:
.anurag/tier_2_lifecycle.md
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


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

    Corresponds to the 4 control levers:
    1. feedforward_force_offset: [dFx, dFy, dFz] in Newtons
    2. gain_multipliers: {pos_kp, pos_kd, att_kp, att_kd}
    3. motor_trim_scale: [s1, s2, s3, s4] (e.g. [1.4, 1.0, 1.0, 1.0] for chipped prop)
    4. safety_envelope: {max_tilt_deg, temp_setpoint_offset}
    """
    diagnosis: str = "Nominal"
    feedforward_force_offset: Optional[List[float]] = None
    gain_multipliers: Optional[GainMultipliers] = None
    motor_trim_scale: Optional[List[float]] = None
    safety_envelope: Optional[SafetyEnvelope] = None
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

        ff = data.get("feedforward_force_offset")
        if ff is not None:
            ff = [float(x) for x in ff]

        trim = data.get("motor_trim_scale")
        if trim is not None:
            trim = [float(x) for x in trim]

        return cls(
            diagnosis=str(data.get("diagnosis", "")),
            feedforward_force_offset=ff,
            gain_multipliers=gains,
            motor_trim_scale=trim,
            safety_envelope=envelope,
            confidence=float(data.get("confidence", 1.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "diagnosis": self.diagnosis,
            "feedforward_force_offset": self.feedforward_force_offset,
            "gain_multipliers": self.gain_multipliers.to_dict() if self.gain_multipliers else None,
            "motor_trim_scale": self.motor_trim_scale,
            "safety_envelope": self.safety_envelope.to_dict() if self.safety_envelope else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
