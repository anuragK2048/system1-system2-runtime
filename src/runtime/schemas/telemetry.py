"""Schemas for Telemetry Summaries and Fast Trigger Events.

Pre-processed sliding window metrics and fast trigger detection for System 2.
Directly aligns with specifications in .anurag/tier_2_lifecycle.md.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FastTriggerEvent:
    """Fast Trigger event produced at 500 Hz when stability or error limits are exceeded."""
    timestamp: float
    reasons: List[str]
    pos_error_norm: float
    tilt_deg: float
    gyro_norm: float
    force_residual_norm: float
    force_residual_vec: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    moment_residual_vec: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def format_trigger_header(self) -> str:
        """Formats the trigger header for LLM prompts."""
        reasons_str = "; ".join(self.reasons)
        fx, fy, fz = self.force_residual_vec
        f_vec_str = f", F_dist=[{fx:+.1f},{fy:+.1f},{fz:+.1f}]N" if (abs(fx) > 0.5 or abs(fy) > 0.5 or abs(fz) > 0.5) else f", |F_dist|={self.force_residual_norm:.1f}N"
        return (
            f"TRIGGER: {reasons_str} "
            f"(pos_err={self.pos_error_norm:.2f}m, tilt={self.tilt_deg:.1f}°, "
            f"|gyro|={self.gyro_norm:.2f}rad/s{f_vec_str})"
        )


@dataclass
class TelemetrySummary:
    """Pre-processed physical summary aggregated over a sliding window (e.g. 1.0s).

    Translates raw 500 Hz physics into semantic insights the LLM can reason about:
    1. Human-Readable Orientation & Rates (Euler deg, gyro RMS, sign-flip oscillation)
    2. Tracking Error & Drift Vector (world mean pos error, velocity)
    3. Actuator Effort & Saturation (mean motor Newtons, saturation %)
    4. Estimated Unmodeled Force & Moment (L1 disturbance observer estimates)
    """
    timestamp: float
    window_duration_s: float
    samples_count: int

    # 1. Orientation & Rates
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    gyro_rms: List[float]  # [wx_rms, wy_rms, wz_rms]
    oscillation_detected: bool
    oscillation_note: str

    # 2. Tracking Error & Drift
    pos_error: List[float]  # [dx, dy, dz] in world frame (m)
    velocity: List[float]   # [vx, vy, vz] in world frame (m/s)

    # 3. Actuator Effort & Saturation
    motor_commands_mean: List[float]  # [m1, m2, m3, m4] in Newtons
    motor_saturation_pct: List[float] # % of time at 0N or 13N

    # 4. Estimated Unmodeled Disturbance (from L1 disturbance observer)
    unmodeled_force_disturbance: List[float]   # [Fx, Fy, Fz] in Newtons (1.0s mean)
    unmodeled_moment_disturbance: List[float]  # [Mx, My, Mz] in N*m (1.0s mean)
    instantaneous_force_disturbance: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def format_llm_text(self, trigger_header: Optional[str] = None) -> str:
        """Produces the exact prompt text format described in .anurag/tier_2_lifecycle.md."""
        lines = []
        if trigger_header:
            lines.append(trigger_header)

        lines.append(f"LAST {self.window_duration_s:.1f}s TELEMETRY SUMMARY:")
        dx, dy, dz = self.pos_error
        lines.append(f"- Position Error (World): dx = {dx:+.2f}m, dy = {dy:+.2f}m, dz = {dz:+.2f}m")

        vx, vy, vz = self.velocity
        lines.append(f"- Velocity (World): vx = {vx:+.2f} m/s, vy = {vy:+.2f} m/s, vz = {vz:+.2f} m/s")

        lines.append(
            f"- Attitude (Euler): Roll = {self.roll_deg:+.1f}°, Pitch = {self.pitch_deg:+.1f}°, Yaw = {self.yaw_deg:+.1f}°"
        )
        wx_rms, wy_rms, wz_rms = self.gyro_rms
        osc_str = f" ({self.oscillation_note})" if self.oscillation_detected else ""
        lines.append(
            f"- Gyro RMS: wx = {wx_rms:.2f} rad/s, wy = {wy_rms:.2f} rad/s, wz = {wz_rms:.2f} rad/s{osc_str}"
        )

        m1, m2, m3, m4 = self.motor_commands_mean
        s1, s2, s3, s4 = self.motor_saturation_pct
        lines.append(
            f"- Motor Controls (0-13N): M1={m1:.1f}N ({s1:.0f}% sat), M2={m2:.1f}N ({s2:.0f}% sat), "
            f"M3={m3:.1f}N ({s3:.0f}% sat), M4={m4:.1f}N ({s4:.0f}% sat)"
        )

        fx, fy, fz = self.unmodeled_force_disturbance
        cur_fx, cur_fy, cur_fz = getattr(self, "instantaneous_force_disturbance", [fx, fy, fz])
        mx, my, mz = self.unmodeled_moment_disturbance
        # In ENU coordinates, downward gravity load is negative Fz
        f_ext_z = -abs(fz) if fz > 0.5 else fz
        cur_f_ext_z = -abs(cur_fz) if cur_fz > 0.5 else cur_fz

        if abs(cur_fx - fx) > 1.0 or abs(cur_fy - fy) > 1.0 or abs(cur_fz - fz) > 1.0:
            lines.append(
                f"- External Disturbance Force on Drone: Current=[Fx={cur_fx:+.2f}N, Fy={cur_fy:+.2f}N, Fz={cur_f_ext_z:+.2f}N], "
                f"1.0s Mean=[Fx={fx:+.2f}N, Fy={fy:+.2f}N, Fz={f_ext_z:+.2f}N]"
                f"{' (downward payload/gravity load)' if cur_f_ext_z < -0.5 else ''}"
            )
        else:
            lines.append(
                f"- External Disturbance Force on Drone: Fx = {fx:+.2f}N, Fy = {fy:+.2f}N, Fz = {f_ext_z:+.2f}N"
                f"{' (downward payload/gravity load)' if f_ext_z < -0.5 else ''}"
            )
        lines.append(
            f"- External Disturbance Moment: Mx = {mx:+.3f}Nm, My = {my:+.3f}Nm, Mz = {mz:+.3f}Nm"
        )
        return "\n".join(lines)

    @property
    def estimated_altitude(self) -> float:
        """Estimated altitude in world frame (m) based on 1.0m nominal hover target."""
        return float(1.0 - self.pos_error[2])

    @property
    def tilt_deg(self) -> float:
        """Estimated total tilt angle magnitude in degrees."""
        return float((self.roll_deg**2 + self.pitch_deg**2) ** 0.5)

    @property
    def tilt_deg_mean(self) -> float:
        """Alias for tilt_deg."""
        return self.tilt_deg

    @property
    def pos_mean(self) -> List[float]:
        """Estimated position [x, y, z] in world frame based on nominal hover target."""
        return [float(-self.pos_error[0]), float(-self.pos_error[1]), float(1.0 - self.pos_error[2])]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "window_duration_s": self.window_duration_s,
            "samples_count": self.samples_count,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "gyro_rms": self.gyro_rms,
            "oscillation_detected": self.oscillation_detected,
            "oscillation_note": self.oscillation_note,
            "pos_error": self.pos_error,
            "velocity": self.velocity,
            "motor_commands_mean": self.motor_commands_mean,
            "motor_saturation_pct": self.motor_saturation_pct,
            "unmodeled_force_disturbance": self.unmodeled_force_disturbance,
            "unmodeled_moment_disturbance": self.unmodeled_moment_disturbance,
        }

