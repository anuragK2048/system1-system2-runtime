"""Real-Time Thread-Safe System 1 / System 2 Coordination Bridge.

This bridge enables asynchronous, non-blocking communication between:
- System 1 (500 Hz fast reflex loop in MuJoCo physics thread)
- System 2 (1-5 Hz async LLM reasoning engine in background thread / task)

Implements:
1. Ring buffer for 500 Hz telemetry history (sliding window).
2. Fast Trigger detector (position error, attitude tilt, gyro rates, L1 innovation).
3. Sliding window summarizer producing structured TelemetrySummary.
4. Non-blocking patch repository for System 2 control adjustments.
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Deque
import numpy as np

from runtime.schemas.tier2 import System2Patch, GainMultipliers, SafetyEnvelope
from runtime.schemas.telemetry import FastTriggerEvent, TelemetrySummary


class System1System2Bridge:
    """Thread-safe real-time communication bridge between System 1 and System 2.

    Attributes:
        max_buffer_size: Maximum number of 500 Hz telemetry samples stored (default: 1000 = 2.0s).
        pos_error_threshold: Trigger limit for position error in meters (default: 0.30m).
        tilt_deg_threshold: Trigger limit for tilt angle in degrees (default: 25.0°).
        gyro_norm_threshold: Trigger limit for angular velocity norm in rad/s (default: 3.5 rad/s).
        force_residual_threshold: Trigger limit for L1 force disturbance in Newtons (default: 3.0N).
    """

    def __init__(
        self,
        max_buffer_size: int = 1500,  # 3.0 seconds at 500 Hz
        pos_error_threshold: float = 0.30,
        tilt_deg_threshold: float = 25.0,
        gyro_norm_threshold: float = 3.5,
        force_residual_threshold: float = 3.0,
        trigger_cooldown_s: float = 0.5,
    ) -> None:
        self.max_buffer_size = max_buffer_size
        self.pos_error_threshold = pos_error_threshold
        self.tilt_deg_threshold = tilt_deg_threshold
        self.gyro_norm_threshold = gyro_norm_threshold
        self.force_residual_threshold = force_residual_threshold
        self.trigger_cooldown_s = trigger_cooldown_s

        self._lock = threading.Lock()
        self._trigger_event = threading.Event()
        self._last_trigger_time = 0.0

        # Ring buffer storing dictionaries of sample points
        self._buffer: Deque[dict] = deque(maxlen=max_buffer_size)

        # Trigger event history queue (up to 50 events)
        self._trigger_history: Deque[FastTriggerEvent] = deque(maxlen=50)

        # Active System 2 Patch applied to System 1
        self._active_patch: Optional[System2Patch] = None
        self._patch_applied_time: Optional[float] = None

    # ==========================================================================
    # SYSTEM 1 FAST INTERFACE (Called every 2 ms from 500 Hz loop)
    # ==========================================================================
    def push_telemetry(
        self,
        sim_time: float,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        motor_ctrl: np.ndarray,
        target_pos: np.ndarray,
        sigma_f_hat: np.ndarray,
        sigma_M_hat: np.ndarray,
    ) -> Tuple[bool, List[str]]:
        """Appends current 500 Hz telemetry frame and evaluates Fast Trigger conditions.

        Non-blocking execution time: ~5-10 microseconds.

        Returns:
            Tuple of:
              - is_triggered (bool): True if any trigger threshold was exceeded.
              - reasons (List[str]): Explanations for why trigger fired.
        """
        # Calculate instant metrics
        pos_err_vec = target_pos - pos
        pos_err_norm = float(np.linalg.norm(pos_err_vec))

        # Euler angles from quaternion [w, x, y, z]
        w, qx, qy, qz = quat
        roll_deg = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
        yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))

        # Total tilt from vertical (z-axis inclination)
        tilt_deg = float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))))

        gyro_norm = float(np.linalg.norm(omega))
        force_res_norm = float(np.linalg.norm(sigma_f_hat))

        reasons: List[str] = []
        if pos_err_norm > self.pos_error_threshold:
            reasons.append(
                f"Position error ({pos_err_norm:.2f}m) exceeded threshold ({self.pos_error_threshold:.2f}m)"
            )
        if tilt_deg > self.tilt_deg_threshold:
            reasons.append(
                f"Tilt angle ({tilt_deg:.1f}°) exceeded safe limit ({self.tilt_deg_threshold:.1f}°)"
            )
        if gyro_norm > self.gyro_norm_threshold:
            reasons.append(
                f"Angular rate ({gyro_norm:.2f} rad/s) spiked above threshold ({self.gyro_norm_threshold:.2f} rad/s)"
            )
        if force_res_norm > self.force_residual_threshold:
            dominant_axis = "X" if abs(sigma_f_hat[0]) >= max(abs(sigma_f_hat[1]), abs(sigma_f_hat[2])) else ("Y" if abs(sigma_f_hat[1]) >= abs(sigma_f_hat[2]) else "Z")
            idx = {"X": 0, "Y": 1, "Z": 2}[dominant_axis]
            sign_str = "+" if sigma_f_hat[idx] >= 0 else "-"
            reasons.append(
                f"L1 estimated disturbance force ({force_res_norm:.1f}N along {sign_str}{dominant_axis}) spiked above threshold ({self.force_residual_threshold:.1f}N)"
            )

        # Ignore triggers during initial spawn ascent warmup or if drone has crashed on the ground
        is_crashed = (pos[2] < 0.12 and tilt_deg > 60.0) or (pos[2] < 0.08)
        is_triggered = (len(reasons) > 0) and (sim_time >= 1.0) and not is_crashed

        sample = {
            "time": sim_time,
            "pos": pos.copy(),
            "vel": vel.copy(),
            "quat": quat.copy(),
            "omega": omega.copy(),
            "ctrl": motor_ctrl.copy(),
            "target": target_pos.copy(),
            "pos_err": pos_err_vec.copy(),
            "pos_err_norm": pos_err_norm,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "tilt_deg": tilt_deg,
            "sigma_f_hat": sigma_f_hat.copy(),
            "sigma_M_hat": sigma_M_hat.copy(),
        }

        with self._lock:
            self._buffer.append(sample)

            if is_triggered:
                now = time.time()
                if now - self._last_trigger_time >= self.trigger_cooldown_s:
                    event = FastTriggerEvent(
                        timestamp=sim_time,
                        reasons=reasons,
                        pos_error_norm=pos_err_norm,
                        tilt_deg=tilt_deg,
                        gyro_norm=gyro_norm,
                        force_residual_norm=force_res_norm,
                        force_residual_vec=[float(x) for x in sigma_f_hat],
                        moment_residual_vec=[float(x) for x in sigma_M_hat],
                    )
                    self._trigger_history.append(event)
                    self._trigger_event.set()
                    self._last_trigger_time = now

        return is_triggered, reasons

    def get_active_patch(self) -> Optional[System2Patch]:
        """Thread-safe retrieval of the currently active System 2 patch."""
        with self._lock:
            return self._active_patch

    # ==========================================================================
    # SYSTEM 2 ASYNC INTERFACE (Called from LLM Reasoning thread)
    # ==========================================================================
    def wait_for_trigger(self, timeout: Optional[float] = None) -> bool:
        """Blocks until a Fast Trigger occurs, or until timeout expires."""
        signaled = self._trigger_event.wait(timeout=timeout)
        if signaled:
            self._trigger_event.clear()
        return signaled

    def is_triggered(self) -> bool:
        """Returns True if a trigger event is currently pending."""
        return self._trigger_event.is_set()

    def clear_trigger(self) -> None:
        """Clears the pending trigger event."""
        self._trigger_event.clear()

    def get_latest_trigger_event(self, max_age_s: float = 2.0, current_time: Optional[float] = None) -> Optional[FastTriggerEvent]:
        """Retrieves the most recent trigger event if recent."""
        with self._lock:
            if not self._trigger_history:
                return None
            event = self._trigger_history[-1]
            if current_time is not None and (current_time - event.timestamp) > max_age_s:
                return None
            return event

    def reset(self) -> None:
        """Resets the bridge history, triggers, and active patches."""
        with self._lock:
            self._buffer.clear()
            self._trigger_history.clear()
            self._trigger_event.clear()
            self._active_patch = None
            self._patch_applied_time = None

    def get_telemetry_summary(self, window_seconds: float = 1.0) -> TelemetrySummary:
        """Computes statistical telemetry summary over the sliding window.

        Args:
            window_seconds: Window length in seconds (default: 1.0s).

        Returns:
            TelemetrySummary object with ready-to-prompt physical metrics.
        """
        with self._lock:
            if not self._buffer:
                # Return empty summary if buffer is blank
                return TelemetrySummary(
                    timestamp=0.0,
                    window_duration_s=window_seconds,
                    samples_count=0,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    yaw_deg=0.0,
                    gyro_rms=[0.0, 0.0, 0.0],
                    oscillation_detected=False,
                    oscillation_note="No data",
                    pos_error=[0.0, 0.0, 0.0],
                    velocity=[0.0, 0.0, 0.0],
                    motor_commands_mean=[3.25, 3.25, 3.25, 3.25],
                    motor_saturation_pct=[0.0, 0.0, 0.0, 0.0],
                    unmodeled_force_disturbance=[0.0, 0.0, 0.0],
                    unmodeled_moment_disturbance=[0.0, 0.0, 0.0],
                )

            # Filter samples within the window
            latest_time = self._buffer[-1]["time"]
            window_start = latest_time - window_seconds
            samples = [s for s in self._buffer if s["time"] >= window_start]

            if not samples:
                samples = [self._buffer[-1]]

        n = len(samples)

        # 1. Orientation & Gyro Rates
        latest = samples[-1]
        roll_deg = latest["roll_deg"]
        pitch_deg = latest["pitch_deg"]
        yaw_deg = latest["yaw_deg"]

        omegas = np.array([s["omega"] for s in samples])  # [N, 3]
        gyro_rms = np.sqrt(np.mean(omegas**2, axis=0))

        # Check for sign-flips in pitch and roll rates (oscillation / jitter)
        pitch_sign_flips = int(np.count_nonzero(np.diff(np.signbit(omegas[:, 1]))))
        roll_sign_flips = int(np.count_nonzero(np.diff(np.signbit(omegas[:, 0]))))
        duration = max(0.01, samples[-1]["time"] - samples[0]["time"])
        pitch_freq = (pitch_sign_flips / 2.0) / duration
        roll_freq = (roll_sign_flips / 2.0) / duration

        oscillation_detected = False
        notes = []
        if pitch_freq > 3.0 and gyro_rms[1] > 2.0:
            oscillation_detected = True
            notes.append(f"pitch oscillation detected at ~{pitch_freq:.1f}Hz, wy_rms={gyro_rms[1]:.1f}rad/s")
        if roll_freq > 3.0 and gyro_rms[0] > 2.0:
            oscillation_detected = True
            notes.append(f"roll oscillation detected at ~{roll_freq:.1f}Hz, wx_rms={gyro_rms[0]:.1f}rad/s")
        oscillation_note = "; ".join(notes) if notes else "Attitude rates stable"

        # 2. Position Error & Velocity
        pos_errors = np.array([s["pos_err"] for s in samples])
        mean_pos_err = np.mean(pos_errors, axis=0)
        velocities = np.array([s["vel"] for s in samples])
        mean_vel = np.mean(velocities, axis=0)

        # 3. Motor Commands & Saturation
        ctrls = np.array([s["ctrl"] for s in samples])  # [N, 4]
        ctrl_mean = np.mean(ctrls, axis=0)
        # Saturation defined as ctrl <= 0.05 or ctrl >= 12.95
        sat_mask = (ctrls <= 0.05) | (ctrls >= 12.95)
        sat_pct = (np.sum(sat_mask, axis=0) / n) * 100.0

        # 4. Unmodeled Disturbance Estimates from L1
        sigma_f = np.mean(np.array([s["sigma_f_hat"] for s in samples]), axis=0)
        sigma_M = np.mean(np.array([s["sigma_M_hat"] for s in samples]), axis=0)

        # Recent instantaneous / 50ms window (last 25 samples) to capture fast-onset gusts
        recent_samples = samples[-25:] if len(samples) >= 25 else samples
        inst_sigma_f = np.mean(np.array([s["sigma_f_hat"] for s in recent_samples]), axis=0)

        return TelemetrySummary(
            timestamp=float(latest_time),
            window_duration_s=float(duration),
            samples_count=n,
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            gyro_rms=[float(x) for x in gyro_rms],
            oscillation_detected=oscillation_detected,
            oscillation_note=oscillation_note,
            pos_error=[float(x) for x in mean_pos_err],
            velocity=[float(x) for x in mean_vel],
            motor_commands_mean=[float(x) for x in ctrl_mean],
            motor_saturation_pct=[float(x) for x in sat_pct],
            unmodeled_force_disturbance=[float(x) for x in sigma_f],
            unmodeled_moment_disturbance=[float(x) for x in sigma_M],
            instantaneous_force_disturbance=[float(x) for x in inst_sigma_f],
        )

    def apply_system2_patch(self, patch: System2Patch) -> None:
        """Applies a new patch from System 2 to be executed by System 1.

        Args:
            patch: System2Patch instance containing new feedforward, gains, trim, or envelope.
        """
        with self._lock:
            self._active_patch = patch
            self._patch_applied_time = time.time()

    def clear_active_patch(self) -> None:
        """Removes the active patch, resetting System 1 to baseline."""
        with self._lock:
            self._active_patch = None
            self._patch_applied_time = None

    def get_patch_applied_time(self) -> Optional[float]:
        """Returns unix timestamp when the currently active patch was applied, or None."""
        with self._lock:
            if self._active_patch is None:
                return None
            return self._patch_applied_time

    def get_patch_age_s(self) -> Optional[float]:
        """Returns elapsed seconds since the active patch was applied, or None if unpatched."""
        with self._lock:
            if self._active_patch is None or self._patch_applied_time is None:
                return None
            return max(0.0, time.time() - self._patch_applied_time)
