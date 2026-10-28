"""Pure physics equations and perturbations for the Chaos Engine.

Executes physical perturbations directly against MuJoCo data structures:
1. Multi-rotor actuator degradation (thrust scaling, chipped propeller harmonics, ESC cutouts).
2. 3D aerodynamic wind vectors, gust pulses, and Dryden-style turbulence.
3. Tethered payload coupling with wind aerodynamic drag (wind directly swings the cargo).
4. IMU sensor corruption (gyro chatter, accelerometer noise, bias drift, vibration resonance).
"""

from typing import Optional, Set, Tuple
import numpy as np
import mujoco

from runtime.chaos.models import (
    MotorChaosConfig,
    WindChaosConfig,
    PayloadChaosConfig,
    SensorChaosConfig,
)


# ==============================================================================
# 1. ACTUATOR & MOTOR DEGRADATION
# ==============================================================================
def apply_actuator_degradation(
    commanded_thrust: np.ndarray,
    cfg: MotorChaosConfig,
    sim_time: float,
) -> np.ndarray:
    """Computes degraded actual motor thrusts based on failure configuration.

    Args:
        commanded_thrust: 4-element array of desired motor thrusts (Newtons).
        cfg: MotorChaosConfig specifying multipliers and failure modes.
        sim_time: Current simulation timestamp in seconds.

    Returns:
        Actual thrust produced by the 4 physical rotors.
    """
    actual_thrust = np.array(commanded_thrust, dtype=np.float64, copy=True)

    # 1. Direct thrust multipliers (e.g., M1 at 40% power)
    for i in range(min(4, len(cfg.multipliers))):
        actual_thrust[i] *= cfg.multipliers[i]

    # 2. Chipped propeller blade: creates RPM-dependent aerodynamic flutter & thrust jitter
    if cfg.chipped_motor_idx is not None and 0 <= cfg.chipped_motor_idx < 4:
        idx = cfg.chipped_motor_idx
        # High-frequency rotor vibration ripple (~85 Hz fundamental prop passage)
        ripple = 0.12 * np.sin(2.0 * np.pi * 85.0 * sim_time) + 0.06 * np.cos(2.0 * np.pi * 170.0 * sim_time)
        actual_thrust[idx] = max(0.0, actual_thrust[idx] * (1.0 + ripple))

    # 3. Intermittent ESC / wiring cutout dropout
    if cfg.dropout_prob > 0.0:
        for i in range(4):
            if np.random.random() < cfg.dropout_prob:
                actual_thrust[i] = 0.0

    return actual_thrust


# ==============================================================================
# 2. 3D WIND & ATMOSPHERIC TURBULENCE
# ==============================================================================
def compute_wind_force(
    cfg: WindChaosConfig,
    sim_time: float,
) -> np.ndarray:
    """Computes the total 3D wind disturbance force vector [Fx, Fy, Fz] in Newtons.

    Combines:
    - Constant steady crosswind / downdraft
    - Dynamic gust bursts (half-sine pulse)
    - Stochastic Dryden-style turbulence buffeting
    """
    total_force = np.zeros(3, dtype=np.float64)

    # 1. Steady wind force
    if cfg.enabled:
        total_force += np.array(cfg.steady_force, dtype=np.float64)

    # 2. Dynamic gust burst (half-sine profile)
    if cfg.gust_active:
        dt = sim_time - cfg.gust_start_time
        if 0.0 <= dt <= cfg.gust_duration_s:
            # Half-sine pulse peaking at gust_force
            pulse = np.sin(np.pi * (dt / cfg.gust_duration_s))
            direction = np.array(cfg.gust_direction, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            total_force += direction * (cfg.gust_force * pulse)

    # 3. Atmospheric turbulence (stochastic Dryden buffeting)
    if cfg.turbulence_std > 0.0:
        turbulence = np.random.normal(0.0, cfg.turbulence_std, size=3)
        total_force += turbulence

    return total_force


def apply_wind_to_body(
    data: mujoco.MjData,
    body_id: int,
    wind_force: np.ndarray,
) -> None:
    """Applies the computed 3D wind force vector to a MuJoCo body's center of mass."""
    data.xfrc_applied[body_id, 0:3] += wind_force[0:3]


# ==============================================================================
# 3. PAYLOAD COUPLING & WIND-INDUCED SWING
# ==============================================================================
def apply_payload_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_body_id: int,
    tether_geom_id: int,
    bob_geom_id: int,
    cfg: PayloadChaosConfig,
    wind_force: np.ndarray,
    kicked_set: Set[int],
    qpos_adr: int,
    qvel_adr: int,
) -> None:
    """Simulates physical coupling or unhooking of the swinging cargo.

    When payload is attached:
      - Makes tether (gold) and bob (red) visible.
      - Clears counter-force so full gravity (m*g) pulls down on the drone.
      - Gives an initial pendulum kick on first attachment.
      - APPLIES AERODYNAMIC WIND DRAG TO THE PAYLOAD BOB:
        When wind blows, it pushes the hanging bob laterally, inducing realistic
        chaotic pendulum swings that yank on the drone airframe!

    When payload is detached:
      - Makes tether and bob invisible (alpha = 0.0).
      - Applies an exact upward counter-force (m*g) to neutralize mass.
      - Clamps swing angles and velocities to zero.
    """
    if cfg.attached:
        # 1. Newly attached: impart an initial swing angle and angular velocity
        if payload_body_id not in kicked_set:
            data.qpos[qpos_adr] = cfg.kick_angle_rad
            data.qvel[qvel_adr] = cfg.kick_velocity
            kicked_set.add(payload_body_id)

        # 2. Make tether (gold) and bob (red) visible only when visibility changes
        if model.geom_rgba[tether_geom_id, 3] < 0.5:
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 1.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 1.0]

        # 3. Clear vertical counter-force so gravity pulls naturally
        data.xfrc_applied[payload_body_id, :] = 0.0

        # 4. Physical wind aerodynamic drag on the payload bob!
        # When wind blows, it directly pushes the hanging bob laterally
        if cfg.aerodynamic_drag_enabled and np.any(np.abs(wind_force) > 0.01):
            # Scale wind force to spherical bob cross-section (~18% of drone frontal area)
            bob_wind_force = wind_force * 0.18
            data.xfrc_applied[payload_body_id, 0:3] = bob_wind_force
    else:
        kicked_set.discard(payload_body_id)

        # Make tether and bob invisible only when visibility changes
        if model.geom_rgba[tether_geom_id, 3] > 0.5:
            model.geom_rgba[tether_geom_id] = [1.0, 0.85, 0.10, 0.0]
            model.geom_rgba[bob_geom_id] = [0.95, 0.15, 0.15, 0.0]

        # Neutralize payload weight so the drone feels exactly 0.00 kg
        data.xfrc_applied[payload_body_id, :] = 0.0
        data.xfrc_applied[payload_body_id, 2] = cfg.mass * 9.81
        data.qpos[qpos_adr : qpos_adr + 2] = 0.0
        data.qvel[qvel_adr : qvel_adr + 2] = 0.0


# ==============================================================================
# 4. SENSOR NOISE & IMU VIBRATION
# ==============================================================================
def corrupt_imu_sensors(
    raw_omega: np.ndarray,
    cfg: SensorChaosConfig,
    sim_time: float,
    raw_accel: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Injects high-frequency Gaussian vibration, bias drift, and harmonics into IMU.

    Args:
        raw_omega: True 3D angular velocity from MuJoCo gyro (rad/s).
        cfg: SensorChaosConfig.
        sim_time: Simulation timestamp (s).
        raw_accel: Optional true 3D specific force from accelerometer (m/s^2).

    Returns:
        Tuple of (corrupted_omega, corrupted_accel).
    """
    if not cfg.enabled:
        return raw_omega, raw_accel

    corrupted_omega = raw_omega.copy()

    # 1. Gaussian gyro vibration noise
    if cfg.gyro_noise_std > 0.0:
        noise = np.random.normal(0.0, cfg.gyro_noise_std, size=3)
        corrupted_omega += noise

    # 2. Constant / temperature bias drift
    if any(abs(b) > 1e-6 for b in cfg.gyro_bias):
        corrupted_omega += np.array(cfg.gyro_bias, dtype=np.float64)

    # 3. Mechanical harmonic vibration chatter (motor imbalance resonance)
    if cfg.harmonic_amplitude > 0.0 and cfg.harmonic_freq_hz > 0.0:
        phase = 2.0 * np.pi * cfg.harmonic_freq_hz * sim_time
        harmonic = cfg.harmonic_amplitude * np.array([
            np.sin(phase),
            np.cos(phase),
            np.sin(phase + 0.5),
        ])
        corrupted_omega += harmonic

    corrupted_accel = None
    if raw_accel is not None:
        corrupted_accel = raw_accel.copy()
        if cfg.accel_noise_std > 0.0:
            corrupted_accel += np.random.normal(0.0, cfg.accel_noise_std, size=3)

    return corrupted_omega, corrupted_accel
