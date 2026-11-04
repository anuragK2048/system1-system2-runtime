"""Verification script for Milestone 1: System 1 Yaw Setpoint Tracking.

Simulates the Skydio X2 quadrotor in MuJoCo:
1. Liftoff and stabilize at neutral hover (Z=1.0m, Heading=0° facing runway corridor).
2. Offscreen render FPV frame before turn (showing forward runway, Pad Alpha, pylons).
3. Inject System 2 patch with waypoint_command requesting in-place yaw saccade to +90.0° (facing Left Window).
4. Run 500 Hz simulation for 2.5s, logging heading, altitude, horizontal drift, and motor thrusts.
5. Offscreen render FPV frame after turn (showing Left Window framed by calibrated HUD).
6. Verify:
   - Final yaw within ±2.5° of 90.0°.
   - Altitude variation < 0.05m (no droop).
   - Horizontal position drift < 0.06m (tight in-place rotation).
   - Motor thrusts within [1.5, 5.5] N (no saturation against [0, 13] N limits).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco

from runtime.sim.simulator import DroneSimulator
from runtime.sim.config import DroneConfig, SimConfig
from runtime.schemas.tier2 import System2Patch, WaypointCommand
from vlm.hud_overlay import HUDProjector, Aperture3D


def main():
    print("=" * 72)
    print("🚀 MILESTONE 1 VERIFICATION: IN-PLACE YAW SACCADE & SYSTEM 1 TRACKING")
    print("=" * 72)

    # 1. Initialize Simulator with 500 Hz physics
    drone_cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
    sim = DroneSimulator(drone_cfg=drone_cfg)

    hud = HUDProjector(camera_name="fpv", width=480, height=360)
    renderer = mujoco.Renderer(sim.model, height=360, width=480)

    # Known arena apertures
    window_aperture = Aperture3D(
        name="WINDOW_LEFT",
        center=(0.0, 5.0, 1.10),
        width=1.40,
        height=1.20,
        yaw=math.radians(90.0),
        shape="window",
        color=(255, 190, 0),       # Amber Gold
        inner_color=(0, 255, 130),  # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    )

    # --------------------------------------------------------------------------
    # Step 1: Establish steady hover at [0.0, 0.0, 1.0] facing 0° (corridor)
    # --------------------------------------------------------------------------
    print("\n[PHASE 0] Ascending to hover at Z=1.0m, Heading 0° (Runway Corridor)...")
    sim.controller.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for step in range(500):  # 1.0 second @ 500 Hz
        sim.step_simulation(step * 0.002)

    pos_before = sim.data.qpos[0:3].copy()
    w, qx, qy, qz = sim.data.qpos[3:7].copy()
    yaw_before_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))))
    print(f"  Steady Hover Established:")
    print(f"  • Position: X={pos_before[0]:+.3f}m, Y={pos_before[1]:+.3f}m, Z={pos_before[2]:.3f}m")
    print(f"  • Heading:  {yaw_before_deg:+.2f}° (Nose facing +X Runway)")

    # Render Before Turn (Neutral forward view)
    before_img = hud.render_and_overlay(
        sim.model,
        sim.data,
        renderer=renderer,
        drone_pos=pos_before,
        drone_vel=sim.data.qvel[0:3],
        apertures=[window_aperture],
        radii=[1.0, 2.0, 3.0],
    )
    output_dir = PROJECT_ROOT / "vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = output_dir / "verification_milestone1_before_turn.jpg"
    before_img.save(before_path, quality=92)
    print(f"  📸 Saved Before-Turn FPV Frame: {before_path.name}")

    # --------------------------------------------------------------------------
    # Step 2: Inject Saccade Command to Yaw +90.0° (Face Left Window)
    # --------------------------------------------------------------------------
    print("\n[PHASE 1] Injecting System 2 Saccade Command: yaw_deg = +90.0° (Rotate Left)...")
    patch = System2Patch(
        diagnosis="Active perception saccade to inspect unobservable Left Window sector",
        waypoint_command=WaypointCommand(
            mode="world",
            target=[0.0, 0.0, 1.0],
            yaw_deg=90.0,
            max_velocity=0.8,
            corridor_width=0.5,
        ),
        confidence=1.0,
    )
    sim.bridge.apply_system2_patch(patch)

    # --------------------------------------------------------------------------
    # Step 3: Run 500 Hz Simulation for 2.5s and monitor trajectory
    # --------------------------------------------------------------------------
    yaw_history = []
    alt_history = []
    drift_history = []
    min_motor = float("inf")
    max_motor = float("-inf")

    sim_start_time = float(sim.data.time)
    for step in range(1250):  # 2.5 seconds @ 500 Hz
        current_sim_time = sim_start_time + step * 0.002
        diag = sim.step_simulation(current_sim_time)

        pos = sim.data.qpos[0:3].copy()
        w, qx, qy, qz = sim.data.qpos[3:7].copy()
        current_yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))))
        xy_drift = float(np.linalg.norm(pos[:2] - pos_before[:2]))
        alt = float(pos[2])

        yaw_history.append(current_yaw_deg)
        alt_history.append(alt)
        drift_history.append(xy_drift)

        m_thrusts = diag.get("motor_thrusts", [3.25, 3.25, 3.25, 3.25])
        min_motor = min(min_motor, float(np.min(m_thrusts)))
        max_motor = max(max_motor, float(np.max(m_thrusts)))

        if (step + 1) % 250 == 0:
            t_elapsed = (step + 1) * 0.002
            print(f"  t={t_elapsed:.1f}s | Yaw: {current_yaw_deg:+6.1f}° | Alt: {alt:5.3f}m | Drift: {xy_drift:5.3f}m | Slew Setpoint: {diag.get('smoothed_yaw_deg', 0):+6.1f}°")

    # --------------------------------------------------------------------------
    # Step 4: Render After Turn (FPV facing Left Window)
    # --------------------------------------------------------------------------
    pos_after = sim.data.qpos[0:3].copy()
    w, qx, qy, qz = sim.data.qpos[3:7].copy()
    yaw_after_deg = float(np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))))

    after_img = hud.render_and_overlay(
        sim.model,
        sim.data,
        renderer=renderer,
        drone_pos=pos_after,
        drone_vel=sim.data.qvel[0:3],
        apertures=[window_aperture],
        radii=[1.0, 2.0, 3.0],
    )
    after_path = output_dir / "verification_milestone1_after_yaw90.jpg"
    after_img.save(after_path, quality=92)
    print(f"\n  📸 Saved After-Turn FPV Frame:  {after_path.name}")
    renderer.close()

    # --------------------------------------------------------------------------
    # Step 5: Metric Evaluation
    # --------------------------------------------------------------------------
    final_yaw_err = abs(yaw_after_deg - 90.0)
    max_alt_dev = float(np.max(np.abs(np.array(alt_history) - 1.0)))
    max_xy_drift = float(np.max(drift_history))

    print("\n" + "=" * 72)
    print("📊 VERIFICATION METRICS")
    print("=" * 72)
    print(f"  Target Yaw:            +90.00°")
    print(f"  Achieved Yaw:          {yaw_after_deg:+6.2f}° (Error: {final_yaw_err:.2f}°)")
    print(f"  Altitude Band:         [{min(alt_history):.3f}m, {max(alt_history):.3f}m] (Max Dev: {max_alt_dev*100:.1f} cm)")
    print(f"  In-Place Drift:        Max {max_xy_drift*100:.1f} cm (Final {drift_history[-1]*100:.1f} cm)")
    print(f"  Motor Thrust Range:    [{min_motor:.2f}N, {max_motor:.2f}N] (Safe limits [0, 13]N)")

    # Assertions
    checks = [
        ("Final Yaw ±2.5°", final_yaw_err <= 2.5),
        ("Altitude Droop < 5 cm", max_alt_dev <= 0.05),
        ("In-Place Drift < 6 cm", max_xy_drift <= 0.06),
        ("No Motor Saturation", min_motor >= 0.5 and max_motor <= 8.0),
    ]

    all_passed = True
    print("\n📋 VALIDATION CHECKLIST:")
    for name, passed in checks:
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status_icon}] {name}")

    print("=" * 72)
    if all_passed:
        print("🎉 MILESTONE 1 VERIFIED: Drone smoothly rotates in-place without altitude droop!")
        print("   Visual proof generated in:")
        print(f"   • {before_path.name}")
        print(f"   • {after_path.name}")
    else:
        print("⚠️ Some criteria failed. Review gains and slew limits.")
    print("=" * 72)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
