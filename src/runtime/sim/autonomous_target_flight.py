"""Autonomous Target Flight Demonstration.

Demonstrates the Skydio X2 quadrotor autonomously navigating along the flight
corridor in single-drone mode, tracking target pads, passing through the pylon
gate, and hovering directly over Target Pad Alpha (x=5.5m) and Elevated Station
Delta (x=8.0m) using the System-1 (500 Hz reflex) and System-2 (VLM waypoint)
architecture.

Usage:
    # Run interactive 3D visualizer (macOS requires mjpython):
    uv run mjpython src/runtime/sim/autonomous_target_flight.py

    # Run headless test with metrics output:
    uv run python src/runtime/sim/autonomous_target_flight.py --headless
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import mujoco

from runtime.sim.config import SCENE_PATH, SimConfig, DroneConfig
from runtime.sim.simulator import DroneSimulator
from runtime.schemas.tier2 import System2Patch, WaypointCommand, SafetyEnvelope, GainMultipliers


class AutonomousTargetNavigator:
    """Coordinates autonomous waypoint mission execution across the target pads."""

    def __init__(self, sim: DroneSimulator, headless: bool = False):
        self.sim = sim
        self.headless = headless
        self.waypoints = [
            # 1. Lift off to stable hover at spawn
            {"name": "Spawn Hover (Z=1.0m)", "target": [0.0, 0.0, 1.0], "hold_time": 2.0, "tolerance": 0.15},
            # 2. Traverse corridor pylon gate
            {"name": "Pylon Gate Traverse (X=3.5m)", "target": [3.5, 0.0, 1.0], "hold_time": 2.5, "tolerance": 0.25},
            # 3. Precision hover over Target Pad Alpha
            {"name": "Target Pad Alpha Hover (X=5.5m)", "target": [5.5, 0.0, 1.0], "hold_time": 3.0, "tolerance": 0.15},
            # 4. Elevated Station Delta approach & elevated hover
            {"name": "Elevated Station Delta Hover (X=8.0m, Z=1.1m)", "target": [8.0, 0.0, 1.1], "hold_time": 3.0, "tolerance": 0.15},
        ]
        self.current_wp_idx = 0
        self.wp_reached_time: Optional[float] = None
        self.flight_log: List[dict] = []

    def update_vlm_system2(self, sim_time: float) -> Optional[dict]:
        """Simulates the System 2 VLM cognitive loop executing at 1-2 Hz."""
        if self.current_wp_idx >= len(self.waypoints):
            return None

        wp = self.waypoints[self.current_wp_idx]
        pos = self.sim.data.qpos[0:3].copy()
        target = np.array(wp["target"], dtype=np.float32)
        dist = float(np.linalg.norm(pos - target))

        # Check if current waypoint reached within tolerance
        if dist <= wp["tolerance"]:
            if self.wp_reached_time is None:
                self.wp_reached_time = sim_time
                print(f"🎯 [VLM COGNITION] Reached waypoint [{wp['name']}]! Target Dist: {dist:.3f}m. Holding station...")
            elif sim_time - self.wp_reached_time >= wp["hold_time"]:
                print(f"✅ [VLM COGNITION] Hold complete for [{wp['name']}]. Advancing to next target!")
                self.current_wp_idx += 1
                self.wp_reached_time = None
                if self.current_wp_idx < len(self.waypoints):
                    wp = self.waypoints[self.current_wp_idx]
                    target = np.array(wp["target"], dtype=np.float32)
                else:
                    print("🏆 [MISSION COMPLETE] All target pads successfully visited and hovered over!")
                    return None

        # Build System 2 VLM Patch
        patch = System2Patch(
            diagnosis=f"Autonomous visual servoing toward {wp['name']}",
            waypoint_command=WaypointCommand(
                mode="world",
                target=[float(target[0]), float(target[1]), float(target[2])],
                yaw_deg=0.0,
                max_velocity=1.5,
                corridor_width=0.8,
            ),
            feedforward_force_offset=[0.0, 0.0, 0.0],
            gain_multipliers=GainMultipliers(pos_kp=1.1, pos_kd=1.0, att_kp=1.0, att_kd=1.0),
            safety_envelope=SafetyEnvelope(max_tilt_deg=20.0),
            timestamp=sim_time,
        )

        self.sim.bridge.apply_system2_patch(patch)
        return {"wp": wp["name"], "dist": dist, "target": target.tolist(), "pos": pos.tolist()}

    def run_headless(self, max_duration: float = 16.0) -> bool:
        """Executes simulation headlessly and reports trajectory telemetry."""
        print("\n" + "═" * 78)
        print("🚁 [AUTONOMOUS TARGET FLIGHT DEMO - HEADLESS MODE]")
        print(f"📁 Scene: {SCENE_PATH}")
        print("🎯 Target Flight Plan:")
        for i, wp in enumerate(self.waypoints):
            print(f"   {i+1}. {wp['name']} -> Target: {wp['target']}")
        print("═" * 78 + "\n")

        dt = self.sim.sim_cfg.physics_timestep
        last_s2_time = -1.0
        s2_interval = 0.5  # 2 Hz System-2 cognitive cycle

        while self.sim.data.time < max_duration:
            t = float(self.sim.data.time)

            # System 2 Reasoning update
            if t - last_s2_time >= s2_interval:
                last_s2_time = t
                wp_info = self.update_vlm_system2(t)
                if wp_info:
                    pos = wp_info["pos"]
                    print(f"⏱️ t={t:5.2f}s | Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                          f"Target: {wp_info['wp']} (Dist: {wp_info['dist']:4.2f}m)")

            # Step physics
            self.sim.step_simulation(current_time=time.time())

            if self.current_wp_idx >= len(self.waypoints):
                print(f"\n🎉 Mission completed ahead of schedule at t={t:.2f}s!")
                return True

        success = self.current_wp_idx >= 3  # At least reached Pad Alpha
        print(f"\nSimulation finished at t={self.sim.data.time:.2f}s. Success: {success}")
        return success


def main():
    parser = argparse.ArgumentParser(description="Autonomous Target Flight Demo")
    parser.add_argument("--headless", action="store_true", help="Run without OpenGL visualizer window")
    parser.add_argument("--duration", type=float, default=16.0, help="Max simulation duration in seconds")
    args = parser.parse_args()

    sim = DroneSimulator(scene_path=SCENE_PATH)
    navigator = AutonomousTargetNavigator(sim=sim, headless=args.headless)

    if args.headless:
        success = navigator.run_headless(max_duration=args.duration)
        sys.exit(0 if success else 1)
    else:
        print("Launching interactive visualizer...")
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            last_s2_time = -1.0
            while viewer.is_running():
                t = float(sim.data.time)
                if t - last_s2_time >= 0.5:
                    last_s2_time = t
                    navigator.update_vlm_system2(t)

                sim.step_simulation(current_time=time.time())
                viewer.sync()
                time.sleep(sim.sim_cfg.physics_timestep)


if __name__ == "__main__":
    main()
