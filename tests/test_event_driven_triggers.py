"""Tests for Event-Driven System 2 Triggers & Synchronized Guidance Metrics.

Verifies:
1. No spurious heartbeat VLM invocations or patches when drone is stationary or holding station.
2. VLM triggers fire strictly on genuine events (disturbances, phase changes, motor failure).
3. Telemetry snapshot synchronizes live System 1 inputs with System 2 guidance.
"""

import unittest
import numpy as np

from runtime.sim.simulator import DroneSimulator, DroneConfig
from runtime.sim.autonomous_target_flight import AutonomousTargetNavigator, SCENE_PATH
from runtime.server.navigation_state import NavigationSimulationCoordinator


class TestEventDrivenTriggersAndGuidance(unittest.TestCase):
    def setUp(self):
        cfg = DroneConfig(spawn_pos=[0.0, 0.0, 0.8])
        self.sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=cfg)
        self.navigator = AutonomousTargetNavigator(
            sim=self.sim,
            target_mission="alpha",
            use_vlm=True,
            vlm_model="meta-llama/Llama-3.2-11B-Vision-Instruct (Modal A100 GPU)",
        )

    def tearDown(self):
        self.navigator.close()

    def test_no_heartbeat_spam_during_calm_station_keeping(self):
        """Verifies no heartbeat triggers fire when drone is calmly holding station even after 30s."""
        telemetry = self.navigator.get_onboard_telemetry()
        self.navigator.last_vlm_call_time = 0.0
        self.navigator.phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._prev_phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._last_vlm_force_norm = 0.0
        self.navigator._last_vlm_pos_err = 0.02
        self.navigator.mission_complete = True

        # Test at t=5s, t=10s, t=30s - none should trigger
        for t_eval in [5.0, 10.0, 20.0, 30.0]:
            needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=t_eval)
            self.assertFalse(
                needed,
                f"Heartbeat unexpectedly triggered at t={t_eval}s: {reason}"
            )
            self.assertIn("station-keeping", reason.lower())

    def test_disturbance_triggers_system2_even_while_holding_station(self):
        """Verifies an active disturbance (wind gust) immediately triggers System 2 while hovering."""
        telemetry = self.navigator.get_onboard_telemetry()
        self.navigator.last_vlm_call_time = 0.0
        self.navigator.phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._prev_phase = "PHASE_4_PRECISION_HOVER"
        self.navigator._last_vlm_force_norm = 0.0
        self.navigator.mission_complete = True

        # Inject sudden crosswind (+4.8N)
        telemetry["unmodeled_force_disturbance_N"] = [4.80, 0.00, 0.00]

        needed, reason = self.navigator.should_query_vlm(telemetry, sim_time=5.0)
        self.assertTrue(needed)
        self.assertIn("L1 force disturbance spike", reason)
        self.assertIn("4.8N", reason)

    def test_telemetry_snapshot_contains_synchronized_system1_inputs(self):
        """Verifies telemetry snapshot provides real-time System 1 inputs matching System 2 patch."""
        coord = NavigationSimulationCoordinator()
        snap = coord.get_telemetry_snapshot()
        self.assertIn("cmd_target_pos", snap)
        self.assertIn("cmd_yaw_deg", snap)
        self.assertIn("cmd_max_vel", snap)
        self.assertIn("feedforward_force", snap)
        self.assertIn("motor_trim", snap)
        self.assertIn("max_tilt_clamp", snap)
        self.assertIn("phase", snap)

        # Confirm Meta Llama 3.2 Vision branding
        self.assertIn("Llama-3.2", snap["vlm_model"])
        self.assertIn("Modal", snap["vlm_model"])

    def test_dashboard_html_contains_no_removed_scorecard_or_world_pos(self):
        """Verifies frontend HTML completely removed the Scorecard and 3D World Pos card."""
        from pathlib import Path
        html_path = Path(__file__).resolve().parents[1] / "frontend" / "navigation_dashboard.html"
        content = html_path.read_text(encoding="utf-8")

        self.assertNotIn("Automated Navigation Test Suite Scorecard", content)
        self.assertNotIn("btn-evals-suite", content)
        self.assertNotIn("evals-scorecard-body", content)
        self.assertNotIn("3D Position (World)", content)
        self.assertNotIn("val-pos-xyz", content)
        self.assertIn("META LLAMA 3.2 VISION ON MODAL GPU", content)
        self.assertIn("grid-template-columns: repeat(3, 1fr)", content)


if __name__ == "__main__":
    unittest.main()
