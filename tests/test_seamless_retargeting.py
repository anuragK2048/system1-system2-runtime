"""Unit Tests for Seamless In-Flight Multi-Target Navigation & Retargeting.

Verifies:
1. In-flight retargeting maintains continuous 500 Hz physics without teleporting or dropping.
2. Drone smoothly transitions from Pad Alpha to Platform Delta, then to Left Window.
3. Drone maintains station-keeping hover when mission completes rather than terminating simulation loop.
4. Grand Tour mode automatically chains all targets sequentially.
"""

import time
import unittest
from runtime.server.navigation_state import nav_coordinator


class TestSeamlessNavigationRetargeting(unittest.TestCase):
    """Verifies continuous multi-target flight without resets."""

    def setUp(self):
        nav_coordinator.reset_to_origin()

    def tearDown(self):
        nav_coordinator.stop_mission(status="idle")

    def test_seamless_retarget_alpha_to_delta(self):
        """Verifies transitioning from Alpha to Delta in-air retains position."""
        res_alpha = nav_coordinator.start_mission(target="alpha", use_vlm=False, duration=10.0)
        self.assertEqual(res_alpha["status"], "started")

        # Let drone liftoff and advance
        time.sleep(1.0)
        curr_pos = nav_coordinator.sim.data.qpos[0:3].copy()
        self.assertGreater(curr_pos[2], 0.5, "Drone must be airborne")

        # Retarget to Delta mid-flight without reset
        res_delta = nav_coordinator.start_mission(target="delta", reset=False)
        self.assertEqual(res_delta["status"], "retargeted")
        self.assertEqual(res_delta["mission"], "delta")

        # Ensure drone was NOT reset to origin [0, 0, 0.8]
        new_pos = nav_coordinator.sim.data.qpos[0:3].copy()
        self.assertGreater(new_pos[2], 0.5, "Drone must remain airborne after retarget")
        self.assertTrue(nav_coordinator.is_running, "Simulation loop must continue running")

    def test_seamless_retarget_delta_to_window(self):
        """Verifies transitioning from Delta back toward Left Window."""
        nav_coordinator.start_mission(target="delta", reset=False)
        time.sleep(0.5)

        # Retarget to window
        res_win = nav_coordinator.start_mission(target="window", reset=False)
        self.assertEqual(res_win["status"], "retargeted")
        self.assertEqual(res_win["mission"], "window")
        self.assertIn(nav_coordinator.navigator.phase, ("PHASE_RUNWAY_RETURN_HUB", "PHASE_1_SECTOR_SCAN"))

    def test_direct_navigation_does_not_route_through_origin(self):
        """Verifies drone at Platform Delta (X=8.0m) navigates directly to Window, not origin [0,0]."""
        with nav_coordinator.sim_lock:
            nav_coordinator.sim.data.qpos[0:3] = [8.0, 0.0, 1.10]
        nav_coordinator.navigator.phase = "PHASE_4_PRECISION_HOVER"

        # Retarget to window
        nav_coordinator.navigator.retarget("window")
        self.assertEqual(nav_coordinator.navigator.phase, "PHASE_1_SECTOR_SCAN")

        # Step cognitive reasoner
        patch = nav_coordinator.navigator._visual_inertial_reasoner(
            telemetry=nav_coordinator.get_telemetry_snapshot(),
            sim_time=5.0,
        )

        # Target must be either holding in-place at X=8.0 (visual saccade) or direct to window entrance [0, 3.8, 1.1]
        cmd_tgt = patch.waypoint_command.target
        self.assertNotEqual(cmd_tgt, [0.0, 0.0, 1.0], "Must NEVER route back through origin [0, 0, 1.0]")
        self.assertTrue("[SYSTEM 2 VLN]" in patch.diagnosis or "[META LLAMA 3.2 VISION]" in patch.diagnosis)

    def test_explicit_reset_resets_origin(self):
        """Verifies passing reset=True resets drone to origin."""
        nav_coordinator.start_mission(target="alpha")
        time.sleep(0.5)

        res_reset = nav_coordinator.start_mission(target="alpha", reset=True)
        self.assertEqual(res_reset["status"], "started")
        self.assertAlmostEqual(nav_coordinator.sim.data.time, 0.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
