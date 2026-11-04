"""Automated unit and integration tests for chaos injection and flight path retention.

Verifies:
1. NavigationSimulationCoordinator chaos preset application, toggling, gust bursts, and clearing.
2. Trajectory retention and altitude preservation under all 5 major chaos presets:
   - Preset 1: Motor 1 Cut (40% thrust + rotor flutter)
   - Preset 2: 6.5N Crosswind Microburst + Dryden turbulence
   - Preset 3: Tethered Cargo Pendulum (0.35kg mass + aerodynamic drag)
   - Preset 4: IMU Gyroscope Sensor Chatter (0.85 rad/s noise)
   - Preset 5: Triple Compound Stress (Motor 1 cut + Wind + Cargo)
3. In-flight seamless retargeting while subjected to active physical disturbances.
4. FastAPI REST endpoints for chaos presets, gusts, and state reporting.
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from runtime.sim.config import DroneConfig
from runtime.sim.simulator import DroneSimulator
from runtime.sim.autonomous_target_flight import AutonomousTargetNavigator, SCENE_PATH
from runtime.server.navigation_state import NavigationSimulationCoordinator
from runtime.server.navigation_server import app


class TestChaosCoordinatorMethods(unittest.TestCase):
    """Verifies NavigationSimulationCoordinator methods for chaos control."""

    def setUp(self):
        self.coord = NavigationSimulationCoordinator()
        self.coord.clear_all_chaos()

    def tearDown(self):
        self.coord.clear_all_chaos()

    def test_preset_application_and_clearing(self):
        """Verifies applying presets updates active presets and summary."""
        res = self.coord.apply_chaos_preset("1")
        self.assertEqual(res["status"], "ok")
        self.assertIn("1", res["active_presets"])
        self.assertTrue(self.coord.sim.chaos.drone1.motor.is_active())

        # Apply Preset 2
        res2 = self.coord.apply_chaos_preset("2", additive=True)
        self.assertEqual(res2["status"], "ok")
        self.assertTrue(self.coord.sim.chaos.drone1.wind.is_active())

        # Clear All
        clear_res = self.coord.clear_all_chaos()
        self.assertEqual(clear_res["status"], "ok")
        self.assertEqual(len(clear_res["active_presets"]), 0)
        self.assertFalse(self.coord.sim.chaos.drone1.motor.is_active())
        self.assertFalse(self.coord.sim.chaos.drone1.wind.is_active())

    def test_trigger_wind_gust(self):
        """Verifies trigger_wind_gust activates transient burst."""
        res = self.coord.trigger_wind_gust(force=7.5, duration_s=1.0)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(self.coord.sim.chaos.drone1.wind.gust_active)
        self.assertAlmostEqual(self.coord.sim.chaos.drone1.wind.gust_force, 7.5)

    def test_backward_compatible_set_chaos(self):
        """Verifies set_chaos maps legacy strings to presets."""
        res_wind = self.coord.set_chaos("wind", active=True)
        self.assertIn("2", res_wind["active_presets"])

        res_motor = self.coord.set_chaos("motor", active=True)
        self.assertIn("1", res_motor["active_presets"])

        res_clear = self.coord.set_chaos("none", active=False)
        self.assertEqual(len(res_clear["active_presets"]), 0)

    def test_telemetry_snapshot_includes_chaos(self):
        """Verifies telemetry snapshot includes chaos summary and active presets."""
        self.coord.apply_chaos_preset("2")
        snap = self.coord.get_telemetry_snapshot()
        self.assertIn("chaos", snap)
        self.assertIn("active_presets", snap)
        self.assertIn("2", snap["active_presets"])


class TestChaosFlightPathRetention(unittest.TestCase):
    """Verifies that the quadrotor retains its navigation flight path under all major chaos presets."""

    def _run_flight(self, preset_id: str, target: str = "alpha", sim_seconds: float = 3.0):
        sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=DroneConfig(spawn_pos=[0.0, 0.0, 0.8]))
        nav = AutonomousTargetNavigator(sim=sim, target_mission=target)

        # 1. Nominal liftoff for 1.0 second (500 steps)
        for i in range(500):
            t = float(sim.data.time)
            if i % 250 == 0:
                nav.step_cognitive_cycle(t)
            sim.step_simulation(t)

        start_pos = sim.data.qpos[0:3].copy()
        self.assertGreater(start_pos[2], 0.70, "Drone failed initial liftoff")

        # 2. Inject specified chaos preset
        if preset_id != "0":
            sim.chaos.apply_preset(preset_id)

        # 3. Fly for remaining duration
        steps = int(sim_seconds * 500)
        for i in range(steps):
            t = float(sim.data.time)
            if i % 250 == 0:
                nav.step_cognitive_cycle(t)
            sim.step_simulation(t)

        final_pos = sim.data.qpos[0:3].copy()
        quat = sim.data.qpos[3:7].copy()
        w, qx, qy, qz = quat
        roll_deg = float(np.degrees(np.arctan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2))))
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2.0 * (w * qy - qz * qx), -1.0, 1.0))))
        tilt_deg = float(np.hypot(roll_deg, pitch_deg))

        return start_pos, final_pos, tilt_deg, nav.phase

    def test_flight_path_retention_preset1_motor_cut(self):
        """Preset 1: Motor 1 Cut (40% thrust) maintains altitude and advances along corridor."""
        start_pos, final_pos, tilt_deg, phase = self._run_flight("1", target="alpha", sim_seconds=2.5)
        self.assertGreater(final_pos[2], 0.75, "Drone lost altitude under Motor 1 degradation")
        self.assertLess(tilt_deg, 40.0, "Attitude tilt exceeded safe recovery envelope")
        self.assertGreater(final_pos[0], start_pos[0], "Drone failed to advance forward along runway")
        self.assertEqual(phase, "PHASE_2_CORRIDOR_APPROACH")

    def test_flight_path_retention_preset2_crosswind(self):
        """Preset 2: 6.5N Crosswind advances toward Pad Alpha without drifting out of arena."""
        start_pos, final_pos, tilt_deg, phase = self._run_flight("2", target="alpha", sim_seconds=2.5)
        self.assertGreater(final_pos[2], 0.80, "Drone lost altitude under crosswind")
        self.assertLess(tilt_deg, 45.0, "Attitude tilt exceeded recovery bounds")
        self.assertGreater(final_pos[0], 0.80, "Drone failed to penetrate 6.5N crosswind toward target")
        self.assertLess(abs(final_pos[1]), 1.80, "Drone blew off corridor bounds")
        self.assertEqual(phase, "PHASE_2_CORRIDOR_APPROACH")

    def test_flight_path_retention_preset3_tethered_cargo(self):
        """Preset 3: Tethered Cargo Pendulum (0.35kg) advances along corridor with altitude hold."""
        start_pos, final_pos, tilt_deg, phase = self._run_flight("3", target="alpha", sim_seconds=2.5)
        self.assertGreater(final_pos[2], 0.80, "Cargo pulled drone down")
        self.assertLess(tilt_deg, 40.0, "Pendulum excitation tumbled drone")
        self.assertGreater(final_pos[0], 0.60, "Drone failed to advance forward with cargo")
        self.assertEqual(phase, "PHASE_2_CORRIDOR_APPROACH")

    def test_flight_path_retention_preset4_sensor_chatter(self):
        """Preset 4: IMU Sensor Noise (0.85 rad/s) maintains stable hover and advances."""
        start_pos, final_pos, tilt_deg, phase = self._run_flight("4", target="alpha", sim_seconds=2.5)
        self.assertGreater(final_pos[2], 0.75, "Sensor chatter caused ground collision")
        self.assertLess(tilt_deg, 35.0, "Sensor noise caused attitude divergence")
        self.assertGreater(final_pos[0], start_pos[0], "Drone failed to progress")

    def test_flight_path_retention_preset5_triple_stress(self):
        """Preset 5: Compound failure (M1 cut + 5N wind + Cargo) survives and retains corridor path."""
        start_pos, final_pos, tilt_deg, phase = self._run_flight("5", target="alpha", sim_seconds=3.0)
        self.assertGreater(final_pos[2], 0.70, "Drone crashed under compound stress")
        self.assertLess(tilt_deg, 45.0, "Compound failure caused irrecoverable tumble")
        self.assertGreater(final_pos[0], 0.60, "Drone failed to advance toward Alpha under compound failure")

    def test_in_flight_retargeting_under_crosswind(self):
        """Verifies mid-flight retargeting to Platform Delta and Left Window under 6.5N wind."""
        sim = DroneSimulator(scene_path=SCENE_PATH, drone_cfg=DroneConfig(spawn_pos=[0.0, 0.0, 0.8]))
        nav = AutonomousTargetNavigator(sim=sim, target_mission="alpha")

        # Nominal liftoff
        for i in range(500):
            t = float(sim.data.time)
            if i % 250 == 0: nav.step_cognitive_cycle(t)
            sim.step_simulation(t)

        # Inject 6.5N wind
        sim.chaos.apply_preset("2")

        # Retarget to Platform Delta
        nav.retarget("delta")
        for i in range(500):
            t = float(sim.data.time)
            if i % 250 == 0: nav.step_cognitive_cycle(t)
            sim.step_simulation(t)

        pos_delta = sim.data.qpos[0:3].copy()
        self.assertGreater(pos_delta[0], 0.20)
        self.assertGreater(pos_delta[2], 0.80)

        # Retarget to Left Window
        nav.retarget("window")
        for i in range(500):
            t = float(sim.data.time)
            if i % 250 == 0: nav.step_cognitive_cycle(t)
            sim.step_simulation(t)

        pos_window = sim.data.qpos[0:3].copy()
        self.assertGreater(pos_window[2], 0.80, "Drone crashed during window turn under wind")


class TestChaosRESTEndpoints(unittest.TestCase):
    """Verifies FastAPI endpoints for chaos control."""

    def setUp(self):
        self.client = TestClient(app)
        self.client.post("/api/chaos/clear")

    def tearDown(self):
        self.client.post("/api/chaos/clear")

    def test_chaos_preset_endpoints(self):
        # Apply Preset 1
        r = self.client.post("/api/navigation/chaos/preset", json={"preset_id": "1", "toggle": False})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("1", data["active_presets"])

        # Check state endpoint
        state_r = self.client.get("/api/navigation/chaos/state")
        self.assertEqual(state_r.status_code, 200)
        self.assertIn("1", state_r.json()["active_presets"])

        # Toggle Preset 2
        toggle_r = self.client.post("/api/chaos/preset/toggle", json={"preset_id": "2"})
        self.assertEqual(toggle_r.status_code, 200)

        # Clear chaos
        clear_r = self.client.post("/api/chaos/clear")
        self.assertEqual(clear_r.status_code, 200)
        self.assertEqual(len(clear_r.json()["active_presets"]), 0)

    def test_chaos_gust_endpoint(self):
        r = self.client.post("/api/chaos/gust", json={"force": 8.0, "duration_s": 0.5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_chaos_presets_catalog(self):
        r = self.client.get("/api/chaos/presets")
        self.assertEqual(r.status_code, 200)
        catalog = r.json()
        self.assertGreaterEqual(len(catalog), 5)
        keys = [p["key"] for p in catalog]
        self.assertIn("1", keys)
        self.assertIn("2", keys)
        self.assertIn("5", keys)


if __name__ == "__main__":
    unittest.main()
