"""Integration tests for FastAPI REST Endpoints & WebSocket Telemetry Stream.

Tests:
1. Root endpoint: Metadata and sitemap.
2. Status endpoint: Simulator health, controller modes, timestep.
3. Chaos endpoints: Motor, Wind, Gust, Payload, Sensor, Presets, Reset.
4. Telemetry endpoints: Latest frame, history ring buffer, bridge metrics.
5. WebSocket endpoint: Real-time broadcast and bi-directional control messages.
"""

import unittest
from fastapi.testclient import TestClient

from runtime.server.app import app
from runtime.server.state import sim_context


class TestServerAPI(unittest.TestCase):
    """Test suite for runtime.server REST APIs and WebSocket stream."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure a simulator is attached
        sim_context.ensure_simulator(drone1_mode="pid", drone2_mode="l1_adaptive")

    def setUp(self):
        sim_context.chaos.clear_all()

    def test_root_endpoint(self):
        """Verify root endpoint serves dashboard HTML and /api returns discovery JSON."""
        res_html = self.client.get("/")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("text/html", res_html.headers.get("content-type", ""))

        res_api = self.client.get("/api")
        self.assertEqual(res_api.status_code, 200)
        data = res_api.json()
        self.assertIn("endpoints", data)
        self.assertEqual(data["endpoints"]["telemetry_ws"], "/ws/telemetry")

    def test_status_endpoint(self):
        """Verify system status reports active simulator."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["has_simulator"])
        self.assertIn("drone1_mode", data)
        self.assertIn("drone2_mode", data)

    def test_chaos_state_and_presets(self):
        """Verify reading chaos state and listing presets."""
        res = self.client.get("/api/chaos/state")
        self.assertEqual(res.status_code, 200)
        state = res.json()
        self.assertIn("drone1", state)
        self.assertIn("drone2", state)
        self.assertIn("target_mode", state)

        res_p = self.client.get("/api/chaos/presets")
        self.assertEqual(res_p.status_code, 200)
        presets = res_p.json()
        self.assertGreaterEqual(len(presets), 5)

    def test_chaos_motor_endpoints(self):
        """Verify updating motor thrust scaling via REST."""
        res = self.client.post("/api/chaos/motor", json={"motor_idx": 0, "multiplier": 0.40})
        self.assertEqual(res.status_code, 200)
        self.assertAlmostEqual(sim_context.chaos.drone1.motor.multipliers[0], 0.40)

        # Multi-motor update
        res = self.client.post("/api/chaos/motor", json={"multipliers": [0.6, 0.7, 0.8, 0.9]})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(sim_context.chaos.drone1.motor.multipliers, [0.6, 0.7, 0.8, 0.9])

    def test_chaos_wind_and_gust(self):
        """Verify setting 3D wind and triggering gust bursts."""
        res = self.client.post("/api/chaos/wind", json={"fx": 4.5, "fy": -1.0, "fz": 0.5, "turbulence_std": 0.3})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(sim_context.chaos.drone1.wind.steady_force, [4.5, -1.0, 0.5])

        # Gust
        res_g = self.client.post("/api/chaos/gust", json={"force": 7.0, "duration_s": 0.5})
        self.assertEqual(res_g.status_code, 200)
        self.assertTrue(sim_context.chaos.drone1.wind.gust_active)

    def test_chaos_payload_and_sensor(self):
        """Verify payload attachment and sensor noise adjustments."""
        res = self.client.post("/api/chaos/payload", json={"attached": True, "mass": 0.45})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(sim_context.chaos.drone1.payload.attached)
        self.assertAlmostEqual(sim_context.chaos.drone1.payload.mass, 0.45)

        res_s = self.client.post("/api/chaos/sensor", json={"gyro_noise_std": 0.85})
        self.assertEqual(res_s.status_code, 200)
        self.assertAlmostEqual(sim_context.chaos.drone1.sensor.gyro_noise_std, 0.85)

    def test_chaos_presets_and_clear(self):
        """Verify applying presets and clearing back to hover."""
        res = self.client.post("/api/chaos/preset", json={"preset_id": "1"})
        self.assertEqual(res.status_code, 200)
        self.assertAlmostEqual(sim_context.chaos.drone1.motor.multipliers[0], 0.40)

        # Clear
        res_c = self.client.post("/api/chaos/clear")
        self.assertEqual(res_c.status_code, 200)
        self.assertFalse(sim_context.chaos.drone1.is_any_active())

    def test_chaos_preset_multi_toggle_api(self):
        """Verify multi-selection toggling through REST API endpoints."""
        # 1. Toggle on Preset 1 (Motor 1)
        r1 = self.client.post("/api/chaos/preset", json={"preset_id": "1", "toggle": True})
        self.assertEqual(r1.status_code, 200)
        self.assertIn("1", r1.json()["chaos"]["active_presets"])

        # 2. Toggle on Preset 2 (Wind)
        r2 = self.client.post("/api/chaos/preset/toggle", json={"preset_id": "2"})
        self.assertEqual(r2.status_code, 200)
        active = r2.json()["chaos"]["active_presets"]
        self.assertIn("1", active)
        self.assertIn("2", active)

        # 3. Toggle off Preset 1
        r3 = self.client.post("/api/chaos/preset/toggle", json={"preset_id": "1"})
        self.assertEqual(r3.status_code, 200)
        active = r3.json()["chaos"]["active_presets"]
        self.assertNotIn("1", active)
        self.assertIn("2", active)

        # 4. Clear
        r_clean = self.client.post("/api/chaos/preset", json={"preset_id": "0"})
        self.assertEqual(r_clean.status_code, 200)
        self.assertEqual(r_clean.json()["chaos"]["active_presets"], [])

    def test_telemetry_latest_and_history(self):
        """Verify telemetry endpoints return valid frames."""
        # Step simulation to produce a frame
        if sim_context.sim:
            t1, t2 = sim_context.sim.step_simulation(0.0)
            sim_context.record_telemetry(t1, t2)

        res = self.client.get("/api/telemetry/latest")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("drone1", data)
        self.assertIn("drone2", data)
        self.assertIn("chaos", data)

        res_h = self.client.get("/api/telemetry/history")
        self.assertEqual(res_h.status_code, 200)
        self.assertIsInstance(res_h.json(), list)

    def test_websocket_telemetry_stream(self):
        """Verify WebSocket connection, telemetry frame streaming, and client control."""
        # Step simulation so latest_frame is populated
        if sim_context.sim:
            t1, t2 = sim_context.sim.step_simulation(0.0)
            sim_context.record_telemetry(t1, t2)

        with self.client.websocket_connect("/ws/telemetry") as ws:
            # Receive streamed telemetry frame
            frame = ws.receive_json()
            self.assertIn("drone1", frame)
            self.assertIn("drone2", frame)
            self.assertIn("chaos", frame)

            # Send client control message over WebSocket
            ws.send_json({"type": "preset", "id": "2", "target": "both"})
            ack = ws.receive_json()
            # If next message is ack or next telemetry frame
            if ack.get("type") == "ack":
                self.assertEqual(ack["action"], "preset")
                self.assertEqual(ack["id"], "2")

    def test_bridge_status_and_llm_history(self):
        """Verify bridge status endpoint includes patch age and system2 payload, and history endpoint works."""
        res = self.client.get("/api/bridge/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("drone1", data)
        self.assertIn("drone2", data)
        self.assertIn("system2", data)
        self.assertIn("patch_age_s", data["drone1"])
        self.assertIn("patch_age_s", data["drone2"])

        res_hist = self.client.get("/api/bridge/llm_history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertIsInstance(res_hist.json(), list)


if __name__ == "__main__":
    unittest.main()
