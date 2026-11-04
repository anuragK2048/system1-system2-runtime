"""Integration Tests for the Single-Drone Navigation Dashboard & Server API.

Validates:
1. Dashboard HTML serving (GET /).
2. Navigation status endpoint (GET /api/status).
3. Navigation test catalog (GET /api/navigation/tests).
4. Mission start, stop, reset, and chaos controls.
5. Telemetry snapshot and event logging endpoints.
6. Single FPV frame rendering with HUD overlay (GET /api/fpv/frame).
7. WebSocket bi-directional telemetry and command acknowledgements.

Runnable via:
    uv run python -m unittest tests/test_navigation_server_api.py
"""

import json
import unittest
from fastapi.testclient import TestClient

from runtime.server.navigation_server import app
from runtime.server.navigation_state import nav_coordinator


class TestNavigationServerAPI(unittest.TestCase):
    """Test suite for navigation server REST endpoints and WebSocket stream."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        nav_coordinator.close()

    def tearDown(self):
        nav_coordinator.stop_mission(status="idle")
        nav_coordinator.set_chaos("none", False)

    def test_root_dashboard_html_served(self):
        """Verifies GET / serves the navigation dashboard HTML."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("AUTONOMOUS TARGET NAVIGATION COCKPIT", res.text)
        self.assertIn("fpv-stream-img", res.text)
        self.assertIn("PAD ALPHA", res.text)

    def test_status_endpoint(self):
        """Verifies GET /api/status returns coordinator health and arena metadata."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("DroneSimulator", data["simulator"])
        self.assertIn("mission", data)
        self.assertIn("phase", data)

    def test_navigation_test_catalog(self):
        """Verifies GET /api/navigation/tests returns all core navigation scenarios."""
        res = self.client.get("/api/navigation/tests")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("tests", data)
        targets = [t["target"] for t in data["tests"]]
        self.assertIn("alpha", targets)
        self.assertIn("runway_poles", targets)
        self.assertIn("delta", targets)
        self.assertIn("window", targets)
        self.assertIn("arch", targets)

    def test_telemetry_snapshot(self):
        """Verifies GET /api/navigation/telemetry returns complete avionics telemetry."""
        res = self.client.get("/api/navigation/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("latest", data)
        snap = data["latest"]
        self.assertIn("pos", snap)
        self.assertIn("vel", snap)
        self.assertIn("attitude", snap)
        self.assertIn("motor_thrusts", snap)
        self.assertIn("l1_force", snap)
        self.assertIn("phase", snap)
        self.assertEqual(len(snap["motor_thrusts"]), 4)

    def test_mission_lifecycle_controls(self):
        """Verifies start, stop, and reset mission REST endpoints."""
        # 1. Start Mission
        res_start = self.client.post("/api/navigation/start", json={
            "target": "alpha",
            "use_vlm": False,
            "duration": 5.0,
        })
        self.assertEqual(res_start.status_code, 200)
        self.assertEqual(res_start.json()["status"], "started")
        self.assertTrue(nav_coordinator.is_running)

        # 2. Stop Mission
        res_stop = self.client.post("/api/navigation/stop")
        self.assertEqual(res_stop.status_code, 200)
        self.assertFalse(nav_coordinator.is_running)

        # 3. Reset Drone
        res_reset = self.client.post("/api/navigation/reset")
        self.assertEqual(res_reset.status_code, 200)
        self.assertEqual(res_reset.json()["status"], "reset")
        self.assertAlmostEqual(res_reset.json()["pos"][2], 0.80, places=2)

    def test_chaos_injection_endpoint(self):
        """Verifies live toggling of physical disturbances."""
        # Inject lateral wind
        res_wind = self.client.post("/api/navigation/chaos", json={"chaos_type": "wind", "active": True})
        self.assertEqual(res_wind.status_code, 200)
        self.assertEqual(res_wind.json()["chaos_mode"], "wind")

        # Inject motor degradation
        res_motor = self.client.post("/api/navigation/chaos", json={"chaos_type": "motor", "active": True})
        self.assertEqual(res_motor.status_code, 200)
        self.assertEqual(res_motor.json()["chaos_mode"], "motor")

        # Clear
        res_clear = self.client.post("/api/navigation/chaos", json={"chaos_type": "none", "active": False})
        self.assertEqual(res_clear.status_code, 200)
        self.assertIsNone(res_clear.json()["chaos_mode"])

    def test_single_fpv_frame_rendering(self):
        """Verifies GET /api/fpv/frame returns valid JPEG bytes with HUD overlay."""
        res = self.client.get("/api/fpv/frame?hud=true")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "image/jpeg")
        # Check JPEG SOI marker (0xFFD8)
        self.assertTrue(res.content.startswith(b"\xff\xd8"))
        self.assertGreater(len(res.content), 2000)

    def test_websocket_telemetry_stream(self):
        """Verifies WebSocket accepts connection and delivers initial telemetry packet."""
        with self.client.websocket_connect("/ws/telemetry") as ws:
            # First message should be initialization packet
            raw_msg = ws.receive_text()
            msg = json.loads(raw_msg)
            self.assertIn(msg["type"], ("init", "telemetry"))
            self.assertIn("pos", msg["data"])

            # Send a reset command over WebSocket
            ws.send_text(json.dumps({"action": "reset"}))
            ack_raw = ws.receive_text()
            ack = json.loads(ack_raw)
            # Either next telemetry tick or command ack
            self.assertIn(ack["type"], ("command_ack", "telemetry"))

    def test_camera_mode_toggle_api_and_telemetry(self):
        """Verifies GET and POST /api/navigation/camera_mode and telemetry synchronization [Cmd+F]."""
        # 1. Initial mode should be 'free' to allow cursor rotation at startup
        res_get = self.client.get("/api/navigation/camera_mode")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.json()["camera_mode"], "free")

        # 2. Toggle to 'drone_fixed'
        res_toggle1 = self.client.post("/api/navigation/camera_mode", json={"mode": "toggle"})
        self.assertEqual(res_toggle1.status_code, 200)
        self.assertEqual(res_toggle1.json()["camera_mode"], "drone_fixed")
        self.assertEqual(nav_coordinator.camera_mode, "drone_fixed")

        # 3. Telemetry snapshot reflects 'drone_fixed'
        snap = nav_coordinator.get_telemetry_snapshot()
        self.assertEqual(snap["camera_mode"], "drone_fixed")

        # 4. Toggle back to 'free'
        res_toggle2 = self.client.post("/api/navigation/camera_mode", json={"mode": "toggle"})
        self.assertEqual(res_toggle2.status_code, 200)
        self.assertEqual(res_toggle2.json()["camera_mode"], "free")
        self.assertEqual(nav_coordinator.camera_mode, "free")

        # 5. Explicit set to 'drone_fixed'
        res_set = self.client.post("/api/navigation/camera_mode", json={"mode": "drone_fixed"})
        self.assertEqual(res_set.status_code, 200)
        self.assertEqual(res_set.json()["camera_mode"], "drone_fixed")

        # Clean up back to 'free'
        nav_coordinator.toggle_camera_mode("free")
        self.assertEqual(nav_coordinator.camera_mode, "free")


if __name__ == "__main__":
    unittest.main()
