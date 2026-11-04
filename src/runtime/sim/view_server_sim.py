"""Passive mirror of the headless server simulation.

Polls ``GET /api/fpv/sim-state`` on the dashboard server over one persistent
HTTP connection and copies the reported ``qpos`` into a local ``MjData``
shown in a passive MuJoCo viewer. Fetching runs in a background thread so
network jitter never stalls rendering: the viewer redraws from the latest
known state every frame. No physics is stepped here -- the server owns the
simulation; this process only mirrors it. Start the server first
(``--headless --port 8000``).
"""

import http.client
import json
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parents[3] / "assets" / "skydio_x2" / "dual_scene.xml"
STATE_HOST = "localhost"
STATE_PORT = 8000
STATE_PATH = "/api/fpv/sim-state"
FETCH_HZ = 60.0


def fetch_state(conn: http.client.HTTPConnection):
    """GETs one state payload over a persistent connection.

    Returns the decoded ``{"qpos", "time"}`` dict, or ``None`` when the
    server is up but the sim isn't running yet (HTTP 503). Raises
    ``ConnectionError``/``OSError`` when the server is unreachable.
    """
    conn.request("GET", STATE_PATH, headers={"Connection": "keep-alive"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status == 503:
        return None
    if resp.status != 200:
        raise ConnectionError("HTTP %s %s" % (resp.status, resp.reason))
    return json.loads(body.decode("utf-8"))


def _new_connection() -> http.client.HTTPConnection:
    return http.client.HTTPConnection(STATE_HOST, STATE_PORT, timeout=5.0)


class StateFetcher(threading.Thread):
    """Background 60 Hz poller; always holds the latest known sim state."""

    daemon = True

    def __init__(self) -> None:
        super().__init__(name="sim-state-fetcher")
        self._lock = threading.Lock()
        self._latest = None
        self._stop_event = threading.Event()
        self._waiting_printed = False
        self._down_printed = False
        self._polls = 0
        self._stat_wall = time.perf_counter()
        self._stat_sim0 = None
        self._stat_sim1 = None

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()

    def _store(self, payload) -> None:
        with self._lock:
            self._latest = payload

    def _note_poll(self, sim_time: float) -> None:
        self._polls += 1
        if self._stat_sim0 is None:
            self._stat_sim0 = sim_time
        self._stat_sim1 = sim_time
        now = time.perf_counter()
        if now - self._stat_wall >= 2.0:
            wall = now - self._stat_wall
            rate = self._polls / wall
            sim_rate = (self._stat_sim1 - self._stat_sim0) / wall if wall > 0 else 0.0
            print("mirror: sim %.2fx | %.1f polls/s" % (sim_rate, rate))
            self._polls = 0
            self._stat_wall = now
            self._stat_sim0 = None
            self._stat_sim1 = None

    def run(self) -> None:
        conn = _new_connection()
        interval = 1.0 / FETCH_HZ
        next_poll = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                payload = fetch_state(conn)
            except (OSError, http.client.HTTPException, ConnectionError):
                # Server unreachable mid-run: keep the last frame and retry.
                try:
                    conn.close()
                except Exception:
                    pass
                if not self._down_printed:
                    print("mirror: connection lost; retrying...")
                    self._down_printed = True
                self._stop_event.wait(1.0)
                conn = _new_connection()
                continue
            self._down_printed = False
            if payload is None:
                if not self._waiting_printed:
                    print("mirror: server up, waiting for sim...")
                    self._waiting_printed = True
                self._stop_event.wait(1.0)
                continue
            self._waiting_printed = False
            self._store(payload)
            self._note_poll(float(payload["time"]))
            next_poll += interval
            delay = next_poll - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_poll = time.perf_counter()


def main() -> None:
    # Fail fast before opening a window when the server isn't there.
    try:
        probe = _new_connection()
        try:
            fetch_state(probe)
        finally:
            probe.close()
    except (OSError, http.client.HTTPException, ConnectionError):
        print("Cannot reach server at localhost:8000 — start it first")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)
    fetcher = StateFetcher()
    fetcher.start()
    try:
        while viewer.is_running():
            latest = fetcher.latest()
            if latest is not None:
                data.qpos[:] = latest["qpos"]
                mujoco.mj_forward(model, data)
            viewer.sync()
    finally:
        fetcher.stop()


if __name__ == "__main__":
    main()
