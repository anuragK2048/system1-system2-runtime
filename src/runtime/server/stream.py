"""High-Performance WebSocket Telemetry & Real-Time Control Stream.

Broadcasts dual-drone telemetry, ground-truth chaos states, and System-2 LLM events
to web clients at 30-60 Hz with sub-millisecond serialization latency.

Also accepts incoming WebSocket control messages for instant interactive adjustments.
"""

import asyncio
import json
import time
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from runtime.chaos.models import ChaosTarget
from runtime.server.state import sim_context


class TelemetryStreamManager:
    """Manages active WebSocket connections and runs the async broadcast loop."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self.broadcast_fps: float = 30.0  # 30 Hz stream (keeps M3 memory and network light)

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new client WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        # Ensure broadcast loop is running
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters a disconnected WebSocket client."""
        self.active_connections.discard(websocket)

    async def _broadcast_loop(self) -> None:
        """Continuous async loop broadcasting latest simulation telemetry to clients."""
        interval = 1.0 / self.broadcast_fps
        while len(self.active_connections) > 0:
            t0 = time.time()
            if sim_context.latest_frame:
                frame_data = sim_context.latest_frame
                # Broadcast to all connected clients
                dead_connections = []
                for ws in self.active_connections:
                    try:
                        await ws.send_json(frame_data)
                    except Exception:
                        dead_connections.append(ws)

                for dead in dead_connections:
                    self.disconnect(dead)

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)
            else:
                await asyncio.sleep(0.001)

    async def handle_client_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Processes real-time commands received directly over the WebSocket."""
        msg_type = data.get("type", "")

        # 1. Preset command: {"type": "preset", "id": "1", "target": "both"}
        if msg_type == "preset":
            preset_id = str(data.get("id", "0"))
            target_str = data.get("target", "both")
            target = ChaosTarget(target_str) if target_str in ("both", "drone1", "drone2") else ChaosTarget.BOTH
            sim_context.chaos.apply_preset(preset_id, target=target)
            await websocket.send_json({"type": "ack", "action": "preset", "id": preset_id})

        # 2. Motor multiplier: {"type": "motor", "motor_idx": 0, "multiplier": 0.4}
        elif msg_type == "motor":
            motor_idx = int(data.get("motor_idx", 0))
            multiplier = float(data.get("multiplier", 1.0))
            sim_context.chaos.set_motor_multiplier(motor_idx, multiplier)
            await websocket.send_json({"type": "ack", "action": "motor", "motor_idx": motor_idx, "multiplier": multiplier})

        # 3. Wind force: {"type": "wind", "fx": 5.0, "fy": 0.0, "fz": 0.0, "turbulence": 0.5}
        elif msg_type == "wind":
            fx = float(data.get("fx", 0.0))
            fy = float(data.get("fy", 0.0))
            fz = float(data.get("fz", 0.0))
            turb = float(data.get("turbulence", 0.0))
            sim_context.chaos.set_wind_force(fx, fy, fz, turbulence_std=turb)
            await websocket.send_json({"type": "ack", "action": "wind", "fx": fx, "fy": fy, "fz": fz})

        # 4. Wind gust pulse: {"type": "gust", "force": 6.0}
        elif msg_type == "gust":
            force = float(data.get("force", 5.0))
            sim_time = 0.0
            if sim_context.sim and sim_context.sim.data:
                sim_time = float(sim_context.sim.data.time)
            sim_context.chaos.trigger_wind_gust(force=force, sim_time=sim_time)
            await websocket.send_json({"type": "ack", "action": "gust", "force": force})

        # 5. Payload toggle: {"type": "payload", "attached": true, "mass": 0.35}
        elif msg_type == "payload":
            attached = bool(data.get("attached", True))
            mass = float(data.get("mass", 0.35))
            sim_context.chaos.set_payload_attached(attached, mass=mass)
            await websocket.send_json({"type": "ack", "action": "payload", "attached": attached})

        # 6. Sensor noise: {"type": "sensor", "gyro_std": 0.8}
        elif msg_type == "sensor":
            gyro_std = float(data.get("gyro_std", 0.0))
            sim_context.chaos.set_sensor_noise(gyro_std)
            await websocket.send_json({"type": "ack", "action": "sensor", "gyro_std": gyro_std})

        # 7. Reset: {"type": "reset"}
        elif msg_type == "reset":
            sim_context.chaos.clear_all()
            sim_context.chaos.reset_requested = True
            if sim_context.sim:
                sim_context.sim.reset_simulation()
            await websocket.send_json({"type": "ack", "action": "reset"})


stream_manager = TelemetryStreamManager()


async def telemetry_websocket_endpoint(websocket: WebSocket) -> None:
    """FastAPI WebSocket route handler."""
    await stream_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await stream_manager.handle_client_message(websocket, data)
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception:
        stream_manager.disconnect(websocket)
