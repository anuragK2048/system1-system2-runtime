"""Gemini Live API Client for Autonomous Quadrotor Flight Reasoner.

Uses Google GenAI SDK (google-genai >= 2.3.0) to stream onboard camera frames
and flight telemetry over bidirectional WebSockets to Gemini Live models:
  - gemini-3.8-live (Default, low-latency native audio/multimodal)
  - gemini-3.1-flash-live-preview (Legacy preview)
  - gemini-3.8-live-extended-thinking (Deep background reasoning)

Features:
  - Real-time multimodal streaming via WebSockets
  - Non-blocking & blocking function calling (Tool Execution) for zero-latency structured patches
  - High-precision latency breakdown:
      * Handshake Latency (WebSocket connection negotiation)
      * Inference Latency (Time to tool execution / decision)
      * Total Turnaround Latency
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from google import genai
from google.genai import types

DEFAULT_GEMINI_KEY = "YOUR_GEMINI_API_KEY"

LIVE_SYSTEM_PROMPT = """You are the System 2 Multimodal Flight Reasoner for an autonomous Skydio X2 quadrotor.
CRITICAL INSTRUCTION: Analyze the onboard FPV camera frame (with HUD overlay) and telemetry. You MUST immediately execute emit_flight_patch with your decision. Keep diagnosis under 20 words.

[VEHICLE PHYSICAL PARAMETERS]
- Mass: 1.325 kg, total hover weight W = mg ≈ 13.0 N.
- Rotor thrust: 4 motors, hover thrust ~3.25 N per motor (max 13.0 N/rotor).
- Frame axes: +X forward, +Y right (lateral), +Z up (altitude).

[SPATIAL SECTOR MAP (Global Arena Coordinates)]
- SECTOR_FORWARD (Heading ~0°):
  • Target Pad Alpha: Primary forward landing pad (Orange/Amber rings). Forward target: [2.0, 0.0, 0.0], flight_phase: "CORRIDOR_APPROACH".
  • Elevated Pedestal Delta: Raised hover platform (0.70m height, Yellow disc). Forward target: [2.5, 0.0, 0.0], flight_phase: "CORRIDOR_APPROACH".
  • Target Pad Bravo: Left-flank pad (Cyan/Blue rings). Vector forward-left: target [2.2, 0.8, 0.0], yaw_deg: +15.0°, flight_phase: "CORRIDOR_APPROACH".
  • Target Pad Charlie: Right-flank pad (Emerald/Green rings). Vector forward-right: target [2.2, -0.8, 0.0], yaw_deg: -15.0°, flight_phase: "CORRIDOR_APPROACH".
  • Runway Hazard Pylons (y=±0.9m): Traversal target [2.0, 0.0, 0.0], flight_phase: "CORRIDOR_APPROACH".
- SECTOR_LEFT (Heading ~+90°): Window Aperture (0.0, +5.0, 1.1m). If active objective is left window but heading is ~0°, command in-place yaw rotation yaw_deg: +90.0, target: [0.0, 0.0, 0.0], flight_phase: "SECTOR_SCAN". When facing window, ingress target: [1.8, 0.0, 0.0], flight_phase: "APERTURE_INGRESS".
- SECTOR_RIGHT (Heading ~-90°): Arch Aperture (0.0, -5.0, 0.9m). If active objective is right arch but heading is ~0°, command in-place yaw rotation yaw_deg: -90.0, target: [0.0, 0.0, 0.0], flight_phase: "SECTOR_SCAN". When facing arch, ingress target: [1.8, 0.0, 0.0], flight_phase: "APERTURE_INGRESS".

[STATION-KEEPING HOVER]
- Precision hover between runway poles or atop pedestal disc: target [0.0, 0.0, 0.0], flight_phase: "HOVER_HOLD".

[CHAOS & DISTURBANCE REJECTION]
1. Wind Disturbance (feedforward_force_x/y):
   - Command equal and opposite feedforward: dF = -F_ext (e.g. if Fx = +5.2N, feedforward_force_x = -5.2N; if Fy = +4.8N, feedforward_force_y = -4.8N). Set flight_phase: "HOVER_HOLD", target: [0.0, 0.0, 0.0], gain_pos_kp = 1.2.
   - In calm flight: feedforward_force: 0.0.
2. Motor Degradation (motor_trim_m1):
   - If Motor 1 power loss / elevated M1, set motor_trim_m1: 2.50, max_tilt_deg: 18.0, target: [0.0, 0.0, 0.0], flight_phase: "HOVER_HOLD".
   - Nominal flight: motor_trim_m1: 1.0.
3. Gyro Chatter / Oscillation:
   - If gyro RMS > 1.0 rad/s or oscillation, set gain_att_kp: 0.40, gain_att_kd: 1.60.
4. Obstacle Avoidance:
   - Obstacle dead-ahead in corridor: lateral bypass target [0.9, 0.65, 0.0] or [0.9, -0.65, 0.0], yaw_deg: 10.0, flight_phase: "CORRIDOR_APPROACH"."""

LIVE_TOOL_DECLARATION = {
    "function_declarations": [
        {
            "name": "emit_flight_patch",
            "description": "Commands 3D navigation waypoint, flight phase, and physical chaos compensation for Skydio X2 quadrotor.",
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "diagnosis": {
                        "type": "STRING",
                        "description": "Brief 1-sentence physical and visual assessment.",
                    },
                    "flight_phase": {
                        "type": "STRING",
                        "description": "Flight phase state: 'CORRIDOR_APPROACH', 'HOVER_HOLD', 'SECTOR_SCAN', or 'APERTURE_INGRESS'.",
                    },
                    "target_dx": {
                        "type": "NUMBER",
                        "description": "Relative forward displacement target in meters (body X).",
                    },
                    "target_dy": {
                        "type": "NUMBER",
                        "description": "Relative lateral displacement target in meters (body Y, positive right/cyan-pad, negative emerald-pad).",
                    },
                    "target_dz": {
                        "type": "NUMBER",
                        "description": "Relative vertical displacement target in meters (body Z, positive up, negative descent).",
                    },
                    "yaw_deg": {
                        "type": "NUMBER",
                        "description": "Yaw heading turn in degrees.",
                    },
                    "max_velocity": {
                        "type": "NUMBER",
                        "description": "Maximum corridor flight velocity in m/s (nominal 1.0).",
                    },
                    "feedforward_force_x": {
                        "type": "NUMBER",
                        "description": "Opposing feedforward force in X (N). Must equal -F_ext_x (e.g. if wind Fx=+5.2N, feedforward_force_x=-5.2N; if calm, 0.0).",
                    },
                    "feedforward_force_y": {
                        "type": "NUMBER",
                        "description": "Opposing feedforward force in Y (N). Must equal -F_ext_y.",
                    },
                    "feedforward_force_z": {
                        "type": "NUMBER",
                        "description": "Opposing feedforward force in Z (N).",
                    },
                    "motor_trim_m1": {
                        "type": "NUMBER",
                        "description": "Motor 1 trim multiplier (nominal 1.0; 2.50 if Motor 1 is degraded/failing).",
                    },
                    "gain_pos_kp": {
                        "type": "NUMBER",
                        "description": "Position Kp gain multiplier (nominal 1.0, 1.2 in wind).",
                    },
                    "gain_att_kp": {
                        "type": "NUMBER",
                        "description": "Attitude Kp gain multiplier (nominal 1.0; 0.40 if gyro chatter/noise).",
                    },
                    "gain_att_kd": {
                        "type": "NUMBER",
                        "description": "Attitude Kd gain multiplier (nominal 1.0; 1.60 if gyro chatter/noise).",
                    },
                    "max_tilt_deg": {
                        "type": "NUMBER",
                        "description": "Safety envelope maximum tilt angle in degrees (nominal 20.0, 18.0 if degraded motor, 25.0 in high wind).",
                    },
                },
                "required": ["diagnosis", "flight_phase", "target_dx", "target_dy", "target_dz"],
            },
        }
    ]
}


def convert_tool_args_to_flight_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    """Converts tool call arguments into the benchmark's standardized patch dictionary."""
    return {
        "diagnosis": args.get("diagnosis", "Autonomous multimodal flight patch"),
        "flight_phase": args.get("flight_phase", "CORRIDOR_APPROACH"),
        "waypoint_command": {
            "mode": "relative_body",
            "target": [
                float(args.get("target_dx", 0.0)),
                float(args.get("target_dy", 0.0)),
                float(args.get("target_dz", 0.0)),
            ],
            "yaw_deg": float(args.get("yaw_deg", 0.0)),
            "max_velocity": float(args.get("max_velocity", 1.0)),
            "corridor_width": 0.6,
        },
        "feedforward_force_offset": [
            float(args.get("feedforward_force_x", 0.0)),
            float(args.get("feedforward_force_y", 0.0)),
            float(args.get("feedforward_force_z", 0.0)),
        ],
        "gain_multipliers": {
            "pos_kp": float(args.get("gain_pos_kp", 1.0)),
            "pos_kd": 1.0,
            "att_kp": float(args.get("gain_att_kp", 1.0)),
            "att_kd": float(args.get("gain_att_kd", 1.0)),
        },
        "motor_trim_scale": [
            float(args.get("motor_trim_m1", 1.0)),
            1.0,
            1.0,
            1.0,
        ],
        "safety_envelope": {
            "max_tilt_deg": float(args.get("max_tilt_deg", 20.0)),
        },
    }


async def query_gemini_live_async(
    image_path: Path,
    system_prompt: str,
    mission_instruction: str,
    telemetry: Dict[str, Any],
    model: str = "gemini-3.8-live",
    api_key: Optional[str] = None,
    timeout_s: float = 25.0,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Connects to Gemini Live API over WebSockets and executes a flight patch tool call.

    Returns:
        (flight_patch_dict, latency_dict)
        where latency_dict = {"handshake": s, "inference": s, "total": s}
    """
    key = api_key or os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)
    client = genai.Client(api_key=key)

    # For extended thinking, tool behavior must be NON_BLOCKING
    tools = [dict(LIVE_TOOL_DECLARATION)]
    kwargs: Dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "tools": tools,
        "system_instruction": types.Content(
            parts=[types.Part(text=LIVE_SYSTEM_PROMPT)]
        ),
    }

    if "extended-thinking" in model:
        tools[0]["function_declarations"][0]["behavior"] = "NON_BLOCKING"
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    config = types.LiveConnectConfig(**kwargs)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    telemetry_str = json.dumps(telemetry, indent=2)
    prompt_text = (
        f"MISSION INSTRUCTION: {mission_instruction}\n\n"
        f"TELEMETRY STATE:\n{telemetry_str}\n\n"
        "Analyze the attached onboard FPV camera view and call emit_flight_patch immediately."
    )

    t_conn_start = time.time()
    async with client.aio.live.connect(model=model, config=config) as session:
        t_handshake = time.time() - t_conn_start

        t_send = time.time()
        parts = [
            types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
            types.Part(text=prompt_text),
        ]
        await session.send_client_content(
            turns=types.Content(role="user", parts=parts),
            turn_complete=True,
        )

        args = None
        t_inference = 0.0

        async for response in session.receive():
            now = time.time()
            if response.tool_call:
                t_inference = now - t_send
                args = response.tool_call.function_calls[0].args
                break
            if response.server_content and response.server_content.turn_complete:
                t_inference = now - t_send
                break

    if args is None:
        raise RuntimeError(
            f"Gemini Live ({model}) ended turn without calling emit_flight_patch."
        )

    t_total = t_handshake + t_inference
    latencies = {
        "handshake": t_handshake,
        "inference": t_inference,
        "total": t_total,
    }

    patch = convert_tool_args_to_flight_patch(args)
    return patch, latencies


def query_gemini_live(
    image_path: Path,
    system_prompt: str,
    mission_instruction: str,
    telemetry: Dict[str, Any],
    model: str = "gemini-3.8-live",
    api_key: Optional[str] = None,
    timeout_s: float = 25.0,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Synchronous wrapper for query_gemini_live_async."""
    import asyncio
    return asyncio.run(
        query_gemini_live_async(
            image_path=image_path,
            system_prompt=system_prompt,
            mission_instruction=mission_instruction,
            telemetry=telemetry,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    )
