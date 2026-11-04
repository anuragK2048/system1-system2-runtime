"""Test Local Qwen2.5-VL via Ollama (OpenAI-compatible API).

Connects to local Ollama server (http://localhost:11434/v1) and queries Qwen2.5-VL
with a real 480x360 MuJoCo FPV frame and autonomous flight routing instructions.

Prerequisites:
    1. Ensure Ollama is running:
       ollama serve
    2. Pull the Qwen2.5-VL model (3B recommended for lowest latency):
       ollama pull qwen2.5vl:3b   # or: ollama pull qwen2.5vl

Usage:
    uv run python vlm/test_qwen_ollama.py
    uv run python vlm/test_qwen_ollama.py --model qwen2.5vl:3b --stream
"""

import sys
import os
import time
import json
import base64
import argparse
from pathlib import Path
from openai import OpenAI

IMAGE_PATH = Path("vlm/sample_fpv.jpg")

SYSTEM_PROMPT = """You are the System 2 Multimodal Flight Reasoner for a Skydio X2 quadrotor.
CRITICAL INSTRUCTION: Analyze the onboard FPV camera view and kinematic telemetry. Output ONLY raw valid JSON. No markdown backticks, no prose. Keep diagnosis under 20 words.

Physical Specifications:
- Mass: 1.325 kg, total hover weight W = mg ≈ 13.0 N.
- Hover thrust per motor: ~3.25 N (Motors 1-4, max thrust 13.0 N/rotor).
- Body coordinates: X=forward, Y=right, Z=up.

Rules for Control Levers & Waypoint Routing:
1. Wind & Force Disturbance (feedforward_force_offset):
   - If telemetry reports external disturbance force F_ext along X or Y (e.g. unmodeled_force_disturbance_N = [+5.2, 0, 0]), you MUST command an EQUAL AND OPPOSITE feedforward force offset: dF = -F_ext (e.g. if Fx = +5.2 N, command dFx = -5.2 N; if Fy = +3.0 N, command dFy = -3.0 N). This tilts the thrust vector into the wind to hold position. Hold waypoint at [0.0, 0.0, 0.0] and set pos_kp = 1.2.
   - If no persistent physical force (e.g. calm flight or pure gyro noise), set feedforward_force_offset: [0.0, 0.0, 0.0].

2. Actuator Degradation (motor_trim_scale):
   - Motor 1 Failure: If Motor 1 thrust is elevated (M1 > 5.0 N vs others ~2.9 N) or Motor 1 is degraded (40% power / moment disturbance), set motor_trim_scale: [2.50, 1.0, 1.0, 1.0] to restore M1 effective thrust. Tighten safety_envelope.max_tilt_deg to 18.0.
   - In nominal flight without motor failure, keep motor_trim_scale: [1.0, 1.0, 1.0, 1.0].

3. Sensor Noise & Gyro Chatter (gain_multipliers):
   - If gyro RMS > 1.0 rad/s or oscillation_detected is true, set gain_multipliers: {"att_kp": 0.40, "att_kd": 1.60, "pos_kp": 1.0, "pos_kd": 1.0} to damp high-frequency vibration chatter. Keep feedforward_force_offset at [0.0, 0.0, 0.0].
   - In nominal calm flight, keep all gain multipliers at 1.0.

4. Visual Navigation & Corridor Routing (waypoint_command):
   - Clear corridor: command forward progress target: [1.5, 0.0, 0.0] with mode: "relative_body", yaw_deg: 0.0, max_velocity: 1.0, corridor_width: 0.6.
   - Target pad landing approach: if approaching target pad at lower altitude, command descent (e.g. dz = -0.45m) and forward alignment target: [0.8, 0.0, -0.45].
   - Obstacle in flight path: if an obstacle or drone is directly ahead in the forward sightline (< 2.0m), do NOT stop or reverse; steer into the open airspace beside the obstacle with a lateral bypass waypoint target: [0.9, 0.65, 0.0] (or -0.65) and yaw_deg: 10.0.

Required JSON Structure:
{
  "diagnosis": "Brief 1-sentence physical and visual assessment.",
  "waypoint_command": {
    "mode": "relative_body",
    "target": [dx, dy, dz],
    "yaw_deg": 0.0,
    "max_velocity": 1.0,
    "corridor_width": 0.6
  },
  "feedforward_force_offset": [0.0, 0.0, 0.0],
  "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
  "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
  "safety_envelope": {"max_tilt_deg": 20.0}
}
OUTPUT ONLY RAW VALID JSON."""


def encode_image_base64(path: Path) -> str:
    """Reads image file and returns base64 data URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def run_test(
    model: str = "qwen2.5vl:3b",
    base_url: str = "http://localhost:11434/v1",
    user_instruction: str = "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.",
    stream: bool = False,
) -> None:
    # Ensure sample image exists
    if not IMAGE_PATH.exists():
        print(f"⚠️ {IMAGE_PATH} not found. Generating fresh frame from MuJoCo...")
        from grab_sample_frame import grab_fpv_frame
        grab_fpv_frame()

    image_uri = encode_image_base64(IMAGE_PATH)

    client = OpenAI(
        base_url=base_url,
        api_key="ollama",  # Dummy key required by OpenAI client
    )

    user_content = [
        {
            "type": "text",
            "text": (
                f"MISSION INSTRUCTION: {user_instruction}\n\n"
                "CURRENT TELEMETRY:\n"
                "- Altitude (AGL): 1.00 m\n"
                "- Velocity (Body): [vx=+0.02, vy=-0.01, vz=0.00] m/s\n"
                "- Attitude: Roll=0.0°, Pitch=0.0°, Yaw=0.0°\n"
                "- Disturbance Force (L1): [0.0, 0.0, 0.0] N\n"
                "- Motor Health: M1-M4 Nominal (~3.25 N hover)\n\n"
                "Analyze the attached FPV camera frame and emit the flight patch JSON."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_uri},
        },
    ]

    print("=" * 74)
    print(f"🚁 [TESTING LOCAL VLM] Model: {model} @ {base_url}")
    print(f"📸 Image Input: {IMAGE_PATH} (480x360 FPV frame)")
    print(f"🎯 Instruction: '{user_instruction}'")
    print("=" * 74)

    start_t = time.time()
    raw_output = ""
    ttft = None

    try:
        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta and ttft is None:
                    ttft = time.time() - start_t
                raw_output += delta
                sys.stdout.write(delta)
                sys.stdout.flush()
            print()
        else:
            print("⏳ Querying local model (non-streaming)...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            raw_output = response.choices[0].message.content or ""

        total_latency = time.time() - start_t

        print("\n" + "═" * 74)
        print(f"⚡ [LATENCY RESULTS] Total Latency: {total_latency:.2f}s ({total_latency * 1000:.0f} ms)")
        if ttft:
            print(f"⏱️ Time-To-First-Token (TTFT): {ttft:.3f}s ({ttft * 1000:.0f} ms)")
        print("═" * 74)

        # Parse JSON
        cleaned = raw_output.strip()
        if "```" in cleaned:
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if m:
                cleaned = m.group(1).strip()

        data = json.loads(cleaned)
        print("✅ [JSON PARSE SUCCESS] Verified Structured Payload:")
        print(json.dumps(data, indent=2))

        # Check payload completeness
        has_waypoint = "waypoint_command" in data
        has_ff = "feedforward_force_offset" in data
        has_trim = "motor_trim_scale" in data
        has_gains = "gain_multipliers" in data

        print("-" * 74)
        print(f"  • Waypoint Command: {'✅ PRESENT' if has_waypoint else '❌ MISSING'}")
        print(f"  • Feedforward Force: {'✅ PRESENT' if has_ff else '❌ MISSING'}")
        print(f"  • Motor Trim Scale: {'✅ PRESENT' if has_trim else '❌ MISSING'}")
        print(f"  • Gain Multipliers: {'✅ PRESENT' if has_gains else '❌ MISSING'}")
        print("═" * 74 + "\n")

    except Exception as e:
        print(f"\n❌ [ERROR] Failed to query local model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Is Ollama running? Run: ollama serve")
        print(f"2. Is the model downloaded? Run: ollama pull {model}")
        print("3. Check available models: curl http://localhost:11434/api/tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local Qwen2.5-VL via Ollama")
    parser.add_argument("--model", type=str, default="qwen2.5vl:3b", help="Ollama model tag")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", help="Ollama API base URL")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--instruction", type=str, default="Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.")
    args = parser.parse_args()

    run_test(
        model=args.model,
        base_url=args.base_url,
        user_instruction=args.instruction,
        stream=args.stream,
    )
