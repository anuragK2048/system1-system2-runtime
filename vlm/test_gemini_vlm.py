"""Test Cloud Multimodal VLM via Google Gemini.

Queries Gemini 3.5 Flash-Lite with the onboard MuJoCo FPV frame and autonomous
flight routing instructions. Zero RAM usage on your local Mac.

Usage:
    uv run python vlm/test_gemini_vlm.py
"""

import os
import time
import json
import base64
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

IMAGE_PATH = Path("vlm/sample_fpv.jpg")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
MODEL = "gemini-3.5-flash-lite"

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
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_gemini_test(instruction: str = "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude."):
    if not IMAGE_PATH.exists():
        from grab_sample_frame import grab_fpv_frame
        grab_fpv_frame()

    b64_img = encode_image_base64(IMAGE_PATH)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

    user_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"MISSION INSTRUCTION: {instruction}\n"
        "TELEMETRY: Alt=1.00m, Vel=[0.0, 0.0, 0.0], Disturbance=[0.0, 0.0, 0.0]N.\n"
        "Analyze the attached FPV camera frame and emit the flight patch JSON."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": user_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    print("=" * 74)
    print(f"✨ [TESTING GEMINI CLOUD VLM] Model: {MODEL}")
    print(f"📸 Image Input: {IMAGE_PATH} (480x360 FPV frame)")
    print(f"🎯 Instruction: '{instruction}'")
    print("=" * 74)

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=15)
    latency = time.time() - t0

    if r.status_code == 200:
        raw_text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("\n" + "═" * 74)
        print(f"⚡ [GEMINI CLOUD LATENCY] Roundtrip: {latency:.2f}s ({latency * 1000:.0f} ms)")
        print("═" * 74)
        try:
            data = json.loads(raw_text)
            print("✅ [JSON PARSE SUCCESS] Verified Structured Payload:")
            print(json.dumps(data, indent=2))
        except Exception:
            print("Raw text:", raw_text)
    else:
        print(f"❌ Gemini Error ({r.status_code}):", r.text[:200])


if __name__ == "__main__":
    run_gemini_test()
