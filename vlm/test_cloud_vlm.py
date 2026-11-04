"""Test Cloud Multimodal VLMs (Fireworks AI / Gemini / OpenAI).

Benchmark cloud VLM latency and structured JSON output using an onboard
MuJoCo FPV frame and autonomous flight routing instructions.

Supported Providers:
    1. Fireworks AI:
       - Model: accounts/fireworks/models/qwen2p5-vl-7b-instruct (Top Recommendation)
       - Uses FIREWORK_API_KEY or FIREWORKS_API_KEY
    2. Google Gemini:
       - Model: gemini-3.8-flash (or gemini-2.5-flash)
       - Uses GEMINI_API_KEY
    3. OpenAI:
       - Model: gpt-4o-mini
       - Uses OPENAI_API_KEY

Usage:
    # Test Fireworks AI (Qwen2.5-VL 7B):
    uv run python vlm/test_cloud_vlm.py --provider fireworks

    # Test with custom instruction:
    uv run python vlm/test_cloud_vlm.py --provider fireworks --instruction "Navigate to the orange pad, hover at 0.5m"
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load local .env
load_dotenv()

IMAGE_PATH = Path("vlm/sample_fpv.jpg")

SYSTEM_PROMPT = """You are the System 2 Multimodal Flight Reasoner for a Skydio X2 quadrotor.
CRITICAL INSTRUCTION: Analyze the FPV camera view and kinematic telemetry. Output ONLY raw valid JSON. No markdown backticks, no thought tags, no prose.

Physical Specifications:
- Mass: 1.325 kg, Hover weight: ~13.0 N (~3.25 N/motor).
- Max tilt envelope: 25.0°.

Guidance on Levers:
- waypoint_command:
  * mode: "relative_body" (displacement vector [dx, dy, dz] in meters relative to drone nose).
  * target: [dx_forward, dy_right, dz_up]
  * yaw_deg: delta heading adjustment in degrees (+ = turn right, - = turn left).
  * max_velocity: speed ceiling in m/s (default: 0.8 to 1.2 m/s).
  * corridor_width: maximum allowable lateral boundary in meters (default: 0.6m).
- feedforward_force_offset: [dFx, dFy, dFz] in Newtons (neutral: [0, 0, 0]).
- motor_trim_scale: [m1, m2, m3, m4] (neutral: [1.0, 1.0, 1.0, 1.0]).
- gain_multipliers: {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0}.
- safety_envelope: {"max_tilt_deg": 22.0}.

Required JSON Schema:
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


def run_cloud_test(
    provider: str = "fireworks",
    model: str = None,
    api_key: str = None,
    instruction: str = "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.",
) -> None:
    if not IMAGE_PATH.exists():
        print(f"⚠️ {IMAGE_PATH} not found. Generating fresh frame from MuJoCo...")
        from grab_sample_frame import grab_fpv_frame
        grab_fpv_frame()

    image_uri = encode_image_base64(IMAGE_PATH)

    # Resolve API Key & Endpoint
    provider = provider.lower().strip()
    if provider in ("fireworks", "firework"):
        key = api_key or os.getenv("FIREWORK_API_KEY") or os.getenv("FIREWORKS_API_KEY")
        if not key:
            # Check commented line in .env
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if "FIREWORK_API_KEY" in line and "=" in line:
                        key = line.split("=")[1].strip()
                        break
        if not key:
            raise ValueError("FIREWORK_API_KEY not found in environment or .env file!")
        base_url = "https://api.fireworks.ai/inference/v1"
        model = model or "accounts/fireworks/models/qwen2p5-vl-7b-instruct"

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found!")
        base_url = None
        model = model or "gpt-4o-mini"

    else:
        raise ValueError(f"Provider '{provider}' not directly supported in this quick benchmark. Use 'fireworks' or 'openai'.")

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=key)

    user_content = [
        {
            "type": "text",
            "text": (
                f"MISSION INSTRUCTION: {instruction}\n\n"
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
    print(f"☁️ [TESTING CLOUD VLM] Provider: {provider.upper()} | Model: {model}")
    print(f"📸 Image Input: {IMAGE_PATH} (480x360 FPV frame)")
    print(f"🎯 Instruction: '{instruction}'")
    print("=" * 74)

    start_t = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        total_latency = time.time() - start_t
        raw_output = response.choices[0].message.content or ""

        print("\n" + "═" * 74)
        print(f"⚡ [CLOUD LATENCY RESULTS] Roundtrip E2E: {total_latency:.2f}s ({total_latency * 1000:.0f} ms)")
        if response.usage:
            print(f"📊 Tokens: Input={response.usage.prompt_tokens} | Output={response.usage.completion_tokens} | Total={response.usage.total_tokens}")
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
        print(f"\n❌ [ERROR] Cloud API call failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cloud VLM (Fireworks / OpenAI)")
    parser.add_argument("--provider", type=str, default="fireworks", choices=["fireworks", "openai"])
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--api-key", type=str, default=None, help="API Key override")
    parser.add_argument("--instruction", type=str, default="Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.")
    args = parser.parse_args()

    run_cloud_test(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        instruction=args.instruction,
    )
