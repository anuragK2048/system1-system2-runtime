"""Client to query the deployed Modal VLM endpoint over HTTP.

Usage:
    uv run python vlm/test_modal_client.py --url https://<your-username>--drone-qwen-vlm-dronevlmservice-predict.modal.run
"""

import sys
import time
import json
import base64
import argparse
from pathlib import Path
import requests

IMAGE_PATH = Path("vlm/sample_fpv.jpg")


def query_modal_vlm(
    endpoint_url: str,
    instruction: str = "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.",
    image_path: Path = IMAGE_PATH,
):
    if not image_path.exists():
        from grab_sample_frame import grab_fpv_frame
        grab_fpv_frame()

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": b64,
        "instruction": instruction,
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_mps": [0.02, -0.01, 0.0],
            "disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
        },
    }

    print("=" * 74)
    print(f"🌐 [TESTING MODAL DEDICATED GPU ENDPOINT]")
    print(f"🔗 URL: {endpoint_url}")
    print(f"📸 Image: {image_path}")
    print(f"🎯 Instruction: '{instruction}'")
    print("=" * 74)

    t0 = time.time()
    try:
        r = requests.post(endpoint_url, json=payload, timeout=25)
        total_lat = time.time() - t0
        if r.status_code == 200:
            res = r.json()
            gpu_time = res.get("inference_duration_s", 0.0)
            patch = res.get("patch", {})
            print("\n" + "═" * 74)
            print(f"⚡ [LATENCY RESULTS] Network Roundtrip: {total_lat:.2f}s ({total_lat * 1000:.0f} ms)")
            print(f"🚀 Remote GPU Compute Time: {gpu_time:.3f}s ({gpu_time * 1000:.0f} ms)")
            print("═" * 74)
            print(json.dumps(patch, indent=2))
            print("═" * 74)
        else:
            print(f"❌ Error ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Modal Drone VLM")
    parser.add_argument("--url", type=str, required=True, help="Modal HTTPS endpoint URL")
    parser.add_argument("--instruction", type=str, default="Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.")
    args = parser.parse_args()

    query_modal_vlm(endpoint_url=args.url, instruction=args.instruction)
