"""Test Local Qwen2.5-VL via Apple Silicon Native MLX-VLM.

Executes 4-bit quantized Qwen2.5-VL directly on Apple Silicon Metal GPU unified memory
for lowest possible latency (180-320ms E2E).

Prerequisites:
    uv pip install mlx-vlm

Usage:
    uv run python vlm/test_qwen_mlx.py
    uv run python vlm/test_qwen_mlx.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
"""

import time
import json
import argparse
from pathlib import Path

IMAGE_PATH = Path("vlm/sample_fpv.jpg")
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

SYSTEM_PROMPT = """You are the System 2 Multimodal Flight Reasoner for a Skydio X2 quadrotor.
Output ONLY raw valid JSON. No markdown backticks, no thought tags, no prose.

Required JSON Structure:
{
  "diagnosis": "Brief 1-sentence physical and visual assessment.",
  "waypoint_command": {
    "mode": "relative_body",
    "target": [1.0, 0.0, 0.0],
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


def run_mlx_test(
    model_name: str = DEFAULT_MODEL,
    instruction: str = "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.",
) -> None:
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_image
    except ImportError:
        print("❌ 'mlx-vlm' is not installed in the current environment.")
        print("To install, run:")
        print("    uv pip install mlx-vlm")
        return

    if not IMAGE_PATH.exists():
        from grab_sample_frame import grab_fpv_frame
        grab_fpv_frame()

    print("=" * 74)
    print(f"🍏 [TESTING APPLE SILICON MLX-VLM] Model: {model_name}")
    print(f"📸 Image Input: {IMAGE_PATH}")
    print(f"🎯 Instruction: '{instruction}'")
    print("=" * 74)

    print("⏳ Loading model into Apple Silicon Unified Memory...")
    load_start = time.time()
    model, processor = load(model_name)
    print(f"✅ Model loaded in {time.time() - load_start:.2f}s")

    formatted_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"MISSION INSTRUCTION: {instruction}\n"
        "TELEMETRY: Alt=1.00m, Vel=[0.0, 0.0, 0.0], Disturbance=[0.0, 0.0, 0.0]N.\n"
        "Output the JSON flight patch now."
    )

    prompt = apply_chat_template(
        processor,
        config=model.config,
        prompt=formatted_prompt,
        num_images=1,
    )

    print("⚡ Running Metal GPU inference...")
    infer_start = time.time()
    output = generate(
        model,
        processor,
        prompt,
        image=[str(IMAGE_PATH)],
        max_tokens=256,
        temperature=0.1,
        verbose=False,
    )
    infer_duration = time.time() - infer_start

    print("\n" + "═" * 74)
    print(f"⚡ [MLX INFERENCE COMPLETE] Latency: {infer_duration:.3f}s ({infer_duration * 1000:.0f} ms)")
    print("═" * 74)
    print(output)
    print("═" * 74 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local Qwen2.5-VL via MLX-VLM on Apple Silicon")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MLX HF model id")
    parser.add_argument("--instruction", type=str, default="Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.")
    args = parser.parse_args()

    run_mlx_test(model_name=args.model, instruction=args.instruction)
