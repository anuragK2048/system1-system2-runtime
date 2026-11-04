"""Benchmark Evaluator for Single-Drone Target Navigation Multimodal Flight Reasoners.

Evaluates multimodal VLMs (or ideal ground truth) across 7 single-drone FPV targeting scenarios
defined in vlm/test_cases_single.json.

Supported Backends:
    1. --backend ground_truth   (Evaluates ideal ground truth to verify harness and benchmarks)
    2. --backend gemini         (Evaluates Google Gemini multimodal models, e.g. gemini-2.5-flash / gemini-1.5-flash)
    3. --backend ollama         (Evaluates local Ollama Qwen2.5-VL)

Evaluation Metrics:
    - Target Waypoint Accuracy (dx, dy, dz range checks & Euclidean setpoint error)
    - Control Lever Correctness (Feedforward force offset, Motor trim scale, Gain multipliers)
    - Safety Envelope (Max tilt angle limits)
    - Latency (End-to-end wall-clock time)

Usage:
    uv run python vlm/evaluate_target_navigation.py --backend ground_truth
    uv run python vlm/evaluate_target_navigation.py --backend gemini
    uv run python vlm/evaluate_target_navigation.py --backend ollama --model qwen2.5vl:3b
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import requests
import numpy as np

DEFAULT_TEST_FILE = Path("vlm/test_cases_single.json")
DEFAULT_GEMINI_KEY = "YOUR_GEMINI_API_KEY"

SYSTEM_PROMPT = """You are the System 2 Multimodal Flight Reasoner for an autonomous Skydio X2 quadrotor.
CRITICAL INSTRUCTION: Analyze the onboard forward-looking FPV camera frame and high-rate kinematic telemetry.
Output ONLY raw valid JSON. No markdown backticks (no ```json), no explanatory prose. Keep diagnosis concise.

Vehicle Physical Parameters:
- Mass: 1.325 kg, total hover weight W = mg ≈ 13.0 N.
- Quadrotor rotor configuration: 4 motors, hover thrust ~3.25 N per motor (max 13.0 N/rotor).
- Frame axes: +X forward, +Y right (lateral), +Z up (altitude).

Target Objects in Arena:
1. Target Pad Alpha: Primary forward landing pad with Safety Orange & Amber concentric rings at x=5.5m.
2. Elevated Pedestal Delta: Raised hover pedestal platform (0.70m height) with bright yellow caution landing disc at x=8.0m.
3. Target Pad Bravo: Left-flank waypoint pad with Cyan / Neon Blue rings at x=11.0m, y=+2.2m.
4. Target Pad Charlie: Right-flank waypoint pad with Emerald / Neon Green rings at x=11.0m, y=-2.2m.
5. Corridor Pylons: Vertical hazard obstacles framing the narrow passage at x=3.5m, y=±0.9m.

Rules for Control Levers & Waypoint Navigation:
1. Visual Navigation (waypoint_command):
   - Approaching / Aligning with Target Pad Alpha: Command forward progress along the centered corridor (e.g. target: [2.0, 0.0, 0.0], mode: "relative_body", yaw_deg: 0.0).
   - Elevated Pedestal Hover: Command forward target onto the elevated platform disc (e.g. target: [2.5, 0.0, 0.0], dz within ±0.2m of hover plane).
   - Lateral Offset to Pad Bravo (Cyan): Vector forward-left with positive Y offset (target: [2.2, 0.8, 0.0], yaw_deg: +15.0°).
   - Lateral Offset to Pad Charlie (Emerald): Vector forward-right with negative Y offset (target: [2.2, -0.8, 0.0], yaw_deg: -15.0°).
   - Pylon Gate Traversal: Navigate directly between pylons with strict centering (target: [2.0, 0.0, 0.0], corridor_width: 0.5m, dy: 0.0).

2. Wind & External Force Disturbance (feedforward_force_offset):
   - If telemetry reports external disturbance force F_ext (e.g. unmodeled_force_disturbance_N = [+5.2, 0, 0]), you MUST command an EQUAL AND OPPOSITE feedforward force offset: dF = -F_ext (e.g. dFx = -5.2 N). Hold station waypoint at [0.0, 0.0, 0.0] and set pos_kp = 1.2.
   - In calm unperturbed flight, feedforward_force_offset MUST be [0.0, 0.0, 0.0].

3. Actuator Degradation (motor_trim_scale):
   - If Motor 1 is degraded (40% power efficiency / elevated M1 command), set motor_trim_scale: [2.50, 1.0, 1.0, 1.0] to equalize thrust. Tighten safety_envelope.max_tilt_deg to 18.0.
   - In nominal flight without motor failure, motor_trim_scale MUST be [1.0, 1.0, 1.0, 1.0].

Required JSON Output Structure:
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
    """Reads image file and converts to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_gemini(
    image_path: Path, prompt_text: str, model: str = "gemini-3.5-flash-lite"
) -> Tuple[str, float]:
    """Queries Google Gemini multimodal API with an onboard FPV frame."""
    api_key = os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_KEY)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    b64_img = encode_image_base64(image_path)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=25)
    latency = time.time() - t0
    if resp.status_code == 200:
        candidates = resp.json().get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts:
                return parts[0].get("text", ""), latency
        raise RuntimeError(f"Gemini empty response: {resp.text[:200]}")
    raise RuntimeError(f"Gemini API returned {resp.status_code}: {resp.text[:300]}")


def query_ollama(
    image_path: Path,
    prompt_text: str,
    model: str = "qwen2.5vl:3b",
    base_url: str = "http://localhost:11434/v1",
) -> Tuple[str, float]:
    """Queries local Ollama Qwen2.5-VL via OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")
    uri = f"data:image/jpeg;base64,{encode_image_base64(image_path)}"
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            },
        ],
        temperature=0.1,
    )
    latency = time.time() - t0
    return resp.choices[0].message.content or "", latency


def evaluate_rules(parsed: dict, rules: list) -> Tuple[bool, List[str]]:
    """Evaluates output dictionary against test case validation rules."""
    passed = True
    reasons: List[str] = []

    for rule in rules:
        field = rule["field"]
        rtype = rule["type"]
        desc = rule.get("desc", field)

        # Extract nested value
        val = parsed
        try:
            for part in field.replace("]", "").replace("[", ".").split("."):
                val = val[int(part)] if part.isdigit() else val[part]
        except Exception:
            passed = False
            reasons.append(f"❌ Missing key: {field} ({desc})")
            continue

        if rtype == "range":
            num_val = float(val)
            if not (rule["min"] <= num_val <= rule["max"]):
                passed = False
                reasons.append(
                    f"❌ {field}={num_val:.2f} out of range [{rule['min']}, {rule['max']}] ({desc})"
                )

        elif rtype == "abs_max":
            num_val = abs(float(val))
            if num_val > rule["max"]:
                passed = False
                reasons.append(
                    f"❌ |{field}|={num_val:.2f} exceeds limit {rule['max']} ({desc})"
                )

        elif rtype == "abs_min":
            num_val = abs(float(val))
            if num_val < rule["min"]:
                passed = False
                reasons.append(
                    f"❌ |{field}|={num_val:.2f} below required magnitude {rule['min']} ({desc})"
                )

        elif rtype == "zero_vector":
            if not isinstance(val, list) or len(val) != 3:
                passed = False
                reasons.append(f"❌ {field} must be a 3-element vector ({desc})")
            elif any(abs(float(x)) > 0.3 for x in val):
                passed = False
                reasons.append(f"❌ {field}={val} expected near-zero vector ({desc})")

        elif rtype == "ones_vector":
            if not isinstance(val, list) or len(val) != 4:
                passed = False
                reasons.append(f"❌ {field} must be a 4-element vector ({desc})")
            elif any(abs(float(x) - 1.0) > 0.2 for x in val):
                passed = False
                reasons.append(f"❌ {field}={val} expected nominal 1.0 trim ({desc})")

    return passed, reasons


def compute_metrics(parsed: dict, gt: dict) -> Dict[str, Any]:
    """Computes precision metrics comparing parsed patch with ground truth."""
    metrics: Dict[str, Any] = {}

    # Target Waypoint Euclidean Error
    if (
        "waypoint_command" in parsed
        and isinstance(parsed["waypoint_command"], dict)
        and "target" in parsed["waypoint_command"]
        and "waypoint_command" in gt
    ):
        try:
            p_target = np.array(parsed["waypoint_command"]["target"], dtype=float)
            gt_target = np.array(gt["waypoint_command"]["target"], dtype=float)
            error_vec = p_target - gt_target
            metrics["target_error_euclidean_m"] = float(np.linalg.norm(error_vec))
            metrics["target_dx_err_m"] = float(error_vec[0])
            metrics["target_dy_err_m"] = float(error_vec[1])
            metrics["target_dz_err_m"] = float(error_vec[2])
        except Exception:
            metrics["target_error_euclidean_m"] = None

    # Disturbance Rejection Error
    if "feedforward_force_offset" in parsed and "feedforward_force_offset" in gt:
        try:
            p_ff = np.array(parsed["feedforward_force_offset"], dtype=float)
            gt_ff = np.array(gt["feedforward_force_offset"], dtype=float)
            metrics["feedforward_error_N"] = float(np.linalg.norm(p_ff - gt_ff))
        except Exception:
            metrics["feedforward_error_N"] = None

    # Actuator Trim Error
    if "motor_trim_scale" in parsed and "motor_trim_scale" in gt:
        try:
            p_trim = np.array(parsed["motor_trim_scale"], dtype=float)
            gt_trim = np.array(gt["motor_trim_scale"], dtype=float)
            metrics["trim_error_norm"] = float(np.linalg.norm(p_trim - gt_trim))
        except Exception:
            metrics["trim_error_norm"] = None

    return metrics


def run_benchmark(
    backend: str = "ground_truth",
    model: Optional[str] = None,
    test_file: Path = DEFAULT_TEST_FILE,
    ollama_url: str = "http://localhost:11434/v1",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Runs target navigation benchmark evaluation across the suite."""
    if not test_file.exists():
        print(f"⚠️ Test file {test_file} not found. Generating single-drone suite first...")
        from vlm.generate_single_drone_suite import generate_suite

        generate_suite()

    with open(test_file) as f:
        test_cases = json.load(f)

    print("\n" + "═" * 80)
    print("🎯 [AUTONOMOUS TARGET NAVIGATION & HOVER VLM EVALUATOR]")
    print(f"📡 Backend: {backend.upper()} | Model: {model or 'default'}")
    print(f"📁 Test Suite: {test_file} ({len(test_cases)} scenarios)")
    print("═" * 80)

    results: List[Dict[str, Any]] = []

    for tc in test_cases:
        tc_id = tc["id"]
        tc_name = tc["name"]
        cat = tc["category"]
        img_p = Path(tc["image_path"])

        if not img_p.exists():
            print(f"⚠️ Image {img_p} missing. Generating suite...")
            from vlm.generate_single_drone_suite import generate_suite

            generate_suite()

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"MISSION INSTRUCTION: {tc['instruction']}\n\n"
            f"TELEMETRY STATE:\n{json.dumps(tc['telemetry'], indent=2)}\n\n"
            "Analyze the attached onboard FPV camera view and emit the flight patch JSON."
        )

        print(f"\n▶ Running [{tc_id}] {tc_name} ({cat})...")

        raw_json_str = ""
        latency_s = 0.0

        if backend == "ground_truth":
            ideal_patch = dict(tc["ground_truth"])
            raw_json_str = json.dumps(ideal_patch)
            latency_s = 0.001

        elif backend == "gemini":
            target_model = model or "gemini-3.5-flash-lite"
            try:
                raw_json_str, latency_s = query_gemini(img_p, prompt_text, model=target_model)
            except Exception as e:
                print(f"   ❌ Call failed: {e}")
                results.append({
                    "id": tc_id,
                    "name": tc_name,
                    "pass": False,
                    "latency": 0.0,
                    "notes": [f"API Error: {e}"],
                    "metrics": {},
                })
                continue

        elif backend == "ollama":
            target_model = model or "qwen2.5vl:3b"
            try:
                raw_json_str, latency_s = query_ollama(
                    img_p, prompt_text, model=target_model, base_url=ollama_url
                )
            except Exception as e:
                print(f"   ❌ Call failed: {e}")
                results.append({
                    "id": tc_id,
                    "name": tc_name,
                    "pass": False,
                    "latency": 0.0,
                    "notes": [f"Ollama Error: {e}"],
                    "metrics": {},
                })
                continue

        # Parse JSON
        cleaned = raw_json_str.strip()
        if "```" in cleaned:
            import re

            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if m:
                cleaned = m.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            is_valid, reasons = evaluate_rules(parsed, tc["validation_rules"])
            metrics = compute_metrics(parsed, tc["ground_truth"])
            status_str = "✅ PASS" if is_valid else "❌ FAIL"
            lat_ms = latency_s * 1000.0

            err_str = ""
            if metrics.get("target_error_euclidean_m") is not None:
                err_str = f" | TargetErr: {metrics['target_error_euclidean_m']:.2f}m"

            print(f"   {status_str} | Latency: {latency_s:.2f}s ({lat_ms:.0f} ms){err_str}")
            if verbose or not is_valid:
                for r in reasons:
                    print(f"      {r}")

            results.append({
                "id": tc_id,
                "name": tc_name,
                "pass": is_valid,
                "latency": latency_s,
                "notes": reasons,
                "metrics": metrics,
            })
        except Exception as e:
            print(f"   ❌ JSON Parse Error: {e}")
            results.append({
                "id": tc_id,
                "name": tc_name,
                "pass": False,
                "latency": latency_s,
                "notes": [f"JSON Parse Error: {e}"],
                "metrics": {},
            })

    # Summary Table
    passed_count = sum(1 for r in results if r["pass"])
    total_count = len(results)
    avg_latency = (
        sum(r["latency"] for r in results) / total_count if total_count else 0.0
    )
    accuracy_pct = (passed_count / total_count) * 100.0 if total_count else 0.0

    print("\n" + "═" * 80)
    print("📊 [BENCHMARK RESULTS SUMMARY - SINGLE DRONE TARGET NAVIGATION]")
    print("═" * 80)
    print(
        f"{'ID':<7} | {'Scenario Name':<38} | {'Status':<8} | {'Latency':<8} | {'Target Err':<10}"
    )
    print("-" * 80)

    for r in results:
        status_sym = "✅ PASS" if r["pass"] else "❌ FAIL"
        lat_text = f"{r['latency']:.2f}s"
        t_err = r["metrics"].get("target_error_euclidean_m")
        err_text = f"{t_err:.2f}m" if t_err is not None else "N/A"
        print(
            f"{r['id']:<7} | {r['name']:<38} | {status_sym:<8} | {lat_text:<8} | {err_text:<10}"
        )

    print("-" * 80)
    print(
        f"🏆 Overall Accuracy: {passed_count}/{total_count} ({accuracy_pct:.1f}%) | "
        f"Mean Latency: {avg_latency:.2f}s ({avg_latency*1000:.0f} ms)"
    )
    print("═" * 80 + "\n")

    return {
        "backend": backend,
        "total": total_count,
        "passed": passed_count,
        "accuracy_pct": accuracy_pct,
        "avg_latency_s": avg_latency,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous Target Navigation VLM Benchmark Evaluator"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "gemini", "ollama"],
        help="Evaluation backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (e.g. gemini-2.5-flash or qwen2.5vl:3b)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="vlm/test_cases_single.json",
        help="Path to single drone test cases JSON",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for Ollama service",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose rule outputs and details",
    )
    args = parser.parse_args()

    run_benchmark(
        backend=args.backend,
        model=args.model,
        test_file=Path(args.test_file),
        ollama_url=args.ollama_url,
        verbose=args.verbose,
    )
