"""Gemini Live Models Benchmark & Latency Measurement CLI.

Connects to Gemini Live API over WebSockets to evaluate real-time multimodal
flight reasoning across canonical, single-drone, and out-of-sample flight test cases.

Supported Live Models:
  1. gemini-3.8-live (Default, low-latency native audio/multimodal)
  2. gemini-3.1-flash-live-preview (Legacy preview)
  3. gemini-3.8-live-extended-thinking (Deep background reasoning)

Metrics Measured:
  - WebSocket Handshake Latency (connection establishment)
  - Real-Time Inference Latency (multimodal processing & tool call decision)
  - Total End-to-End Latency
  - Rule Validation Accuracy (Target setpoint, disturbance rejection, motor trims, safety envelope)
  - Target Setpoint Euclidean Error (m)
  - Disturbance Force Rejection Error (N)

Usage:
    # 1. Test gemini-3.8-live on canonical test suite (TC-01..06):
    uv run python vlm/test_gemini_live.py --model gemini-3.8-live --suite canonical

    # 2. Test gemini-3.1-flash-live-preview on canonical suite:
    uv run python vlm/test_gemini_live.py --model gemini-3.1-flash-live-preview --suite canonical

    # 3. Test on single-drone target navigation suite (SD-01..07):
    uv run python vlm/test_gemini_live.py --model gemini-3.8-live --suite single

    # 4. Compare all available models side-by-side:
    uv run python vlm/test_gemini_live.py --compare
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from google import genai
from google.genai import types

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from vlm.gemini_live_client import (
    LIVE_SYSTEM_PROMPT,
    LIVE_TOOL_DECLARATION,
    convert_tool_args_to_flight_patch,
)
from vlm.evaluate_model import evaluate_rules as evaluate_rules_canonical
from vlm.evaluate_target_navigation import (
    evaluate_rules as evaluate_rules_single,
    compute_metrics,
)

SUITE_PATHS = {
    "canonical": Path("vlm/test_cases.json"),
    "single": Path("vlm/test_cases_single.json"),
    "new": Path("vlm/test_cases_new.json"),
}

SYSTEM_PROMPT = LIVE_SYSTEM_PROMPT


async def evaluate_case_live(
    client: genai.Client,
    model_name: str,
    tc: Dict[str, Any],
    suite_type: str = "canonical",
) -> Dict[str, Any]:
    """Evaluates a single test case using Gemini Live API WebSocket session."""
    img_path = Path(tc["image_path"])
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    instruction = tc["instruction"]
    telemetry_str = json.dumps(tc["telemetry"], indent=2)
    prompt_text = (
        f"MISSION INSTRUCTION: {instruction}\n\n"
        f"TELEMETRY STATE:\n{telemetry_str}\n\n"
        "Analyze the attached onboard FPV camera view and call emit_flight_patch immediately."
    )

    tools = [dict(LIVE_TOOL_DECLARATION)]
    kwargs: Dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "tools": tools,
        "system_instruction": types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    }

    if "extended-thinking" in model_name:
        tools[0]["function_declarations"][0]["behavior"] = "NON_BLOCKING"
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    config = types.LiveConnectConfig(**kwargs)

    t_connect_start = time.time()
    t_handshake = 0.0
    t_inference = 0.0
    args = None

    try:
        async with client.aio.live.connect(model=model_name, config=config) as session:
            t_handshake = time.time() - t_connect_start

            t_send = time.time()
            parts = [
                types.Part(inline_data=types.Blob(data=img_bytes, mime_type="image/jpeg")),
                types.Part(text=prompt_text),
            ]
            await session.send_client_content(
                turns=types.Content(role="user", parts=parts),
                turn_complete=True,
            )

            async for response in session.receive():
                now = time.time()
                if response.tool_call:
                    t_inference = now - t_send
                    args = response.tool_call.function_calls[0].args
                    break
                if response.server_content and response.server_content.turn_complete:
                    t_inference = now - t_send
                    break
    except Exception as e:
        return {
            "id": tc["id"],
            "name": tc["name"],
            "pass": False,
            "error": str(e),
            "handshake_ms": t_handshake * 1000.0,
            "inference_ms": t_inference * 1000.0,
            "total_ms": (t_handshake + t_inference) * 1000.0,
            "metrics": {},
            "reasons": [f"Connection error: {e}"],
        }

    t_total = t_handshake + t_inference
    if args is None:
        return {
            "id": tc["id"],
            "name": tc["name"],
            "pass": False,
            "error": "No tool call received",
            "handshake_ms": t_handshake * 1000.0,
            "inference_ms": t_inference * 1000.0,
            "total_ms": t_total * 1000.0,
            "metrics": {},
            "reasons": ["Model completed turn without executing emit_flight_patch"],
        }

    patch = convert_tool_args_to_flight_patch(args)
    if suite_type == "single":
        is_valid, reasons = evaluate_rules_single(patch, tc["validation_rules"])
    else:
        is_valid, reasons = evaluate_rules_canonical(patch, tc["validation_rules"])

    metrics = compute_metrics(patch, tc["ground_truth"])

    return {
        "id": tc["id"],
        "name": tc["name"],
        "pass": is_valid,
        "error": None,
        "handshake_ms": t_handshake * 1000.0,
        "inference_ms": t_inference * 1000.0,
        "total_ms": t_total * 1000.0,
        "metrics": metrics,
        "reasons": reasons,
        "patch": patch,
    }


def run_live_benchmark(
    model_name: str = "gemini-3.8-live",
    suite: str = "canonical",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Runs full benchmark suite on the specified Gemini Live model."""
    import asyncio

    test_file = SUITE_PATHS[suite]
    with open(test_file) as f:
        tcs = json.load(f)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please ensure GEMINI_API_KEY is set in your .env file or environment."
        )
    client = genai.Client(api_key=api_key)

    print("\n" + "═" * 84)
    print("🚁 [GEMINI LIVE MODELS BENCHMARK & LATENCY MEASUREMENT]")
    print(f"📡 Model:       {model_name}")
    print(f"📁 Test Suite:  {suite.upper()} ({test_file}) - {len(tcs)} test cases")
    print(f"🔑 Auth:        Gemini API Key configured")
    print("═" * 84)
    print(
        f"{'ID':<7} | {'Scenario Name':<34} | {'Status':<8} | {'Inference':<10} | {'Handshake':<10} | {'Total':<9} | {'Target Err':<10}"
    )
    print("-" * 84)

    results = []

    for tc in tcs:
        res = asyncio.run(evaluate_case_live(client, model_name, tc, suite_type=suite))
        status = "✅ PASS" if res["pass"] else "❌ FAIL"
        inf_str = f"{res['inference_ms']:6.1f} ms"
        hs_str = f"{res['handshake_ms']:6.1f} ms"
        tot_str = f"{res['total_ms']:6.1f} ms"
        t_err = res["metrics"].get("target_error_euclidean_m")
        err_str = f"{t_err:.2f} m" if t_err is not None else "N/A"

        print(
            f"{res['id']:<7} | {res['name']:<34} | {status:<8} | {inf_str:<10} | {hs_str:<10} | {tot_str:<9} | {err_str:<10}"
        )
        if verbose and not res["pass"]:
            for r in res["reasons"]:
                print(f"   ⚠️  {r}")

        results.append(res)

    total_count = len(results)
    passed_count = sum(1 for r in results if r["pass"])
    acc_pct = (passed_count / total_count) * 100.0 if total_count else 0.0

    avg_inf = sum(r["inference_ms"] for r in results) / total_count
    avg_hs = sum(r["handshake_ms"] for r in results) / total_count
    avg_tot = sum(r["total_ms"] for r in results) / total_count

    valid_errs = [
        r["metrics"]["target_error_euclidean_m"]
        for r in results
        if r["metrics"].get("target_error_euclidean_m") is not None
    ]
    avg_err = sum(valid_errs) / len(valid_errs) if valid_errs else 0.0

    print("-" * 84)
    print(f"🏆 [PERFORMANCE & LATENCY SUMMARY FOR {model_name}]")
    print(f"   Success Rate:             {passed_count}/{total_count} ({acc_pct:.1f}%)")
    print(f"   Mean Inference Latency:   {avg_inf:.1f} ms ({avg_inf/1000.0:.3f} s)")
    print(f"   Mean Handshake Latency:   {avg_hs:.1f} ms ({avg_hs/1000.0:.3f} s)")
    print(f"   Mean Total Turnaround:    {avg_tot:.1f} ms ({avg_tot/1000.0:.3f} s)")
    if valid_errs:
        print(f"   Mean Target Setpoint Err: {avg_err:.3f} m")
    print("═" * 84 + "\n")

    return {
        "model": model_name,
        "suite": suite,
        "total": total_count,
        "passed": passed_count,
        "accuracy_pct": acc_pct,
        "mean_inference_ms": avg_inf,
        "mean_handshake_ms": avg_hs,
        "mean_total_ms": avg_tot,
        "mean_target_err_m": avg_err,
        "results": results,
    }


def run_comparison(suite: str = "canonical"):
    """Runs side-by-side comparison between Gemini Live models and standard Gemini REST baseline."""
    models_to_test = [
        ("gemini-3.8-live", "live"),
        ("gemini-3.1-flash-live-preview", "live"),
    ]

    summaries = []
    for model_name, mtype in models_to_test:
        summary = run_live_benchmark(model_name=model_name, suite=suite)
        summaries.append(summary)

    print("\n" + "═" * 88)
    print(f"📊 [MULTIMODAL VLM MODEL COMPARISON MATRIX — SUITE: {suite.upper()}]")
    print("═" * 88)
    print(
        f"{'Model Name':<30} | {'Type':<12} | {'Accuracy':<10} | {'Inference Latency':<18} | {'Total Latency':<14}"
    )
    print("-" * 88)
    for s in summaries:
        acc_str = f"{s['passed']}/{s['total']} ({s['accuracy_pct']:.0f}%)"
        inf_str = f"{s['mean_inference_ms']:.1f} ms"
        tot_str = f"{s['mean_total_ms']:.1f} ms"
        print(
            f"{s['model']:<30} | {'WebSocket':<12} | {acc_str:<10} | {inf_str:<18} | {tot_str:<14}"
        )
    print("═" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Live Models Benchmark & Latency Measurement CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-live",
        choices=[
            "gemini-3.8-live",
            "gemini-3.1-flash-live-preview",
            "gemini-3.8-live-extended-thinking",
        ],
        help="Gemini Live model identifier to benchmark",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="canonical",
        choices=["canonical", "single", "new"],
        help="Flight test suite to run: 'canonical' (TC-01..06), 'single' (SD-01..07), or 'new' (TC-07..12)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comprehensive comparative benchmark across Gemini Live models",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display failure diagnostics and validation discrepancies",
    )

    args = parser.parse_args()

    if args.compare:
        run_comparison(suite=args.suite)
    else:
        run_live_benchmark(model_name=args.model, suite=args.suite, verbose=args.verbose)


if __name__ == "__main__":
    main()
