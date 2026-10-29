"""Skydio X2 Dual-Drone Interactive Comparative Testbed Entry Point.

Launch comparative simulation of 2 drones running different System 1 and System 2 setups
subjected to identical chaos in real-time MuJoCo physics.

Usage:
------
# Recommended Default (PID Baseline vs. L1 Adaptive + System 2 LLM):
uv run mjpython src/runtime/sim/view_dual_drone.py

# Using Quick Presets:
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 1   # PID vs L1 Adaptive (No LLM)
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 2   # L1 Adaptive vs L1 Adaptive + LLM
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 3   # PID vs L1 Adaptive + LLM
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 4   # PID vs PID + LLM

# Fully Custom Drones:
uv run mjpython src/runtime/sim/view_dual_drone.py --drone1 pid --drone2 l1_adaptive+system2

# Interactive Mode:
uv run mjpython src/runtime/sim/view_dual_drone.py --interactive
"""

import sys
import argparse
from typing import Tuple
from runtime.sim.dual_simulator import DualDroneSimulator
from runtime.sim.config import get_system_naming


PRESETS = {
    "1": ("pid", "l1_adaptive", "Baseline PID vs. L1 Adaptive (Reflex Comparison)"),
    "pid_vs_l1": ("pid", "l1_adaptive", "Baseline PID vs. L1 Adaptive (Reflex Comparison)"),

    "2": ("l1_adaptive", "l1_adaptive+system2", "L1 Adaptive vs. L1 Adaptive + System 2 LLM (Patch Impact)"),
    "unpatched_vs_patched": ("l1_adaptive", "l1_adaptive+system2", "L1 Adaptive vs. L1 Adaptive + System 2 LLM (Patch Impact)"),

    "3": ("pid", "l1_adaptive+system2", "Baseline PID vs. L1 Adaptive + System 2 LLM (Full Stack)"),
    "pid_vs_l1_s2": ("pid", "l1_adaptive+system2", "Baseline PID vs. L1 Adaptive + System 2 LLM (Full Stack)"),

    "4": ("pid", "pid+system2", "Baseline PID vs. PID + System 2 LLM (PID Self-Healing)"),
    "pid_vs_pid_s2": ("pid", "pid+system2", "Baseline PID vs. PID + System 2 LLM (PID Self-Healing)"),
}

MODE_ALIASES = {
    "1": "pid",
    "pid": "pid",
    "2": "l1_adaptive",
    "l1": "l1_adaptive",
    "l1_adaptive": "l1_adaptive",
    "3": "pid+system2",
    "pid+s2": "pid+system2",
    "pid+system2": "pid+system2",
    "4": "l1_adaptive+system2",
    "l1+s2": "l1_adaptive+system2",
    "l1_adaptive+system2": "l1_adaptive+system2",
}


def interactive_selection() -> Tuple[str, str]:
    """Provides an interactive terminal menu for selecting drone modes."""
    print("\n" + "=" * 74)
    print("🚁 SKYDIO X2 DUAL-DRONE SIDE-BY-SIDE CONFIGURATION")
    print("=" * 74)
    print("Quick Presets:")
    print("  [1] Baseline PID vs. L1 Adaptive (Reflex Comparison)")
    print("  [2] L1 Adaptive (Unpatched) vs. L1 Adaptive + System 2 LLM")
    print("  [3] Baseline PID vs. L1 Adaptive + System 2 LLM (Default)")
    print("  [4] Baseline PID vs. PID + System 2 LLM")
    print("  [5] Custom Configuration (Choose for each drone independently)")
    print("-" * 74)

    choice = input("Select Preset [1-5] (default: 3): ").strip()
    if choice in ("1", "2", "3", "4"):
        d1, d2, _ = PRESETS[choice]
        return d1, d2

    if choice == "5":
        print("\nAvailable Controller Modes:")
        print("  [1] System-1 (Classical PID)")
        print("  [2] System-1 (L1 Adaptive - QuadLib)")
        print("  [3] System-1 (Classical PID) + System-2 (LLM Meta-Controller)")
        print("  [4] System-1 (L1 Adaptive) + System-2 (LLM Meta-Controller)")

        c2 = input("Choose Left Drone  (Cyan / D2)   [1-4] (default: 4): ").strip() or "4"
        c1 = input("Choose Right Drone (Orange / D1) [1-4] (default: 1): ").strip() or "1"

        d1 = MODE_ALIASES.get(c1, "pid")
        d2 = MODE_ALIASES.get(c2, "l1_adaptive+system2")
        return d1, d2

    # Default fallback
    return "pid", "l1_adaptive+system2"


def main() -> None:
    """Parses command line arguments and launches the dual drone simulator."""
    parser = argparse.ArgumentParser(
        description="Skydio X2 Dual-Drone Side-by-Side Comparative Simulation in MuJoCo"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Quick Preset: 1 (PID vs L1), 2 (L1 vs L1+S2), 3 (PID vs L1+S2), 4 (PID vs PID+S2)",
    )
    parser.add_argument(
        "--drone1",
        type=str,
        default=None,
        help="Drone 1 (Right / Orange): pid, l1_adaptive, pid+system2, l1_adaptive+system2",
    )
    parser.add_argument(
        "--drone2",
        type=str,
        default=None,
        help="Drone 2 (Left / Cyan): pid, l1_adaptive, pid+system2, l1_adaptive+system2",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["groq", "baseten", "fireworks", "openai"],
        help="Inference provider for System 2 LLM (default: groq or LLM_PROVIDER env)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for System 2 LLM (e.g. openai/gpt-oss-120b, meta-models/muse-spark-open-1026)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal selector prompt at startup",
    )
    parser.add_argument(
        "--serve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start FastAPI + WebSocket live control server in background (default: True)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for live control server (default: 8000)",
    )
    parser.add_argument(
        "--s2-config",
        type=str,
        default=None,
        help="Path to JSON configuration file for System 2 (e.g. configs/system2_default.json)",
    )

    args = parser.parse_args()

    # Determine drone modes
    if args.interactive:
        d1_mode, d2_mode = interactive_selection()
    elif args.preset is not None:
        key = str(args.preset).lower()
        if key in PRESETS:
            d1_mode, d2_mode, label = PRESETS[key]
            print(f"\n📋 Loaded Preset: {label}")
        else:
            print(f"⚠️ Unknown preset '{args.preset}'. Using default PID vs L1 Adaptive + System 2.")
            d1_mode, d2_mode = "pid", "l1_adaptive+system2"
    elif args.drone1 or args.drone2:
        d1_mode = MODE_ALIASES.get(args.drone1.lower() if args.drone1 else "pid", "pid")
        d2_mode = MODE_ALIASES.get(
            args.drone2.lower() if args.drone2 else "l1_adaptive+system2", "l1_adaptive+system2"
        )
    else:
        # Default run configuration: Drone 1 = Baseline L1 Adaptive, Drone 2 = L1 Adaptive + System 2 LLM
        d1_mode = "l1_adaptive"
        d2_mode = "l1_adaptive+system2"

    s2_cfg = None
    if args.s2_config:
        from runtime.config import System2MasterConfig
        s2_cfg = System2MasterConfig.load_json(args.s2_config)
        print(f"⚙️ Loaded Custom System 2 Config: {args.s2_config}")

    provider_label = args.provider or (s2_cfg.inference.provider if s2_cfg and s2_cfg.inference.provider else "groq (default)")
    d1_sys = get_system_naming(d1_mode)
    d2_sys = get_system_naming(d2_mode)
    print(f"\n🚀 Launching Dual Drone Simulator:")
    print(f"   🔵 Left Drone  (Cyan / D2):  {d2_sys['system_name']}")
    print(f"   🟠 Right Drone (Orange / D1): {d1_sys['system_name']}")
    print(f"🧠 System 2 LLM Provider: [{provider_label}]")
    sim = DualDroneSimulator(
        drone1_mode=d1_mode,
        drone2_mode=d2_mode,
        llm_provider=args.provider,
        llm_model=args.model,
        s2_cfg=s2_cfg,
    )

    # Launch background FastAPI & WebSocket control server if requested
    if args.serve:
        import threading
        import uvicorn
        from runtime.server.app import app

        def _run_server():
            config = uvicorn.Config(app=app, host="0.0.0.0", port=args.port, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        server_thread = threading.Thread(target=_run_server, name="WebServerThread", daemon=True)
        server_thread.start()
        print(f"🌐 [WEB DASHBOARD] Live Control Server started at: http://localhost:{args.port}")
        print(f"   • API Documentation: http://localhost:{args.port}/docs")
        print(f"   • Telemetry Stream:  ws://localhost:{args.port}/ws/telemetry\n")

    sim.run()


if __name__ == "__main__":
    main()
