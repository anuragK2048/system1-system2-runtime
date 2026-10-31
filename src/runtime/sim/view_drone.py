"""Skydio X2 Interactive Multibody Chaos Testbed Entry Point.

Runs the 500 Hz System 1 reflex simulation and coordinates with the
System 2 LLM Meta-Controller in the background.

Launch the simulation with:
    uv run mjpython src/runtime/sim/view_drone.py
"""

import sys
import argparse
from typing import Optional
from runtime.sim.simulator import DroneSimulator
from runtime.brain.llm import System2Worker


def run_simulation(
    enable_system2: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Instantiates and executes the Skydio X2 Drone Simulator with System 2 Meta-Controller."""
    sim = DroneSimulator()
    worker = None
    if enable_system2:
        try:
            worker = System2Worker(
                bridge=sim.bridge,
                provider=provider,
                model=model,
            )
            worker.start()
        except Exception as e:
            print(f"⚠️ [SYSTEM 2] Could not start LLM Meta-Controller ({e}). Running System 1 only.")

    try:
        sim.run()
    finally:
        if worker:
            worker.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skydio X2 Interactive Simulation with System 2 LLM Meta-Controller"
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
        help="Model name override for System 2 LLM",
    )
    parser.add_argument(
        "--no-system2",
        action="store_true",
        help="Disable System 2 LLM meta-controller (run System 1 only)",
    )
    args = parser.parse_args()

    run_simulation(
        enable_system2=not args.no_system2,
        provider=args.provider,
        model=args.model,
    )


if __name__ == "__main__":
    main()
