"""Skydio X2 Interactive Multibody Chaos Testbed Entry Point.

Runs the 500 Hz System 1 reflex simulation and coordinates with the
System 2 LLM Meta-Controller in the background.

Launch the simulation with:
    uv run mjpython src/runtime/sim/view_drone.py
"""

import sys
from runtime.sim.simulator import DroneSimulator
from runtime.brain.llm import System2Worker


def run_simulation(enable_system2: bool = True) -> None:
    """Instantiates and executes the Skydio X2 Drone Simulator with System 2 Meta-Controller."""
    sim = DroneSimulator()
    worker = None
    if enable_system2:
        try:
            worker = System2Worker(bridge=sim.bridge)
            worker.start()
        except Exception as e:
            print(f"⚠️ [SYSTEM 2] Could not start LLM Meta-Controller ({e}). Running System 1 only.")

    try:
        sim.run()
    finally:
        if worker:
            worker.stop()


if __name__ == "__main__":
    run_simulation()
