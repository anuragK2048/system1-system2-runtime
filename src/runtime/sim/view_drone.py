"""Skydio X2 Interactive Multibody Chaos Testbed Entry Point.

Launch the simulation with:
    uv run mjpython src/runtime/sim/view_drone.py
"""

from runtime.sim.simulator import DroneSimulator


def run_simulation() -> None:
    """Instantiates and executes the Skydio X2 Drone Simulator."""
    sim = DroneSimulator()
    sim.run()


if __name__ == "__main__":
    run_simulation()
