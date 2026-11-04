#!/usr/bin/env python3
"""Launcher for the Single-Drone Target Navigation & Performance Dashboard.

Starts the dedicated navigation server on port 8080 (or custom --port)
and opens the dashboard in the browser or terminal.

Usage:
------
    uv run python scripts/run_navigation_dashboard.py
    uv run python scripts/run_navigation_dashboard.py --port 8080 --autostart alpha
"""

import os
import sys
from pathlib import Path

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# macOS MuJoCo Native Viewer Trampoline:
# On macOS, GLFW/Cocoa requires running under mjpython to service UI events.
# If --gui is requested on macOS and not yet running under mjpython, trampoline automatically!
if "--gui" in sys.argv and sys.platform == "darwin":
    try:
        import mujoco.viewer
        if not isinstance(getattr(mujoco.viewer, "_MJPYTHON", None), getattr(mujoco.viewer, "_MjPythonBase", object)):
            exe_dir = Path(sys.executable).parent
            mjpy = exe_dir / "mjpython"
            if mjpy.exists():
                os.execv(str(mjpy), [str(mjpy)] + sys.argv)
    except Exception:
        pass

from runtime.server.navigation_server import main

if __name__ == "__main__":
    main()
