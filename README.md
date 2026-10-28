# system1-system2-runtime

> **System-1 / System-2 Autonomous Flight Runtime:** Bridging the latency gap between 500ms LLM reasoning and 500Hz hardware control.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │  System 2 (Async LLM Meta-Controller, 1-5 Hz) │
                      │  - Diagnoses failures and environmental chaos │
                      │  - Outputs compensatory patches:              │
                      │    * feedforward_force_offset                 │
                      │    * gain_multipliers                         │
                      │    * motor_trim_scale                         │
                      │    * safety_envelope                          │
                      └──────────────────────▲────────────────────────┘
                                             │ [Async Non-blocking Patches]
                      ┌──────────────────────▼────────────────────────┐
                      │        System1System2Bridge (Thread-Safe)     │
                      │  - Fast Triggers (Pos Err, Tilt, Gyro, Force) │
                      │  - 1.0s Sliding Window Telemetry Summaries    │
                      └──────────────────────▲────────────────────────┘
                                             │ [500 Hz Telemetry & Patches]
                      ┌──────────────────────▼────────────────────────┐
                      │    System 1 Reflex Controller (500 Hz)        │
                      │  - SE(3) Geometric Non-linear Tracking (u_b)  │
                      │  - L1 Adaptive Disturbance Observer (u_ad)    │
                      │  - Disturbance Estimates (sigma_f, sigma_M)   │
                      │  - Allocation Mixer (B_INV -> u1, u2, u3, u4) │
                      └──────────────────────▲────────────────────────┘
                                             │
                      ┌──────────────────────▼────────────────────────┐
                      │     MuJoCo Physics Simulation (Skydio X2)     │
                      │  - 500 Hz C++ Multibody Physics               │
                      │  - Multibody Chaos: Motor cuts, Wind, Payload │
                      │  - 60 FPS OpenGL Tracking Visualizer          │
                      └───────────────────────────────────────────────┘
```

---

## Features

### 1. Robust System 1 Controller (`mini-quadlib`)
- **SE(3) Geometric Controller:** Manifold-based position and attitude tracking.
- **L1 Adaptive Control Augmentation:** Real-time disturbance observer estimating:
  - $\hat{\sigma}_f$: Unmatched horizontal force & matched collective thrust disturbances.
  - $\hat{\sigma}_M$: Matched pitch, roll, and yaw torque disturbances.
- **High Performance:** Sub-microsecond C implementation (`libmini_quadlib.dylib` / `.so`) with seamless Python bindings.

### 2. Precise Real-Time System 1 <-> System 2 Coordination Bridge
- **Non-Blocking 500 Hz Telemetry:** Circular buffer with sliding window aggregation.
- **Fast Trigger Detector:** Instantly signals System 2 on:
  - Position error exceeding threshold ($> 0.30$ m)
  - Excessive tilt ($> 25^\circ$) or gyro rate spikes ($> 3.5$ rad/s)
  - Physics residual / disturbance force spike ($> 3.0$ N)
- **Semantic LLM Summarizer:** Translates raw 500 Hz state data into human- and LLM-readable summaries (Euler angles, gyro RMS/jitter, actuator effort/saturation, and estimated disturbance forces).
- **Dynamic Control Patches:** System 2 injects real-time adjustments without stopping or delaying the 500 Hz reflex loop.

### 3. Dual-Drone Comparative Architecture & Spatial Orientation
In the 3D OpenGL MuJoCo window and the Web Dashboard, the two drones are spatially aligned as follows:
- **🔵 Left Drone (Cyan / Drone 2, $Y = +1.2\text{ m}$)**
- **🟠 Right Drone (Orange / Drone 1, $Y = -1.2\text{ m}$)**

#### Supported Canonical Operating Systems
Each drone can run any of the four canonical configurations:
1. **`System-1 (Classical PID)`**: Baseline PID position and attitude reflex control.
2. **`System-1 (L1 Adaptive - QuadLib)`**: Fast piecewise-constant L1 adaptive disturbance observer canceling unmodeled force and moment disturbances at 500 Hz.
3. **`System-1 (Classical PID) + System-2 (LLM Meta-Controller)`**: Baseline PID controller augmented with high-level asynchronous LLM reasoning.
4. **`System-1 (L1 Adaptive) + System-2 (LLM Meta-Controller)`**: Full dual-tier hybrid architecture combining 500 Hz L1 adaptive reflexes with 1–5 Hz LLM strategic compensation.

---

## Repository Structure

```
system1-system2-runtime/
├── assets/skydio_x2/          # Skydio X2 drone MJCF models and dual_scene.xml
├── frontend/                  # Modern web dashboard for live monitoring & chaos injection
│   └── index.html             # Glassmorphic UI with real-time SSE telemetry & control sliders
├── logs/                      # Audit trails for LLM meta-controller reasoning transactions
│   ├── llm_calls.jsonl        # Machine-readable JSONL execution log
│   └── llm_calls.log          # Human-readable formatted log
├── src/runtime/               # Core Python / C codebase
│   ├── brain/                 # System-2 async LLM reasoning engine & prompt definitions
│   ├── chaos/                 # Real-time chaos injection (motor cuts, wind, payload, noise)
│   │   ├── manager.py         # Thread-safe ChaosManager for Drone 1, Drone 2, or Synchronized
│   │   ├── models.py          # Typed dataclasses for chaos configurations
│   │   ├── physics.py         # MuJoCo physics application of forces, moments, and noise
│   │   └── presets.py         # Repeatable multi-select benchmark presets
│   ├── logging/               # Transaction logging for latency, token count, and control patches
│   ├── quadlib/               # Compiled C library (`libmini_quadlib`) with ctypes bridge
│   ├── schemas/               # Pydantic models for telemetry and System-2 patch packets
│   ├── server/                # FastAPI application exposing REST and SSE streaming endpoints
│   │   ├── app.py             # Server entrypoint and static mount for dashboard
│   │   ├── routes_chaos.py    # REST routes for custom sliders, multi-presets, and target selection
│   │   ├── routes_telemetry.py# Telemetry status and snapshot APIs
│   │   ├── state.py           # Shared simulation context and bridge state
│   │   └── stream.py          # High-frequency Server-Sent Events (SSE) telemetry generator
│   └── sim/                   # 500 Hz MuJoCo simulation environment & controllers
│       ├── bridge.py          # System 1 <-> System 2 thread-safe coordination bridge
│       ├── config.py          # Drone parameters, spatial orientations, and system naming
│       ├── dual_simulator.py  # Synchronized dual-drone simulator with wall-clock pacing & HUD
│       ├── system1_controller.py # 500 Hz SE(3) Geometric and L1 Adaptive reflex loops
│       └── view_dual_drone.py # Interactive launcher with 3D window and automatic web server
└── tests/                     # Comprehensive unit and integration test suite (43+ tests)
```

---

## Quickstart

### Run the 3D Single-Drone Simulation
```bash
uv run mjpython src/runtime/sim/view_drone.py
```

### Run the 3D Dual-Drone Side-by-Side Comparison (With Live Web Control)
```bash
# Recommended default (PID Baseline vs. L1 Adaptive + System 2 LLM):
# Automatically launches the 3D MuJoCo window AND the Web Dashboard on port 8000!
uv run mjpython src/runtime/sim/view_dual_drone.py

# Open Web Dashboard in browser: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Run Standalone Headless Server
```bash
# Runs 500 Hz MuJoCo simulation headlessly with Web Dashboard:
uv run python -m runtime.server.app --headless --port 8000
```

### Run Controller, Bridge, Chaos & Server Tests
```bash
uv run python -m unittest discover tests
```

### Real-Time Chaos Controls (Dual Options: Web Sliders or In-Viewer Hotkeys)
1. **Web Dashboard (`http://localhost:8000`)**:
   - Independent sliders for Motors 1–4 ($0\%\text{--}100\%$)
   - 3D Wind Vector ($0\text{--}15\text{N}$, compass direction, Dryden turbulence)
   - Tethered Cargo coupling (toggle, mass, aerodynamic wind drag on bob)
   - IMU sensor chatter noise slider
   - Target selector (`Synchronize Both`, `🔵 Left (Cyan / D2)`, `🟠 Right (Orange / D1)`)
   - Repeatable multi-select benchmark presets
   - Live telemetry, thrust meters, L1 disturbance observer comparison, and LLM reasoning stream
2. **Interactive Hotkeys (Inside 3D MuJoCo Viewer)**:
   - `[Shift + 1]` (or `6`): Preset 1 - Cut Motor 1 power by 60% (Chipped propeller flutter)
   - `[Shift + 2]` (or `7`): Preset 2 - Attach swinging pendulum cargo (0.35kg bob + wind drag)
   - `[Shift + 3]` (or `8`): Preset 3 - Inject high-frequency IMU sensor vibration noise
   - `[Shift + 4]` (or `9`): Preset 4 - Blow 6.5 N lateral crosswind + turbulence
   - `[Shift + 5]` (or `0`): Preset 0 - Reset drones to clean level hover