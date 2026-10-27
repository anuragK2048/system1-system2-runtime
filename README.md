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

---

## Quickstart

### Run the 3D Single-Drone Simulation
```bash
uv run mjpython src/runtime/sim/view_drone.py
```

### Run the 3D Dual-Drone Side-by-Side Comparison
```bash
# Recommended default (PID Baseline vs. L1 Adaptive + System 2 LLM):
uv run mjpython src/runtime/sim/view_dual_drone.py

# Using Quick Presets:
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 1   # PID vs. L1 Adaptive (Reflex Comparison)
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 2   # L1 Adaptive vs. L1 Adaptive + System 2 LLM
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 3   # PID vs. L1 Adaptive + System 2 LLM
uv run mjpython src/runtime/sim/view_dual_drone.py --preset 4   # PID vs. PID + System 2 LLM

# Fully Custom Comparison:
uv run mjpython src/runtime/sim/view_dual_drone.py --drone1 pid --drone2 l1_adaptive+system2

# Interactive Selector Menu:
uv run mjpython src/runtime/sim/view_dual_drone.py --interactive
```

### Run Controller, Bridge & Dual-Drone Tests
```bash
uv run python -m unittest discover tests
```

### Interactive Chaos Hotkeys (Inside 3D Viewer)
- `[Shift + 1]` (or `6`): Cut Motor 1 power by 60% (Chipped propeller / ESC fault)
- `[Shift + 2]` (or `7`): Attach swinging pendulum cargo (0.35kg tethered payload)
- `[Shift + 3]` (or `8`): Inject high-frequency IMU sensor vibration noise
- `[Shift + 4]` (or `9`): Blow sudden 5.0 N lateral wind gust
- `[Shift + 5]` (or `0`): Reset drone to level hover