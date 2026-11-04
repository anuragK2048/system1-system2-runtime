# Multimodal Vision-Language Model (VLM) Flight Reasoning & Navigation System

This directory houses the **Multimodal Vision-Language Reasoning Engine (System 2)** for autonomous drone operations. It couples high-frequency reflex control (500 Hz System 1) with high-level cognitive scene understanding, spatial navigation, and physical anomaly rejection using vision models (Google Gemini 3.5 Flash-Lite / Gemini 3.8 Live, local Qwen2.5-VL via Ollama / Apple Silicon MLX, or dedicated serverless Modal L4 GPUs).

---

## Directory Layout

```
vlm/
├── hud_overlay.py               # Avionics HUD engine (ground rings, pitch ladder, 3D apertures)
├── evaluate.py                  # Unified benchmark CLI (--suite single|canonical|new)
├── evaluate_target_navigation.py# Evaluator for single-drone target navigation (SD-01..14)
├── evaluate_model.py            # Evaluator for dual-drone disturbance suites (TC-01..12)
├── gemini_live_client.py        # Bidirectional streaming WebSocket client for Gemini Live
├── demo_hud_projection.py       # Demonstrator producing neutral, pitched, banked HUD frames
├── demo_outputs/                # Rendered visual proof frames with HUD overlays
├── grab_sample_frame.py         # Captures calibrated 480x360 FPV frame from MuJoCo scene
├── sample_fpv.jpg               # Standard test frame with burned avionics HUD
├── reference_views/             # Canonical arena FPV references (runway, window, arch)
├── scratch/                     # Transient verification frames and test outputs
│
├── test_cases_single.json       # Active Single-Drone benchmark suite (SD-01..14)
├── test_cases_single/           # Rendered FPV frames for single-drone targets
├── test_cases.json              # Canonical dual-drone disturbance suite (TC-01..06)
├── test_cases/                  # FPV frames for TC-01..06
├── test_cases_new.json          # Out-of-sample generalization suite (TC-07..12)
├── test_cases_new/              # FPV frames for TC-07..12
│
├── generate_single_drone_suite.py # Generates SD-01..14 test dataset & images
├── generate_test_suite.py       # Generates TC-01..06 test dataset & images
├── generate_new_test_suite.py   # Generates TC-07..12 test dataset & images
│
├── test_gemini_vlm.py           # Quick cloud test runner for Gemini Flash-Lite
├── test_gemini_live.py          # Quick WebSocket test runner for Gemini Live
├── test_qwen_ollama.py          # Local test runner for Ollama Qwen2.5-VL
├── test_qwen_mlx.py             # Local test runner for Apple Silicon MLX-VLM
├── test_cloud_vlm.py            # Test runner for Fireworks / OpenAI vision models
├── modal_qwen_vlm.py            # Serverless Nvidia L4 GPU deployment on Modal
├── test_modal_client.py         # Client for deployed Modal service
├── TEST_CASES.md                # Comprehensive specification of all test suites
└── README.md                    # This technical reference
```

---

## 1. Unified Benchmark Evaluation CLI

The unified test runner evaluates any model across all three benchmark suites:

```bash
# Verify Ground Truth on Active Single-Drone Suite (SD-01..14)
uv run python vlm/evaluate.py --suite single --backend ground_truth

# Evaluate Google Gemini on Single-Drone Suite
uv run python vlm/evaluate.py --suite single --backend gemini

# Evaluate Gemini Live over WebSockets
uv run python vlm/evaluate.py --suite single --backend gemini-live

# Evaluate Local Ollama Qwen2.5-VL
uv run python vlm/evaluate.py --suite single --backend ollama --model qwen2.5vl:3b

# Evaluate Canonical Dual-Drone Disturbance Suite (TC-01..06)
uv run python vlm/evaluate.py --suite canonical --backend ground_truth

# Evaluate Out-of-Sample Generalization Suite (TC-07..12)
uv run python vlm/evaluate.py --suite new --backend ground_truth
```

---

## 2. Dedicated Single-Drone Navigation Dashboard (Port 8080)

To launch the real-time simulation cockpit with live FPV video streaming, interactive target navigation (Pad Alpha, Pedestal Delta, Window, Arch), chaos injection, and asynchronous Gemini VLM reasoning:

```bash
uv run python scripts/run_navigation_dashboard.py
```

- Open in browser: `http://localhost:8080`
- **Port Isolation**: Runs on port 8080 to avoid conflicts with the legacy port 8000 dual-drone server.
- **MJPEG FPV Stream**: Smooth 15–20 FPS video with burned HUD ground rings and artificial horizon (`/api/navigation/stream`).
- **Telemetry WebSocket**: 40 Hz real-time flight telemetry, motor thrusts, and event logs (`/ws/navigation`).
- **REST Control**: Trigger targets (`POST /api/navigation/target`), inject wind/motor faults (`POST /api/navigation/chaos`), toggle autonomous VLM navigation (`POST /api/navigation/vlm/toggle`).

---

## 3. Calibrated HUD Projection Engine (`vlm/hud_overlay.py`)

Raw camera pixels are prone to scale ambiguity for Vision-Language Models. The **`HUDProjector`** burns perspective-accurate avionics cues directly onto the FPV image before feeding it to the VLM:

1. **Concentric Metric Ground Rings (1m, 2m, 3m, 4m)**:
   - Ground plane projections centered beneath the drone nadir.
   - Near-plane line clipping prevents perspective inversion behind the camera lens.
2. **Artificial Horizon & Pitch Ladder**:
   - Analytically derived from camera rotation matrix (`cam_xmat`) with $\pm 10^\circ, \pm 20^\circ$ pitch rungs.
3. **3D Portals, Apertures & Gates (Windows, Arches, Hoops)**:
   - Dynamic 3D bounding wireframes with inner rotor-clearance envelopes, bullseyes, and glidepath alignment cues.
4. **Flight Telemetry Ribbon**:
   - In-image altitude, airspeed, pitch, roll, and heading indicators.

To render sample demonstration frames:
```bash
uv run python vlm/demo_hud_projection.py
```
Output images are stored in `vlm/demo_outputs/`.

---

## 4. Local VLM Inference Backends

### Option A: Local Ollama (Zero Setup)
```bash
# Pull model
ollama pull qwen2.5vl:3b

# Run inference test
uv run python vlm/test_qwen_ollama.py
```

### Option B: Apple Silicon Native MLX-VLM (Lowest Latency: 180–300 ms)
```bash
uv pip install mlx-vlm
uv run python vlm/test_qwen_mlx.py
```

### Option C: Google Gemini Live Streaming API
```bash
# Ensure GEMINI_API_KEY is present in your .env
uv run python vlm/test_gemini_live.py
```

### Option D: Dedicated Serverless Nvidia L4 GPU on Modal
```bash
uv run modal setup
uv run modal run vlm/modal_qwen_vlm.py
uv run modal deploy vlm/modal_qwen_vlm.py
uv run python vlm/test_modal_client.py --url <YOUR_MODAL_URL>
```
