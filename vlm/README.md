# Local VLM Testing for Skydio X2 (Qwen2.5-VL)

This directory provides standalone scripts to test **local multimodal vision-language models** (specifically **Qwen2.5-VL-3B**) using actual onboard FPV camera frames rendered from your MuJoCo simulation.

---

## Directory Structure

```
vlm/
├── hud_overlay.py          # High-performance calibrated HUD projection helper (ground rings, horizon, 3D apertures)
├── demo_hud_projection.py  # Visual demonstrator producing neutral, pitched, and banked FPV frames
├── demo_outputs/           # Rendered demonstration frames with burned HUD overlays
├── grab_sample_frame.py    # Renders calibrated 480x360 FPV frame from assets/skydio_x2/dual_scene.xml
├── sample_fpv.jpg          # Generated test frame with metric rings, horizon, and flight telemetry
├── evaluate_model.py       # Automated benchmark evaluator across test suites
├── generate_test_suite.py  # Canonical 6 test cases (TC-01..06) generator
├── generate_new_test_suite.py # Out-of-sample 6 test cases (TC-07..12) generator
├── test_gemini_vlm.py      # Cloud test runner for Google Gemini Flash Lite
├── test_qwen_ollama.py     # Tests Qwen2.5-VL via local Ollama OpenAI-compatible API
├── test_qwen_mlx.py        # Tests Qwen2.5-VL via Apple Silicon native MLX-VLM (sub-300ms)
├── test_cloud_vlm.py       # Tests Fireworks / OpenAI vision models
├── modal_qwen_vlm.py       # Dedicated serverless L4 GPU app for Modal
├── test_modal_client.py    # Client for querying deployed Modal endpoint
└── README.md               # This guide
```

---

## Option 1: Test via Ollama (Zero Setup — Recommended First Step)

Ollama is already installed and running on your Mac. It provides an OpenAI-compatible HTTP endpoint at `http://localhost:11434/v1`.

### Step 1: Pull the Qwen2.5-VL Model
Open a terminal and download the lightweight 3B model (or 7B):
```bash
# Recommended 3B model (fastest inference, ~2.5 GB footprint)
ollama pull qwen2.5vl:3b

# Or standard 7B model (~5.5 GB footprint)
ollama pull qwen2.5vl
```

### Step 2: Render a Fresh FPV Frame from MuJoCo
```bash
uv run python vlm/grab_sample_frame.py
```
*This generates `vlm/sample_fpv.jpg` (480×360 px with boresight reticle and altitude HUD) directly from `assets/skydio_x2/dual_scene.xml`.*

### Step 3: Run the Inference Test
```bash
uv run python vlm/test_qwen_ollama.py
```

### Optional Flags:
```bash
# Stream the tokens in real time to measure Time-To-First-Token (TTFT):
uv run python vlm/test_qwen_ollama.py --stream

# Custom flight instruction:
uv run python vlm/test_qwen_ollama.py --instruction "Identify the cyan landing pad, approach it and descend to 0.5m"

# Test 7B model:
uv run python vlm/test_qwen_ollama.py --model qwen2.5vl
```

---

## Option 2: Test via Apple Silicon Native MLX-VLM (Lowest Latency: 180–300 ms)

For maximum speed directly on Apple Silicon Metal GPU unified memory:

### Step 1: Install MLX-VLM
```bash
uv pip install mlx-vlm
```

### Step 2: Run the MLX Test
```bash
uv run python vlm/test_qwen_mlx.py
```
*The script automatically downloads `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` from Hugging Face on first run and benchmarks raw Metal GPU execution time.*

---

## Calibrated HUD Projection & Spatial Grounding (`vlm/hud_overlay.py`)

Raw camera pixels are prone to depth ambiguity and scale confusion for Vision-Language Models. The **`HUDProjector`** burns perspective-accurate metric ground rings, an artificial horizon, and 3D passage aperture frames directly onto the FPV image before feeding it to the VLM:

### Key Capabilities:
1. **Metric Ground Rings (1m, 2m, 3m, 4m)**:
   - Projects concentric circles on the ground plane ($Z=0$) centered at the drone's nadir projection.
   - Handled with near-plane line clipping ($Z_c \le 0.05$m) to prevent perspective inversion.
   - Distance labels (`"1m"`, `"2m"`, `"3m"`) burned along the flightpath corridor.
2. **Artificial Horizon & Pitch Ladder**:
   - Calculated analytically from camera rotation matrix (`cam_xmat`):
     $$\frac{\text{right}_z}{f_x}(u - c_x) - \frac{\text{up}_z}{f_y}(v - c_y) + \text{forward}_z = 0$$
   - Displays boresight aircraft reticle and pitch ladder ticks at $\pm 10^\circ, \pm 20^\circ$.
3. **3D Portals, Apertures & Gates (Doors, Windows, Arches, Hoops)**:
   - Extensible `Aperture3D` data class defining arbitrary 3D openings.
   - Automatically projects outer frame, inner rotor-clearance green envelope, central bullseye reticle, and 3D approach glidepath vector.
   - Auto-discovers obstacles from MuJoCo bodies/sites matching `gate_*`, `door_*`, `window_*`, `arch_*`.
4. **Flight Telemetry Ribbon**:
   - Displays real-time altitude, airspeed, pitch, roll, and heading.

### Run the Visual Demonstrator:
```bash
uv run python vlm/demo_hud_projection.py
```
Outputs are saved to `vlm/demo_outputs/`:
- `demo_hud_neutral.jpg`: Neutral hover facing an Arch Gate at $X=2.5$m.
- `demo_hud_pitch_down.jpg`: Pitched forward flight ($-12^\circ$ pitch) facing an elevated Window Aperture.
- `demo_hud_banked.jpg`: Banked turn ($+18^\circ$ roll) facing a Doorway with clearance margin.

### Performance:
- Pure vector projection + PIL rendering executes in **~3.5 ms (280+ FPS)**, seamlessly fitting within the 1–5 Hz System 2 loop without impacting the 500 Hz reflex controller.

---

## Expected Output

When running either test script, you will see the benchmark banner and the unified JSON response:

```text
==========================================================================
🚁 [TESTING LOCAL VLM] Model: qwen2.5vl:3b @ http://localhost:11434/v1
📸 Image Input: vlm/sample_fpv.jpg (480x360 FPV frame)
🎯 Instruction: 'Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.'
==========================================================================
⚡ [LATENCY RESULTS] Total Latency: 0.28s (280 ms)
⏱️ Time-To-First-Token (TTFT): 0.142s (142 ms)
==========================================================================
✅ [JSON PARSE SUCCESS] Verified Structured Payload:
{
  "diagnosis": "Corridor clear ahead. Target pad detected at center-ground. Minor leftward drift.",
  "waypoint_command": {
    "mode": "relative_body",
    "target": [1.2, 0.0, 0.0],
    "yaw_deg": 0.0,
    "max_velocity": 1.0,
    "corridor_width": 0.6
  },
  "feedforward_force_offset": [0.0, 0.0, 0.0],
  "gain_multipliers": {
    "pos_kp": 1.0,
    "pos_kd": 1.0,
    "att_kp": 1.0,
    "att_kd": 1.0
  },
  "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
  "safety_envelope": {
    "max_tilt_deg": 20.0
  }
}
--------------------------------------------------------------------------
  • Waypoint Command: ✅ PRESENT
  • Feedforward Force: ✅ PRESENT
  • Motor Trim Scale: ✅ PRESENT
  • Gain Multipliers: ✅ PRESENT
==========================================================================
```

This confirms:
1. The local model successfully parses the FPV image.
2. It outputs both the high-level **`waypoint_command`** for navigation AND the low-level **physical levers** (`feedforward_force_offset`, `motor_trim_scale`, `gain_multipliers`) for recovery.
3. Latency stays within the required **200–500 ms** budget for closed-loop System 2 control.


---

## Option 3: Deploy Dedicated Serverless GPU on Modal ($100 Hackathon Credits)

> [!IMPORTANT]
> **Zero Shared Endpoints:** This deployment uses a **dedicated Nvidia L4 GPU** (`gpu="L4"`, 24GB VRAM) billed strictly per GPU-second against your free $100 Meta Hackathon credits. It does NOT use Modal's pay-per-token managed endpoints.
> - Automatically scales down to 0 replicas after 5 minutes idle (costs **$0.00** when not flying).
> - Typical remote inference latency: **~180–280 ms**.

### Step 1: Authenticate with Modal
```bash
uv run modal setup
```
*(Follow the browser prompt to authorize your account)*

### Step 2: Run a One-Off Remote Test on the L4 GPU
```bash
uv run modal run vlm/modal_qwen_vlm.py
```

### Step 3: Deploy Persistent Web Endpoint
```bash
uv run modal deploy vlm/modal_qwen_vlm.py
```
*Modal will print your public HTTPS endpoint URL:*
`https://<your-username>--drone-qwen-vlm-dronevlmservice-predict.modal.run`

### Step 4: Query the Deployed Endpoint
```bash
uv run python vlm/test_modal_client.py --url https://<your-endpoint-url>
```
