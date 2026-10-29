# ⚙️ System-2 (LLM Meta-Controller) Configuration Reference

> **Comprehensive parameter encyclopedia, file mapping, and tuning manual for System-1 and System-2 fault-tolerance control.**

---

## Table of Contents

1. [Overview & Central Configuration Hub](#1-overview--central-configuration-hub)
2. [File Locations & Architectural Map](#2-file-locations--architectural-map)
3. [Fast Trigger Parameters (`FastTriggerConfig`)](#3-fast-trigger-parameters-fasttriggerconfig)
4. [Telemetry Summary Parameters (`TelemetrySummaryConfig`)](#4-telemetry-summary-parameters-telemetrysummaryconfig)
5. [Hot-Swap & Patch Application Parameters (`PatchApplicationConfig`)](#5-hot-swap--patch-application-parameters-patchapplicationconfig)
6. [LLM Inference & Generation Parameters (`System2InferenceConfig`)](#6-llm-inference--generation-parameters-system2inferenceconfig)
7. [System-2 Worker Lifecycle Parameters (`System2WorkerConfig`)](#7-system-2-worker-lifecycle-parameters-system2workerconfig)
8. [Master Aggregator (`System2MasterConfig`)](#8-master-aggregator-system2masterconfig)
9. [Related System-1 Physical & Controller Configs](#9-related-system-1-physical--controller-configs)
10. [How to Tweak and Run (CLI & Python Examples)](#10-how-to-tweak-and-run-cli--python-examples)
11. [Automated Fine-Tuning & Grid-Search Recipes](#11-automated-fine-tuning--grid-search-recipes)
12. [Observability, Logs & Telemetry Inspection](#12-observability-logs--telemetry-inspection)
13. [Common Configuration Pitfalls & Pro-Tips](#13-common-configuration-pitfalls--pro-tips)

---

## 1. Overview & Central Configuration Hub

In this hybrid autonomous drone architecture:
* **System-1 (500 Hz Reflex Controller)** stabilizes the Skydio X2 quadrotor against high-frequency disturbances using SE(3) geometric tracking and L1 adaptive disturbance rejection.
* **System-2 (Asynchronous LLM Meta-Controller)** acts as the higher-level cognitive reasoner, monitoring 1.0-second sliding windows of physical telemetry, identifying root causes (e.g. motor degradation vs. wind microburst vs. swinging cargo), and injecting compensatory control patches (feedforward force offsets, gain multipliers, motor trims, and safety envelopes).

To achieve maximum fault-tolerance accuracy and benchmark outperformance, **all System-2 and coordination parameters can now be configured from a single place**:
- **On disk via JSON**: [`configs/system2_default.json`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/configs/system2_default.json)
- **In code via Python dataclasses**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py) (importable directly from `runtime.config`)

---

## 2. File Locations & Architectural Map

```
system1-system2-runtime/
│
├── configs/
│   ├── system2_default.json            # 📄 Central JSON config containing all baseline defaults
│   └── system2_tuned_template.json     # 📄 Pre-configured template for fine-tuning & benchmarking
│
├── src/runtime/
│   ├── config/                         # 📦 Centralized Configuration Hub
│   │   ├── __init__.py                 # Re-exports all S1 + S2 + physics configs
│   │   └── system2_config.py           # 🧠 Master dataclasses (Trigger, Telemetry, Patch, LLM, Worker)
│   │
│   ├── brain/
│   │   ├── llm.py                      # System2Brain & System2Worker (wired to configs)
│   │   └── __init__.py                 # Exports System2Brain, System2Worker, and S2 configs
│   │
│   ├── sim/
│   │   ├── bridge.py                   # System1System2Bridge (evaluates triggers & window metrics)
│   │   ├── system1_controller.py       # System1Controller (resolves patches with ramp-in & clamps)
│   │   ├── config.py                   # DroneConfig, PIDConfig, L1Config, SimConfig
│   │   ├── dual_simulator.py           # DualDroneSimulator (accepts s2_cfg for side-by-side sim)
│   │   ├── simulator.py                # DroneSimulator (accepts s2_cfg for single-drone sim)
│   │   ├── view_dual_drone.py          # 3D interactive viewer (supports --s2-config CLI flag)
│   │   └── demo_llm_impact.py          # Automated evaluation runner (supports --config CLI flag)
│   │
│   └── logging/
│       └── llm_logger.py               # Records transactions to logs/llm_calls.jsonl
│
└── tests/
    └── test_system2_config.py          # Unit tests verifying serialization, ramp-in, clamps, and TTL
```

---

## 3. Fast Trigger Parameters (`FastTriggerConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L29-L59)  
> **Wired Into**: [`src/runtime/sim/bridge.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/bridge.py) → `push_telemetry()`

The Fast Trigger evaluates ground-truth state and L1 adaptive innovation every 2 milliseconds (500 Hz). When any parameter exceeds its threshold, a `FastTriggerEvent` is produced and System-2 is alerted.

| Parameter | Type | Default Value | Physical Units | Description & Effect |
|---|---|---|---|---|
| `pos_error_threshold` | `float` | `0.30` | meters (m) | **3D Position Error Limit**: $\|p_{\text{target}} - p_{\text{actual}}\|$. Triggers when horizontal or vertical drift exceeds 30 cm. Lowering to `0.15–0.20m` wakes up System-2 earlier before large trajectory deviations develop. |
| `tilt_deg_threshold` | `float` | `25.0` | degrees (°) | **Attitude Inclination Limit**: Total angle between the drone's vertical $Z$-axis and global gravity vector. Catches dangerous tilts before the drone flips. |
| `gyro_norm_threshold` | `float` | `3.5` | rad/s | **Angular Velocity Norm Limit**: $\|\vec{\omega}\|_2$. Detects high-rate tumbling, propeller strikes, or aggressive yaw/pitch rotations. |
| `force_residual_threshold` | `float` | `3.0` | Newtons (N) | **L1 Force Disturbance Innovation**: $\|\hat{\sigma}_f\|_2$. Measures unmodeled forces estimated by the L1 observer. This detects wind gusts or sudden payloads *before* position error even grows. |
| `trigger_cooldown_s` | `float` | `0.5` | seconds (s) | **Minimum Trigger Interval**: Minimum wall-clock time between consecutive triggers stored in the bridge queue, preventing buffer flooding during sustained disturbances. |
| `min_sim_time` | `float` | `1.0` | seconds (s) | **Warmup Skip Duration**: Physics time ignored after drone spawn to prevent false alarms during initial takeoff ascent. |
| `max_buffer_size` | `int` | `1500` | samples | **Ring Buffer Depth**: 1500 samples at 500 Hz holds exactly 3.0 seconds of high-fidelity state history. |

---

## 4. Telemetry Summary Parameters (`TelemetrySummaryConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L65-L98)  
> **Wired Into**: [`src/runtime/sim/bridge.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/bridge.py) → `get_telemetry_summary()`

Translates raw 500 Hz sensor arrays into physical summaries that the LLM can reason over.

| Parameter | Type | Default Value | Physical Units | Description & Effect |
|---|---|---|---|---|
| `window_seconds` | `float` | `1.0` | seconds (s) | **Sliding Window Duration**: Length of recent physics history used to calculate mean position error, velocity, and actuator effort. Shorter windows (0.5s) capture fast gusts; longer windows (1.5s) capture slow pendulum swings. |
| `recent_instant_samples` | `int` | `25` | samples | **Instantaneous Force Window**: Number of most recent frames used to compute immediate disturbance forces (25 samples at 500 Hz = 50 ms). Contrasted against the 1.0s mean in prompts to highlight rapid changes. |
| `motor_saturation_min` | `float` | `0.05` | Newtons (N) | **Actuator Lower Bound**: Motor commands below this threshold are counted as 0% thrust saturation (idle/stopped rotor). |
| `motor_saturation_max` | `float` | `12.95` | Newtons (N) | **Actuator Upper Bound**: Motor commands above this threshold are counted as 100% thrust saturation (rotor producing maximum 13.0N physical capacity). |
| `oscillation_min_freq_hz` | `float` | `3.0` | Hertz (Hz) | **Chatter Frequency Threshold**: Minimum pitch/roll angular rate sign-flip frequency required to classify drone dynamics as oscillating rather than smoothly drifting. |
| `oscillation_min_gyro_rms` | `float` | `2.0` | rad/s | **Oscillation Amplitude Threshold**: Minimum gyro RMS level required alongside frequency to trigger the `oscillation_detected` diagnostic flag. |
| `include_instantaneous_force` | `bool` | `True` | boolean | **Prompt Dynamic Force Tag**: When enabled, outputs both current instantaneous force and 1.0s mean force in the user prompt. |
| `include_active_patch` | `bool` | `False` | boolean | **Prompt Active Patch Context**: When enabled, informs the LLM of what patch is *currently active* on System-1, enabling the LLM to refine rather than restart corrections from scratch. |

---

## 5. Hot-Swap & Patch Application Parameters (`PatchApplicationConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L104-L147)  
> **Wired Into**: [`src/runtime/sim/system1_controller.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/system1_controller.py) → `_resolve_active_patch()`

Dictates how and when System-2 control adjustments hit the 500 Hz reflex loop.

| Parameter | Type | Default Value | Physical Units | Description & Effect |
|---|---|---|---|---|
| `ramp_duration_s` | `float` | `0.0` | seconds (s) | **Hot-Swap Apply / Ramp Time**: <br>• `0.0` = Instantaneous step patch injection (current baseline).<br>• `> 0.0` (e.g. `0.3`) = Smooth linear fade-in from 0% to 100% over the specified duration. Eliminates actuator shock and prevents transient overshoot. |
| `patch_ttl_s` | `Optional[float]` | `None` | seconds (s) | **Patch Time-To-Live (Expiration)**: If set to a number (e.g. `8.0`), active patches automatically expire back to nominal System-1 baseline if no update arrives, preventing stale offsets from sticking. |
| `confidence_scaling` | `bool` | `False` | boolean | **Graduated Control Action**: When enabled, scales feedforward forces and motor trims by the LLM's returned `confidence` score ($0.0 \dots 1.0$). |
| `feedforward_clamp_min` | `float` | `-10.0` | Newtons (N) | **Feedforward Force Lower Bound**: Maximum negative force offset allowed per axis ($F_x, F_y, F_z$). |
| `feedforward_clamp_max` | `float` | `10.0` | Newtons (N) | **Feedforward Force Upper Bound**: Maximum positive force offset allowed per axis. Protects against LLM numerical hallucinations. |
| `motor_trim_clamp_min` | `float` | `0.0` | multiplier | **Motor Trim Minimum**: Prevents the LLM from commanding negative thrust multipliers. |
| `motor_trim_clamp_max` | `float` | `2.0` | multiplier | **Motor Trim Maximum**: Restricts motor scaling to a safe $2.0\times$ cap to avoid rotor over-speed burnout. |
| `gain_multiplier_clamp_min` | `float` | `0.1` | multiplier | **Gain Multiplier Floor**: Prevents LLM gain retuning from zeroing out System-1 attitude or position loops. |
| `gain_multiplier_clamp_max` | `float` | `3.0` | multiplier | **Gain Multiplier Ceiling**: Prevents LLM gain retuning from causing explosive closed-loop instability. |
| `max_tilt_deg_clamp_min` | `float` | `5.0` | degrees (°) | **Tilt Envelope Minimum**: Ensures the drone always retains at least $5^\circ$ of maneuver authority. |
| `max_tilt_deg_clamp_max` | `float` | `45.0` | degrees (°) | **Tilt Envelope Maximum**: Hard ceiling preventing commanded recovery angles that would induce tumbling. |

---

## 6. LLM Inference & Generation Parameters (`System2InferenceConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L153-L188)  
> **Wired Into**: [`src/runtime/brain/llm.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/brain/llm.py) → `System2Brain`

Controls inference connectivity, model selection, reasoning budget, and reproducibility.

| Parameter | Type | Default Value | Description & Effect |
|---|---|---|---|
| `provider` | `Optional[str]` | `None` | **Inference Provider**: `'groq'`, `'baseten'`, `'fireworks'`, or `'openai'`. (Defaults to environment variable `LLM_PROVIDER` or auto-detection). |
| `model` | `Optional[str]` | `None` | **Model Identifier**: Overrides model string (e.g. `'qwen/qwen3.8-27b'`, `'meta-models/muse-spark-open-1026'`). |
| `temperature` | `float` | `0.2` | **Sampling Temperature**: Controls randomness. Setting to `0.05` or `0.0` produces deterministic, repeatable outputs essential for rigorous benchmarking. |
| `groq_max_completion_tokens` | `int` | `2048` | **Groq Token Limit**: Maximum output tokens allowed on Groq Cloud endpoints. Can be lowered to `512` to speed up generation. |
| `general_max_tokens` | `int` | `4096` | **General Token Limit**: Maximum token budget for OpenAI/Baseten/Fireworks endpoints. |
| `timeout_s` | `float` | `10.0` | **HTTP Client Timeout**: Maximum duration to wait before aborting an in-flight LLM reasoning query. |
| `seed` | `Optional[int]` | `None` | **Deterministic Seed**: Fixed integer seed for endpoints supporting deterministic sampling. |
| `system_prompt` | `Optional[str]` | `None` | **System Prompt Override**: Custom prompt string. If `None`, uses the canonical `SYSTEM_PROMPT`. |
| `enable_json_mode` | `bool` | `False` | **Raw JSON Enforcement**: Enforces `response_format={"type": "json_object"}` where supported by the endpoint. |

---

## 7. System-2 Worker Lifecycle Parameters (`System2WorkerConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L194-L255)  
> **Wired Into**: [`src/runtime/brain/llm.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/brain/llm.py) → `System2Worker._run_loop()`

Governs the asynchronous monitoring loop running in the background daemon thread.

| Parameter | Type | Default Value | Physical Units | Description & Effect |
|---|---|---|---|---|
| `cooldown_s` | `float` | `2.0` | seconds (s) | **Inter-Call Cooldown**: Minimum elapsed time before System-2 is allowed to dispatch another LLM query. Lowering to `1.0s` enables faster second-shot adjustments. |
| `loop_poll_interval_s` | `float` | `0.05` | seconds (s) | **Worker Loop Sleep**: Sleep duration per iteration of the worker thread. |
| `warmup_min_timestamp` | `float` | `1.0` | seconds (s) | **Warmup Time Gate**: Ignores trigger events occurring before this simulation timestamp. |
| `warmup_min_samples` | `int` | `200` | samples | **Warmup Sample Gate**: Requires at least 200 telemetry frames (0.4s) before computing summaries. |
| `attention_pos_error` | `float` | `0.30` | meters (m) | **Case 1 Attention Threshold (Position)**: Position error required to trigger LLM when no patch is active. |
| `attention_force_disturbance` | `float` | `2.5` | Newtons (N) | **Case 1 Attention Threshold (Force)**: Unmodeled disturbance force required to trigger LLM when no patch is active. |
| `attention_moment_disturbance` | `float` | `0.35` | N·m | **Case 1 Attention Threshold (Moment)**: Unmodeled disturbance torque required to trigger LLM when no patch is active. |
| `attention_oscillation` | `bool` | `True` | boolean | **Case 1 Attention Flag (Oscillation)**: Triggers LLM immediately if attitude chatter is detected. |
| `calm_pos_error` | `float` | `0.15` | meters (m) | **Case 2 Calm Hover (Position)**: Maximum error allowed to consider flight restored to nominal hover. |
| `calm_force_disturbance` | `float` | `1.0` | Newtons (N) | **Case 2 Calm Hover (Force)**: Maximum disturbance force allowed to consider perturbation cleared. |
| `calm_moment_disturbance` | `float` | `0.20` | N·m | **Case 2 Calm Hover (Moment)**: Maximum disturbance moment allowed to consider perturbation cleared. |
| `calm_min_settle_time_s` | `float` | `2.5` | seconds (s) | **Case 2 Settle Duration**: Minimum time a calm hover must be held before removing the active patch and returning System-1 to baseline. |
| `reconsult_pos_error` | `float` | `0.75` | meters (m) | **Case 2 Revision Threshold (Position)**: Position error that triggers re-consulting the LLM because the active patch is failing. |
| `reconsult_oscillation_age_s` | `float` | `4.0` | seconds (s) | **Case 2 Revision Age (Oscillation)**: Time that persistent oscillation must last after patch application before re-consulting the LLM. |
| `reconsult_min_patch_age_s` | `float` | `6.0` | seconds (s) | **Case 2 Revision Age (Divergence)**: Minimum time an active patch must be evaluated before discarding it and commanding a revision. |
| `stable_pos_error` | `float` | `0.30` | meters (m) | **Case 2 Stability Gate**: Position error below which an active patch is marked as successfully holding flight. |
| `crash_alt_min` | `float` | `0.20` | meters (m) | **Crash Guard (Ground Contact)**: Altitude below which all reasoning calls halt to save tokens. |
| `crash_tilt_near_ground_alt` | `float` | `0.40` | meters (m) | **Crash Guard (Tumble Altitude)**: Low altitude threshold checked against steep tilt. |
| `crash_tilt_near_ground_deg` | `float` | `55.0` | degrees (°) | **Crash Guard (Tumble Angle)**: Angle threshold near ground that halts reasoning calls. |
| `crash_tilt_extreme_deg` | `float` | `80.0` | degrees (°) | **Crash Guard (Inversion)**: Tilt angle (>80°) indicating an inverted tumble that halts reasoning calls. |
| `crash_runaway_pos_error` | `float` | `4.5` | meters (m) | **Crash Guard (Runaway)**: Error distance (>4.5m) indicating irrecoverable departure. |

---

## 8. Master Aggregator (`System2MasterConfig`)

> **Location**: [`src/runtime/config/system2_config.py`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/config/system2_config.py#L261-L320)

Encapsulates all five sub-configs into a single object:

```python
from runtime.config import System2MasterConfig

cfg = System2MasterConfig()
# Access any parameter:
cfg.trigger.pos_error_threshold = 0.18
cfg.telemetry.window_seconds = 1.0
cfg.patch.ramp_duration_s = 0.3
cfg.inference.temperature = 0.05
cfg.worker.cooldown_s = 1.0

# Export / Import:
cfg.save_json("configs/my_experiment.json")
loaded = System2MasterConfig.load_json("configs/my_experiment.json")
```

---

## 9. Related System-1 Physical & Controller Configs

All System-1 and physical parameters are also accessible via `runtime.config`:

| Class | File | Key Parameters |
|---|---|---|
| [`L1Config`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/config.py#L110-L144) | `src/runtime/sim/config.py` | `pos_gains`, `vel_gains`, `att_gains`, `ang_gains`, Hurwitz adaptation gains `as_v` (`-5.0`), `as_w` (`-20.0`), LPF cutoff frequencies `lpf1_cutoff_freq_force` (`10.0 Hz`), `lpf2_cutoff_freq_moment` (`5.0 Hz`). |
| [`PIDConfig`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/config.py#L72-L104) | `src/runtime/sim/config.py` | Classical PID gains (`kp_z`, `kd_z`, `kp_att`, `kd_att`, `kp_xy`, `kd_xy`, `max_tilt_rad`). |
| [`DroneConfig`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/config.py#L28-L67) | `src/runtime/sim/config.py` | Skydio X2 mass (1.325 kg), hover weight (13.0 N), rotor limits [0, 13] N, allocation mixer `b_inv`. |
| [`SimConfig`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/src/runtime/sim/config.py#L155-L166) | `src/runtime/sim/config.py` | Physics timestep (`0.002s` = 500 Hz), render FPS (`60.0`), auto-tracking camera angles. |

---

## 10. How to Tweak and Run (CLI & Python Examples)

### Running 3D Interactive Simulation with a Custom Config
```bash
# Launch side-by-side comparative simulation using your JSON config:
uv run mjpython src/runtime/sim/view_dual_drone.py --s2-config configs/system2_default.json

# Test with the pre-tuned template:
uv run mjpython src/runtime/sim/view_dual_drone.py --s2-config configs/system2_tuned_template.json
```

### Running Automated Quantitative Evaluation
```bash
# Evaluate impact on wind gust using custom parameters:
uv run python src/runtime/sim/demo_llm_impact.py wind_gust --config configs/system2_default.json

# Evaluate swinging payload:
uv run python src/runtime/sim/demo_llm_impact.py swinging_payload --config configs/system2_tuned_template.json
```

---

## 11. Automated Fine-Tuning & Grid-Search Recipes

For automated hyperparameter sweeps or Bayesian optimization:

```python
import numpy as np
from runtime.config import System2MasterConfig
from runtime.sim.demo_llm_impact import evaluate_scenario

# Define parameter grid
ramp_durations = [0.0, 0.15, 0.30, 0.50]
trigger_force_thresholds = [1.5, 2.0, 2.5, 3.0]
cooldown_times = [0.8, 1.2, 2.0]

best_score = -float("inf")
best_params = None

for ramp in ramp_durations:
    for force_th in trigger_force_thresholds:
        for cd in cooldown_times:
            # 1. Instantiate and configure
            cfg = System2MasterConfig()
            cfg.patch.ramp_duration_s = ramp
            cfg.trigger.force_residual_threshold = force_th
            cfg.worker.cooldown_s = cd
            cfg.inference.temperature = 0.05

            # 2. Evaluate scenario
            res = evaluate_scenario(scenario_name="wind_gust", headless=True, s2_cfg=cfg)
            improvement = res.get("pos_improvement_pct", 0.0)

            print(f"Ramp={ramp}s | ForceThresh={force_th}N | Cooldown={cd}s -> Improvement: {improvement:.1f}%")

            if improvement > best_score:
                best_score = improvement
                best_params = cfg
                cfg.save_json("configs/best_tuned_system2.json")

print(f"\n🏆 Best Result: {best_score:.1f}% improvement saved to configs/best_tuned_system2.json")
```

---

## 12. Observability, Logs & Telemetry Inspection

All System-2 reasoning transactions are logged to disk:

* **Machine-Readable JSONL**: [`logs/llm_calls.jsonl`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/logs/llm_calls.jsonl)  
  Records full timestamps, durations, prompt tokens, completion tokens, reasoning traces, and parsed JSON patches.
* **Human-Readable Ledger**: [`logs/llm_calls.log`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/logs/llm_calls.log)  
  Formatted ledger entries showing the exact prompt sent and output received.
* **Provider Token Usage Summary**: [`logs/provider_token_usage.json`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/logs/provider_token_usage.json)  
  Partitioned token and call totals for Groq, Baseten, Fireworks, and OpenAI.

---

## 13. Common Configuration Pitfalls & Pro-Tips

1. **Ramp Duration (`ramp_duration_s`)**:
   * Setting `ramp_duration_s = 0.0` applies full force offsets instantly. In severe crosswinds, this can cause a transient jolt.
   * Setting `ramp_duration_s = 0.2` to `0.4` seconds produces smooth physical engagement that consistently improves RMS tracking error.
2. **Fast Trigger vs. Worker Cooldown**:
   * If you lower `trigger_cooldown_s` (e.g. `0.2s`), make sure `worker.cooldown_s` is calibrated appropriately (e.g. `1.0s`), otherwise the worker will throttle repeated calls regardless of bridge triggers.
3. **Safety Clamping Bounds**:
   * Never set `feedforward_clamp_max` higher than total drone weight ($\approx 13.0\text{ N}$). An excessive feedforward command can invert or flip the drone.
4. **Reproducible Evals**:
   * For scientific benchmarking, always set `inference.temperature = 0.0` or `0.05` and set `inference.seed = 42` to eliminate non-deterministic LLM variance.

