# Autonomous Flight VLM Benchmark: Test Suite & Ground Truth Reference

This benchmark evaluates any Multimodal Vision-Language Model (VLM) acting as the **System-2 Cognitive Reasoner** for the Skydio X2 quadrotor.

It tests whether the model can simultaneously:
1. **Understand FPV visual scenes** (corridors, landing pads, obstacle proximity).
2. **Translate mission instructions into 3D navigation waypoints** (`waypoint_command`).
3. **Reject physical chaos & actuator degradation** (`feedforward_force_offset`, `motor_trim_scale`, `gain_multipliers`, `safety_envelope`).

---

## The 6 Canonical Test Cases

### 1. TC-01: Corridor Forward Advance (Visual Navigation)
* **Objective:** Test nominal forward navigation through an unobstructed corridor.
* **FPV Image:** [`vlm/test_cases/tc_01_corridor_advance.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_01_corridor_advance.jpg) (clear runway extending into distance, altitude 1.0m, zero obstacles).
* **Telemetry Input:**
  * Altitude: $1.00\text{ m}$
  * Velocity: $[+0.02, -0.01, 0.00]\text{ m/s}$ (calm hover)
  * Attitude: $\text{Roll}=0.0^\circ, \text{Pitch}=0.0^\circ, \text{Yaw}=0.0^\circ$
  * Disturbance Force: $[0.0, 0.0, 0.0]\text{ N}$
* **Mission Instruction:** *"Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "Flight corridor clear ahead. Maintaining centered nominal advance.",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [1.5, 0.0, 0.0],
      "yaw_deg": 0.0,
      "max_velocity": 1.0,
      "corridor_width": 0.6
    },
    "feedforward_force_offset": [0.0, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
    "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 20.0}
  }
  ```
* **Validation Rules:**
  * $dx \in [0.8, 3.0]\text{ m}$ (forward progress).
  * $|dy| \le 0.3\text{ m}$ (centered in corridor).
  * Feedforward force must be $\approx [0, 0, 0]\text{ N}$ (no phantom force during calm flight).

---

### 2. TC-02: Target Pad Approach & Centering (Visual Servoing)
* **Objective:** Test landing pad visual acquisition and simultaneous 3D descent/approach.
* **FPV Image:** [`vlm/test_cases/tc_02_target_pad_approach.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_02_target_pad_approach.jpg) (orange landing pad visible ahead at front-right bearing).
* **Telemetry Input:**
  * Altitude: $0.85\text{ m}$
  * Velocity: $[+0.15, +0.05, -0.05]\text{ m/s}$
  * Attitude: $\text{Roll}=0.2^\circ, \text{Pitch}=-1.5^\circ, \text{Yaw}=-115.0^\circ$
* **Mission Instruction:** *"Align with the orange target pad, center it in your crosshairs, and descend to 0.4m."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "Target pad detected in front-right sector. Aligning heading and commanding descent to 0.4m.",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [0.8, 0.0, -0.45],
      "yaw_deg": 0.0,
      "max_velocity": 0.8,
      "corridor_width": 0.6
    },
    "feedforward_force_offset": [0.0, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
    "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 20.0}
  }
  ```
* **Validation Rules:**
  * $dz \in [-0.6, -0.2]\text{ m}$ (must command negative vertical displacement toward 0.4m target).
  * $dx \in [0.2, 2.0]\text{ m}$ (advances toward pad).

---

### 3. TC-03: Headwind Gust Compensation (Disturbance Rejection)
* **Objective:** Test physical chaos compensation where external force pushes drone backward.
* **FPV Image:** [`vlm/test_cases/tc_03_headwind_compensation.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_03_headwind_compensation.jpg) (pitched nose-down into wind, ground drifting).
* **Telemetry Input:**
  * $L_1$ Disturbance Force: $\hat{\sigma}_f = [+5.2\text{ N}, 0.0\text{ N}, 0.0\text{ N}]$ (strong wind along $+X$)
  * Velocity: $v_x = -0.35\text{ m/s}$ (drifting downwind)
  * Pitch: $-14.0^\circ$
* **Mission Instruction:** *"Hold station over current waypoint against headwind disturbance."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "Strong headwind disturbance force (+5.2N along +X) detected. Tilting into wind with opposing feedforward offset.",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "max_velocity": 0.8,
      "corridor_width": 0.6
    },
    "feedforward_force_offset": [-5.2, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.2, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
    "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 25.0}
  }
  ```
* **Validation Rules:**
  * $dF_x \in [-5.5, -4.5]\text{ N}$ (**Crucial:** Must command negative feedforward force to tilt into wind!).
  * $dx \approx 0.0\text{ m}$ (holds waypoint station rather than running away).

---

### 4. TC-04: Motor 1 Degradation Recovery (Actuator Failure)
* **Objective:** Compensate for a chipped propeller or degraded motor (Motor 1 at 40% power).
* **FPV Image:** [`vlm/test_cases/tc_04_motor1_failure_recovery.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_04_motor1_failure_recovery.jpg) (slight roll tilt, torque imbalance).
* **Telemetry Input:**
  * Motor Thrusts: $M_1 = 5.80\text{ N}$ vs $M_2..M_4 \approx 2.90\text{ N}$ (Motor 1 saturated/failing)
  * Disturbance Moment: $\hat{\sigma}_M = [0.18, 0.22, 0.05]\text{ N}\cdot\text{m}$
* **Mission Instruction:** *"Diagnose flight anomaly, compensate actuator degradation, and maintain stable hover."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "Motor 1 failure/degradation detected (40% thrust deficit). Applying 2.5x trim to restore M1 effective output.",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "max_velocity": 0.6,
      "corridor_width": 0.6
    },
    "feedforward_force_offset": [0.0, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
    "motor_trim_scale": [2.5, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 18.0}
  }
  ```
* **Validation Rules:**
  * $\text{Trim}_{M1} \in [2.0, 2.5]\times$ (restores M1 output).
  * $\text{max\_tilt\_deg} \le 20.0^\circ$ (tightens envelope to prevent rollover).

---

### 5. TC-05: Gyro Chatter & Noise Damping (Sensor Anomaly)
* **Objective:** Suppress high-frequency vibration/chatter without false-alarm force offsets.
* **FPV Image:** [`vlm/test_cases/tc_05_gyro_chatter_damping.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_05_gyro_chatter_damping.jpg).
* **Telemetry Input:**
  * Gyro RMS: $\omega_x = 1.45\text{ rad/s}, \omega_y = 1.38\text{ rad/s}$ (severe angular rate chatter)
  * Disturbance Force: $0.1\text{ N}$ (negligible force)
  * `oscillation_detected: true`
* **Mission Instruction:** *"Damp high-frequency vibration chatter and hold altitude."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "High-frequency gyro chatter detected. Increasing derivative damping (att_kd) and softening proportional gain (att_kp).",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [0.0, 0.0, 0.0],
      "yaw_deg": 0.0,
      "max_velocity": 0.8,
      "corridor_width": 0.6
    },
    "feedforward_force_offset": [0.0, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 0.4, "att_kd": 1.6},
    "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 20.0}
  }
  ```
* **Validation Rules:**
  * `att_kd` $\ge 1.4\times$ (boosts angular rate damping).
  * `att_kp` $\le 0.5\times$ (softens proportional gain).
  * Feedforward force must remain zero (rejects false alarm).

---

### 6. TC-06: Corridor Obstacle Evasion (Spatial Evasion)
* **Objective:** Detect an obstacle directly blocking the corridor in the FPV view and command lateral evasion.
* **FPV Image:** [`vlm/test_cases/tc_06_corridor_obstacle_evasion.jpg`](file:///Users/anuragkodle/Desktop/meta-hackathon/system1-system2-runtime/vlm/test_cases/tc_06_corridor_obstacle_evasion.jpg) (Drone 1 parked directly in Drone 2's sightline at 1.1m distance; clear airspace on left/right).
* **Telemetry Input:**
  * Proximity warning: *"Obstacle detected dead-ahead at 1.1m distance."*
* **Mission Instruction:** *"Proceed down the corridor, avoid the blocking obstacle directly ahead by steering around it into clear space."*
* **Ground Truth Answer:**
  ```json
  {
    "diagnosis": "Obstacle directly obstructing central flight corridor at 1.1m. Commanding lateral bypass waypoint into open airspace.",
    "waypoint_command": {
      "mode": "relative_body",
      "target": [0.9, 0.65, 0.0],
      "yaw_deg": 10.0,
      "max_velocity": 0.8,
      "corridor_width": 0.8
    },
    "feedforward_force_offset": [0.0, 0.0, 0.0],
    "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
    "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
    "safety_envelope": {"max_tilt_deg": 20.0}
  }
  ```
* **Validation Rules:**
  * $|dy| \ge 0.35\text{ m}$ (must command significant lateral offset to clear obstacle).
  * $dx \in [0.4, 2.0]\text{ m}$ (advances past the obstacle).

---

## How to Run the Benchmark Evaluator

### Run Ground-Truth Sanity Check:
```bash
uv run python vlm/evaluate_model.py --backend ground_truth
```

### Run Google Gemini 3.5 Flash-Lite:
```bash
uv run python vlm/evaluate_model.py --backend gemini
```

### Run Local Ollama Qwen2.5-VL:
```bash
uv run python vlm/evaluate_model.py --backend ollama --model qwen2.5vl:3b
```
