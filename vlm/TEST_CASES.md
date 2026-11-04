# Autonomous Flight VLM Benchmark: Test Suite & Ground Truth Reference

This benchmark evaluates any Multimodal Vision-Language Model (VLM) acting as the **System-2 Cognitive Reasoner** for the Skydio X2 quadrotor in MuJoCo simulation.

It assesses whether the vision-language model can simultaneously:
1. **Perceive FPV visual scenes** (metric HUD rings, horizon ladder, landing pads, obstacle pylons, 3D aperture gates).
2. **Translate mission instructions into 3D navigation waypoints** (`waypoint_command`).
3. **Reject physical chaos & actuator anomalies** (`feedforward_force_offset`, `motor_trim_scale`, `gain_multipliers`, `safety_envelope`).

---

## Benchmark Suites Overview

| Suite Identifier | Description | Cases | Target Application |
|---|---|---|---|
| **`single`** (Default) | Single-Drone Target Navigation & Visual Servoing | **SD-01 to SD-14** | Autonomous target acquisition (Pad Alpha, Pedestal Delta, Apertures, Chaos rejection) |
| **`canonical`** | Dual-Drone Canonical Disturbance & Evasion | **TC-01 to TC-06** | Baseline flight corridor navigation, wind rejection, motor trim, obstacle bypass |
| **`new`** | Out-of-Sample Generalization Benchmark | **TC-07 to TC-12** | Curved corridors, touchdown descent, pendulum chatter, forward transit motor fault |

---

## Suite 1: Single-Drone Target Navigation (SD-01 to SD-14)

### SD-01: Nominal Pad Alpha Hover & Alignment
- **Category:** Visual Navigation & Alignment
- **Instruction:** *"Maintain nominal flight alignment toward Target Pad Alpha along the forward corridor, hold 1.0m altitude, and advance smoothly."*
- **Key Criterion:** Advances forward along runway corridor ($dx \in [0.8, 1.8]\text{ m}, |dy| \le 0.2\text{ m}$), holding nominal $1.0\text{ m}$ altitude.

### SD-02: Delta: Navigate to Raised Disc Platform
- **Category:** Elevated Target Acquisition
- **Instruction:** *"Navigate forward onto Elevated Station Delta and command hover waypoint directly over the platform disc (clearance 0.35m above platform top)."*
- **Key Criterion:** Commands forward climb onto elevated pedestal disc at $Z \approx 1.10\text{ m}$ ($dx \approx 1.8\text{ m}, dz \ge +0.05\text{ m}$).

### SD-03: Lateral Offset to Target Pad Bravo
- **Category:** Lateral Target Tracking
- **Instruction:** *"Navigate toward Target Pad Bravo on the forward-left flank (Cyan / Blue multi-ring pad), adjusting heading and commanding left-lateral waypoint tracking."*
- **Key Criterion:** Steers left toward Cyan pad ($dy \in [0.4, 1.5]\text{ m}, \text{yaw} > 0^\circ$).

### SD-04: Lateral Offset to Target Pad Charlie
- **Category:** Lateral Target Tracking
- **Instruction:** *"Navigate toward Target Pad Charlie on the forward-right flank (Emerald green pad), adjusting heading and commanding right-lateral waypoint tracking."*
- **Key Criterion:** Steers right toward Emerald pad ($dy \in [-1.5, -0.4]\text{ m}, \text{yaw} < 0^\circ$).

### SD-05: Pylon Gate Obstacle Traversal
- **Category:** Obstacle Traversal
- **Instruction:** *"Traverse through the center of the pylon obstacle gate, hold zero lateral drift (stay centered between left and right pylons), and maintain 1.0m altitude."*
- **Key Criterion:** Safe centerline traversal ($dx > 0.8\text{ m}, |dy| \le 0.15\text{ m}$) avoiding pylon collisions.

### SD-06: Target Pad Alpha Station-Keeping under Wind Gust
- **Category:** Disturbance Rejection
- **Instruction:** *"Station-keep over Target Pad Alpha against strong headwind gust (+5.2N along +X). Inject opposing feedforward force to maintain position."*
- **Key Criterion:** Opposing feedforward force $F_x \in [-5.5, -4.5]\text{ N}$, station-keeping drift $|dx| \le 0.2\text{ m}$.

### SD-07: Degraded Motor Hover over Target Pad
- **Category:** Actuator Failure Compensation
- **Instruction:** *"Compensate for Motor 1 actuator degradation (40% power output) while holding hover over Target Pad Alpha. Apply motor trim scale and tighten safety tilt."*
- **Key Criterion:** Motor 1 trim scale $\in [2.0, 2.5]\times$, safety tilt clamp $\le 18^\circ$.

### SD-08: Alpha: Precision Hover Between 2 Runway Poles
- **Category:** Runway Corridor Station-Keeping
- **Instruction:** *"Hold precision station-keeping hover centered between the two runway poles (pylons at Y=±0.9m) framing Pad Alpha. Maintain 1.0m altitude, keep zero lateral drift (stay equidistant between poles at dy=0.0m), and hold station."*
- **Key Criterion:** Dead-center lateral hold ($|dy| \le 0.10\text{ m}$) centered at $Y=0.0\text{ m}$ between the dual runway pylons.

### SD-09: Delta: Precision Station-Keeping Atop Raised Disc Platform
- **Category:** Elevated Platform Station-Keeping
- **Instruction:** *"Maintain precision station-keeping hover directly atop Elevated Pedestal Delta (centered over the yellow caution disc). Hold 1.10m altitude (0.40m clearance above platform disc) and zero horizontal drift."*
- **Key Criterion:** Altitude hold over raised disc ($Z = 1.10\text{ m} \pm 0.05\text{ m}$) with tight horizontal position hold ($|dx|, |dy| \le 0.12\text{ m}$).

### SD-10: Left Window Sector Scan (Unobservable Sector Saccade +90°)
- **Category:** Active Perception Saccade
- **Instruction:** *"Objective is Left Window located at Y=+5.0m in Sector Left. Current heading is 000° (window unobservable). Perform an active perception in-place yaw saccade to face Sector Left (+90°)."*
- **Key Criterion:** In-place yaw command $\text{yaw\_deg} \in [+85^\circ, +95^\circ]$, position displacement $\approx [0, 0, 0]\text{ m}$.

### SD-11: Left Window Aperture Ingress
- **Category:** Aperture Traversal
- **Instruction:** *"Traverse forward through the Left Window aperture opening at Z=1.10m. Maintain tight attitude safety tilt envelope (max 12.0°) and approach velocity (0.6 m/s) to pass safely within the inner clearance frame."*
- **Key Criterion:** Target aligns with window aperture center ($Z \approx 1.10\text{ m}$), conservative speed ($v \le 0.7\text{ m/s}$), and tight safety envelope ($\text{max\_tilt} \le 15^\circ$).

### SD-12: Right Arch Sector Scan (Unobservable Sector Saccade -90°)
- **Category:** Active Perception Saccade
- **Instruction:** *"Objective is Right Arch located at Y=-5.0m in Sector Right. Current heading is 000° (arch unobservable). Perform an active perception in-place yaw saccade to face Sector Right (-90°)."*
- **Key Criterion:** In-place yaw command $\text{yaw\_deg} \in [-95^\circ, -85^\circ]$, position displacement $\approx [0, 0, 0]\text{ m}$.

### SD-13: Right Arch Aperture Ingress
- **Category:** Aperture Traversal
- **Instruction:** *"Traverse forward through the Right Arch aperture opening at Z=0.90m. Maintain tight attitude safety tilt envelope (max 12.0°) and approach velocity (0.6 m/s) to pass safely within the inner clearance frame."*
- **Key Criterion:** Target aligns with arch portal center ($Z \approx 0.90\text{ m}$), conservative speed ($v \le 0.7\text{ m/s}$), and tight safety envelope ($\text{max\_tilt} \le 15^\circ$).

### SD-14: Delta: Platform Station-Keeping under Crosswind
- **Category:** Disturbance Rejection
- **Instruction:** *"Maintain station-keeping hover atop Elevated Station Delta against a strong lateral crosswind disturbance (+4.8N along +Y). Inject opposing feedforward force to prevent drifting off the raised platform."*
- **Key Criterion:** Lateral feedforward force $F_y \in [-5.2, -4.2]\text{ N}$, keeping the drone firmly planted atop the elevated disc.

---

## Suite 2: Canonical Dual-Drone Benchmark (TC-01 to TC-06)

1. **TC-01: Corridor Forward Advance:** Clear corridor nominal advance ($dx \in [0.8, 3.0]\text{ m}, |dy| \le 0.3\text{ m}$).
2. **TC-02: Target Pad Approach & Centering:** Visual servoing onto landing pad ($dz \in [-0.6, -0.2]\text{ m}$).
3. **TC-03: Headwind Gust Compensation:** Rejection of $+5.2\text{ N}$ headwind via opposing feedforward ($F_x \in [-5.5, -4.5]\text{ N}$).
4. **TC-04: Motor 1 Degradation Recovery:** Chipped rotor recovery ($\text{Trim}_{M1} \in [2.0, 2.5]\times$, tilt clamp $\le 20^\circ$).
5. **TC-05: Gyro Chatter & Noise Damping:** High-frequency vibration damping (`att_kd` $\ge 1.4\times$, `att_kp` $\le 0.5\times$).
6. **TC-06: Corridor Obstacle Evasion:** Steering around stationary obstacle dead ahead ($|dy| \ge 0.35\text{ m}$).

---

## Suite 3: Generalization Dual-Drone Benchmark (TC-07 to TC-12)

1. **TC-07: Right-Turn Corridor Junction:** Following a curving flight corridor to the right ($dx \ge 0.8\text{ m}, \text{yaw} > 0^\circ$).
2. **TC-08: Lateral Crosswind (+Y Disturbance):** Compensating for $+Y$ crosswind ($F_y \in [-5.5, -4.5]\text{ N}$).
3. **TC-09: Cyan Landing Pad Touchdown:** Low-altitude touchdown descent to $Z = 0.15\text{ m}$ ($dz \le -0.4\text{ m}$).
4. **TC-10: Payload Pendulum Oscillation:** Damping sloshing/pendulum disturbance chatter (`att_kd` boost).
5. **TC-11: Motor 1 Fail during Forward Transit:** Motor failure recovery while simultaneously progressing forward.
6. **TC-12: Left Shoulder Obstacle Bypass:** Lateral bypass around an obstacle encroaching the left flight lane.

---

## Execution Guide

```bash
# Run Ground Truth validation on all suites
uv run python vlm/evaluate.py --suite single --backend ground_truth
uv run python vlm/evaluate.py --suite canonical --backend ground_truth
uv run python vlm/evaluate.py --suite new --backend ground_truth

# Run Gemini 3.5 Flash-Lite against Single-Drone Suite
uv run python vlm/evaluate.py --suite single --backend gemini

# Run Gemini Live Streaming WebSocket against Single-Drone Suite
uv run python vlm/evaluate.py --suite single --backend gemini-live
```
