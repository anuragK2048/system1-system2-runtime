"""Test Suite Generator for Drone VLM Multimodal Reasoning.

Generates:
1. Synthetic 480x360 MuJoCo FPV camera frames for 6 critical drone scenarios.
2. Ground truth specification file: vlm/test_cases.json containing:
   - Instruction & metric telemetry prompt
   - Ground truth reference values
   - Scoring & acceptance criteria (tolerance bands, required keys)

Usage:
    uv run python vlm/generate_test_suite.py
"""

import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw

SCENE_PATH = Path("assets/skydio_x2/dual_scene.xml")
OUTPUT_DIR = Path("vlm/test_cases")
JSON_PATH = Path("vlm/test_cases.json")
WIDTH = 480
HEIGHT = 360


def add_hud(img: Image.Image, text: str, reticle_color=(0, 255, 255)) -> Image.Image:
    """Adds boresight reticle and HUD overlay."""
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH // 2, HEIGHT // 2
    r = 10
    draw.line([(cx - r, cy), (cx + r, cy)], fill=reticle_color, width=1)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=reticle_color, width=1)
    draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], outline=reticle_color, width=1)
    draw.text((10, 10), text, fill=(0, 255, 200))
    return img


def generate_suite():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # --------------------------------------------------------------------------
    # Render Scenario 1: Nominal Corridor Forward Advance
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 1.0] # D2
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [0.0, -1.2, 0.1] # D1 parked away
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img1 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.00m | HDG: 000°")
    p1 = OUTPUT_DIR / "tc_01_corridor_advance.jpg"
    img1.save(p1, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # Render Scenario 2: Target Pad Approach (Looking towards Orange Pad)
    # --------------------------------------------------------------------------
    angle2 = np.radians(-115) # Rotate towards orange pad at (0, -1.2)
    data.qpos[9:12] = [0.7, -0.3, 0.85]
    data.qpos[12:16] = [np.cos(angle2/2), 0.0, 0.0, np.sin(angle2/2)]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img2 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 0.85m | HDG: -115°")
    p2 = OUTPUT_DIR / "tc_02_target_pad_approach.jpg"
    img2.save(p2, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # Render Scenario 3: Headwind Tilt
    # --------------------------------------------------------------------------
    pitch_angle = np.radians(-14) # Nose pitched down into simulated 5N wind
    data.qpos[9:12] = [-0.25, 1.2, 0.98]
    data.qpos[12:16] = [np.cos(pitch_angle/2), 0.0, np.sin(pitch_angle/2), 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img3 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 0.98m | PITCH: -14.0° | WIND ALERT")
    p3 = OUTPUT_DIR / "tc_03_headwind_compensation.jpg"
    img3.save(p3, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # Render Scenario 4: Motor 1 Degradation Asymmetry
    # --------------------------------------------------------------------------
    roll_angle = np.radians(6) # Slight roll tilt from actuator degradation
    data.qpos[9:12] = [0.0, 1.2, 0.95]
    data.qpos[12:16] = [np.cos(roll_angle/2), np.sin(roll_angle/2), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img4 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 0.95m | ROLL: +6.0° | M1 WARN")
    p4 = OUTPUT_DIR / "tc_04_motor1_failure_recovery.jpg"
    img4.save(p4, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # Render Scenario 5: High-Frequency Gyro Chatter
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 1.0]
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img5 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.00m | GYRO JITTER RMS 1.4rad/s", reticle_color=(255, 100, 100))
    p5 = OUTPUT_DIR / "tc_05_gyro_chatter_damping.jpg"
    img5.save(p5, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # Render Scenario 6: Corridor Obstacle Evasion (Drone 1 parked right ahead!)
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 1.0] # D2
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [1.1, 1.2, 1.0] # D1 parked 1.1m ahead directly in sightline
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    img6 = add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.00m | PROXIMITY WARN: OBSTACLE 1.1m", reticle_color=(255, 50, 50))
    p6 = OUTPUT_DIR / "tc_06_corridor_obstacle_evasion.jpg"
    img6.save(p6, format="JPEG", quality=80)

    renderer.close()

    # --------------------------------------------------------------------------
    # Test Cases Definitions & Ground Truth Specification
    # --------------------------------------------------------------------------
    test_cases = [
        {
            "id": "TC-01",
            "name": "Corridor Forward Advance",
            "category": "Visual Navigation",
            "image_path": str(p1),
            "instruction": "Proceed forward along the flight corridor toward the target pad, stay centered, and hold 1.0m altitude.",
            "telemetry": {
                "altitude_m": 1.00,
                "velocity_body_mps": [0.02, -0.01, 0.00],
                "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "gyro_rms_radps": [0.12, 0.15, 0.08]
            },
            "ground_truth": {
                "diagnosis_keyword": "corridor",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [1.0, 2.5],
                    "target_dy_range": [-0.2, 0.2],
                    "target_dz_range": [-0.1, 0.1],
                    "yaw_deg_range": [-5.0, 5.0],
                    "max_velocity_range": [0.8, 1.2]
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg_max": 25.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.8, "max": 3.0, "desc": "Commands forward progress along corridor"},
                {"field": "waypoint_command.target[1]", "type": "abs_max", "max": 0.3, "desc": "Maintains lateral corridor centering"},
                {"field": "feedforward_force_offset", "type": "zero_vector", "desc": "Does not inject phantom force during calm nominal flight"},
                {"field": "motor_trim_scale", "type": "ones_vector", "desc": "Keeps nominal motor trim 1.0"}
            ]
        },
        {
            "id": "TC-02",
            "name": "Target Pad Approach & Centering",
            "category": "Visual Servoing",
            "image_path": str(p2),
            "instruction": "Align with the orange target pad, center it in your crosshairs, and descend to 0.4m.",
            "telemetry": {
                "altitude_m": 0.85,
                "velocity_body_mps": [0.15, 0.05, -0.05],
                "attitude_deg": {"roll": 0.2, "pitch": -1.5, "yaw": -115.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "gyro_rms_radps": [0.14, 0.16, 0.09]
            },
            "ground_truth": {
                "diagnosis_keyword": "target",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [0.4, 1.5],
                    "target_dy_range": [-0.5, 0.5],
                    "target_dz_range": [-0.5, -0.3],
                    "yaw_deg_range": [-30.0, 30.0],
                    "max_velocity_range": [0.5, 1.0]
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg_max": 20.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[2]", "type": "range", "min": -0.6, "max": -0.2, "desc": "Commands descent toward 0.4m AGL target"},
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.2, "max": 2.0, "desc": "Advances toward landing pad in front of drone"},
                {"field": "motor_trim_scale", "type": "ones_vector", "desc": "No actuator failure; trim remains 1.0"}
            ]
        },
        {
            "id": "TC-03",
            "name": "Headwind Gust Compensation",
            "category": "Disturbance Rejection",
            "image_path": str(p3),
            "instruction": "Hold station over current waypoint against headwind disturbance.",
            "telemetry": {
                "altitude_m": 0.98,
                "velocity_body_mps": [-0.35, 0.01, -0.02],
                "attitude_deg": {"roll": 0.1, "pitch": -14.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [5.2, 0.0, 0.0], # +5.2N pushing backwards along X
                "motor_controls_N": [3.6, 3.6, 2.9, 2.9],
                "gyro_rms_radps": [0.22, 0.35, 0.12]
            },
            "ground_truth": {
                "diagnosis_keyword": "wind",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [-0.2, 0.2],
                    "target_dy_range": [-0.2, 0.2],
                    "target_dz_range": [-0.1, 0.1],
                    "yaw_deg_range": [-5.0, 5.0],
                    "max_velocity_range": [0.5, 1.0]
                },
                "feedforward_force_offset": [-5.2, 0.0, 0.0], # Equal and opposite to +5.2N
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.2, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg_max": 25.0}
            },
            "validation_rules": [
                {"field": "feedforward_force_offset[0]", "type": "range", "min": -5.5, "max": -4.5, "desc": "Applies opposing feedforward force (-5.2N) against wind"},
                {"field": "waypoint_command.target[0]", "type": "abs_max", "max": 0.5, "desc": "Holds station at waypoint rather than abandoning position"},
                {"field": "gain_multipliers.pos_kp", "type": "range", "min": 1.0, "max": 1.5, "desc": "Maintains or increases position stiffness"}
            ]
        },
        {
            "id": "TC-04",
            "name": "Motor 1 Degradation Recovery",
            "category": "Actuator Failure",
            "image_path": str(p4),
            "instruction": "Diagnose flight anomaly, compensate actuator degradation, and maintain stable hover.",
            "telemetry": {
                "altitude_m": 0.95,
                "velocity_body_mps": [0.01, -0.08, -0.05],
                "attitude_deg": {"roll": 6.0, "pitch": -2.0, "yaw": 0.5},
                "unmodeled_force_disturbance_N": [0.2, 0.1, -1.9],
                "unmodeled_moment_disturbance_Nm": [0.18, 0.22, 0.05],
                "motor_controls_N": [5.80, 2.85, 2.90, 2.95], # Motor 1 struggling / degraded
                "gyro_rms_radps": [0.45, 0.52, 0.18]
            },
            "ground_truth": {
                "diagnosis_keyword": "motor",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [-0.1, 0.1],
                    "target_dy_range": [-0.1, 0.1],
                    "target_dz_range": [-0.1, 0.1],
                    "yaw_deg_range": [-5.0, 5.0],
                    "max_velocity_range": [0.4, 0.8]
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [2.5, 1.0, 1.0, 1.0], # 2.5x trim on degraded Motor 1
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg_max": 20.0}
            },
            "validation_rules": [
                {"field": "motor_trim_scale[0]", "type": "range", "min": 2.0, "max": 2.5, "desc": "Restores M1 thrust via 2.2-2.5x trim (or trims healthy motors)"},
                {"field": "safety_envelope.max_tilt_deg", "type": "range", "min": 15.0, "max": 22.0, "desc": "Tightens tilt envelope to prevent spin during single-motor recovery"}
            ]
        },
        {
            "id": "TC-05",
            "name": "Gyro Chatter & Noise Damping",
            "category": "Sensor Anomaly",
            "image_path": str(p5),
            "instruction": "Damp high-frequency vibration chatter and hold altitude.",
            "telemetry": {
                "altitude_m": 1.00,
                "velocity_body_mps": [0.01, 0.02, 0.00],
                "attitude_deg": {"roll": 0.5, "pitch": -0.4, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.1, 0.0, 0.0], # NO physical force!
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "gyro_rms_radps": [1.45, 1.38, 0.42], # Severe high-frequency oscillation
                "oscillation_detected": True,
                "oscillation_note": "angular rate spike wx_rms=1.4rad/s, wy_rms=1.4rad/s"
            },
            "ground_truth": {
                "diagnosis_keyword": "chatter",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [-0.1, 0.1],
                    "target_dy_range": [-0.1, 0.1],
                    "target_dz_range": [-0.1, 0.1],
                    "yaw_deg_range": [-2.0, 2.0],
                    "max_velocity_range": [0.5, 1.0]
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0], # Do NOT command force offset
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 0.4, "att_kd": 1.6},
                "safety_envelope": {"max_tilt_deg_max": 20.0}
            },
            "validation_rules": [
                {"field": "gain_multipliers.att_kd", "type": "range", "min": 1.3, "max": 2.0, "desc": "Increases angular rate damping (att_kd >= 1.4)"},
                {"field": "gain_multipliers.att_kp", "type": "range", "min": 0.3, "max": 0.6, "desc": "Softens attitude proportional gain (att_kp <= 0.5) to stop chatter"},
                {"field": "feedforward_force_offset", "type": "zero_vector", "desc": "Rejects false alarm (does not apply force when noise only)"}
            ]
        },
        {
            "id": "TC-06",
            "name": "Corridor Obstacle Evasion",
            "category": "Spatial Evasion",
            "image_path": str(p6),
            "instruction": "Proceed down the corridor, avoid the blocking obstacle directly ahead by steering around it into clear space.",
            "telemetry": {
                "altitude_m": 1.00,
                "velocity_body_mps": [0.10, 0.00, 0.00],
                "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "proximity_warning": "Obstacle detected dead-ahead at 1.1m distance."
            },
            "ground_truth": {
                "diagnosis_keyword": "obstacle",
                "waypoint_command": {
                    "mode": "relative_body",
                    "target_dx_range": [0.6, 1.5],
                    "target_dy_range": [-1.0, 1.0], # Lateral bypass command (+ or -)
                    "target_dz_range": [-0.2, 0.2],
                    "yaw_deg_range": [-25.0, 25.0],
                    "max_velocity_range": [0.6, 1.0],
                    "corridor_width_range": [0.6, 1.2]
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg_max": 20.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[1]", "type": "abs_min", "min": 0.35, "desc": "Commands significant lateral offset (|dy| >= 0.35m) to clear obstacle"},
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.4, "max": 2.0, "desc": "Advances forward past obstacle corridor"}
            ]
        }
    ]

    with open(JSON_PATH, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"✅ Generated {len(test_cases)} scenario FPV images in: {OUTPUT_DIR.resolve()}")
    print(f"✅ Generated test suite definition: {JSON_PATH.resolve()}")


if __name__ == "__main__":
    generate_suite()
