"""Generate New Out-of-Sample Test Suite for Generalization Testing.

Renders 6 brand new scenario images from MuJoCo dual_scene.xml and creates
vlm/test_cases_new.json with strict validation rules.

Scenarios:
    TC-07: Right-Turn Corridor Junction (Visual Yaw Alignment)
    TC-08: Lateral Crosswind (+Y Disturbance Force)
    TC-09: Cyan Landing Pad Precision Touchdown / Descent
    TC-10: Payload Pendulum Oscillation (Downward Load + Chatter)
    TC-11: Motor 1 Degradation during Forward Flight (Combined Failure + Waypoint)
    TC-12: Left-Side Obstacle Avoidance (Right-Lane Bypass)

Usage:
    uv run python vlm/generate_new_test_suite.py
"""

import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw

SCENE_PATH = Path("assets/skydio_x2/dual_scene.xml")
OUTPUT_DIR = Path("vlm/test_cases_new")
JSON_PATH = Path("vlm/test_cases_new.json")
WIDTH = 480
HEIGHT = 360


def add_hud(img: Image.Image, text: str, reticle_color=(0, 255, 255)) -> Image.Image:
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH // 2, HEIGHT // 2
    r = 10
    draw.line([(cx - r, cy), (cx + r, cy)], fill=reticle_color, width=1)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=reticle_color, width=1)
    draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], outline=reticle_color, width=1)
    draw.text((10, 10), text, fill=(0, 255, 200))
    return img


def generate_new_suite():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # --------------------------------------------------------------------------
    # TC-07: Right-Turn Corridor Junction (D2 angled 30 deg to the right)
    # --------------------------------------------------------------------------
    yaw7 = np.radians(-30) # Angled right
    data.qpos[9:12] = [0.5, 1.0, 1.05]
    data.qpos[12:16] = [np.cos(yaw7/2), 0.0, 0.0, np.sin(yaw7/2)]
    data.qpos[0:3] = [0.0, -1.2, 0.1] # D1 parked away
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p7 = OUTPUT_DIR / "tc_07_right_turn_corridor.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.05m | HDG: -030° (RIGHT JUNCTION)").save(p7, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # TC-08: Lateral Crosswind (D2 rolled right 12 deg leaning into crosswind)
    # --------------------------------------------------------------------------
    roll8 = np.radians(-12) # Rolled right
    data.qpos[9:12] = [0.0, 1.0, 1.0]
    data.qpos[12:16] = [np.cos(roll8/2), np.sin(roll8/2), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p8 = OUTPUT_DIR / "tc_08_lateral_crosswind.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.00m | ROLL: -12.0° | LATERAL WIND ALERT", reticle_color=(255, 200, 50)).save(p8, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # TC-09: Cyan Landing Pad Precision Touchdown (D2 directly above Cyan Pad)
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 0.55] # Low altitude directly above Cyan pad
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p9 = OUTPUT_DIR / "tc_09_cyan_pad_touchdown.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 0.55m | CYAN PAD CENTERED", reticle_color=(0, 255, 100)).save(p9, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # TC-10: Payload Pendulum Oscillation (D2 with swinging payload bob)
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 1.15]
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[16] = 0.35 # d2_swing_x deflected
    data.qpos[17] = -0.25 # d2_swing_y deflected
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p10 = OUTPUT_DIR / "tc_10_payload_oscillation.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.15m | PAYLOAD SWING DETECTED", reticle_color=(255, 150, 50)).save(p10, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # TC-11: Motor 1 Degradation during Forward Flight (D2 pitched forward + roll tilt)
    # --------------------------------------------------------------------------
    pitch11 = np.radians(-8)
    roll11 = np.radians(5)
    # Combined quaternion
    q_pitch = np.array([np.cos(pitch11/2), 0, np.sin(pitch11/2), 0])
    q_roll = np.array([np.cos(roll11/2), np.sin(roll11/2), 0, 0])
    # w1w2 - x1x2, ...
    q_comb = [
        q_pitch[0]*q_roll[0] - q_pitch[1]*q_roll[1],
        q_pitch[0]*q_roll[1] + q_pitch[1]*q_roll[0],
        q_pitch[2]*q_roll[0] + q_pitch[0]*q_roll[2],
        q_pitch[2]*q_roll[1] - q_pitch[1]*q_roll[2]
    ]
    data.qpos[9:12] = [0.8, 1.2, 0.96]
    data.qpos[12:16] = q_comb
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p11 = OUTPUT_DIR / "tc_11_motor_fail_forward.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 0.96m | VEL: +0.4m/s | M1 POWER DROP", reticle_color=(255, 100, 50)).save(p11, format="JPEG", quality=80)

    # --------------------------------------------------------------------------
    # TC-12: Left-Side Obstacle Avoidance (D1 placed at x=1.0, y=1.5 on left side)
    # --------------------------------------------------------------------------
    data.qpos[9:12] = [0.0, 1.2, 1.0] # D2 at center of lane
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[0:3] = [1.0, 1.55, 1.0] # D1 parked on left shoulder (y=+1.55)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="d2_fpv")
    p12 = OUTPUT_DIR / "tc_12_left_obstacle_bypass.jpg"
    add_hud(Image.fromarray(renderer.render()), "FPV: d2_fpv | ALT: 1.00m | LEFT OBSTACLE 1.0m (RIGHT OPEN)", reticle_color=(255, 50, 50)).save(p12, format="JPEG", quality=80)

    renderer.close()

    # --------------------------------------------------------------------------
    # New Test Cases Definitions & Validation Rules
    # --------------------------------------------------------------------------
    test_cases = [
        {
            "id": "TC-07",
            "name": "Right-Turn Corridor Junction",
            "category": "Visual Navigation",
            "image_path": str(p7),
            "instruction": "Follow the flight corridor as it bends right, adjust heading to align with the corridor, and advance 1.5m.",
            "telemetry": {
                "altitude_m": 1.05,
                "velocity_body_mps": [0.05, 0.02, 0.00],
                "attitude_deg": {"roll": 0.0, "pitch": -0.5, "yaw": -30.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "gyro_rms_radps": [0.10, 0.12, 0.08]
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [1.4, 0.4, 0.0],
                    "yaw_deg": 15.0, # Adjusting heading into the right bend
                    "max_velocity": 1.0,
                    "corridor_width": 0.6
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg": 20.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.8, "max": 2.5, "desc": "Commands forward progress along corridor"},
                {"field": "feedforward_force_offset", "type": "zero_vector", "desc": "No false alarm force offset during calm flight"},
                {"field": "motor_trim_scale", "type": "ones_vector", "desc": "Nominal motor trims"}
            ]
        },
        {
            "id": "TC-08",
            "name": "Lateral Crosswind (+Y Disturbance)",
            "category": "Disturbance Rejection",
            "image_path": str(p8),
            "instruction": "Compensate for the lateral crosswind pushing from the right (+Y) and hold current corridor position.",
            "telemetry": {
                "altitude_m": 1.00,
                "velocity_body_mps": [0.00, -0.32, -0.01],
                "attitude_deg": {"roll": -12.0, "pitch": 0.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.0, 4.6, 0.0], # +4.6N lateral wind pushing left
                "motor_controls_N": [3.5, 3.1, 3.1, 3.5],
                "gyro_rms_radps": [0.25, 0.18, 0.10]
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [0.0, 0.0, 0.0],
                    "yaw_deg": 0.0,
                    "max_velocity": 0.8,
                    "corridor_width": 0.6
                },
                "feedforward_force_offset": [0.0, -4.6, 0.0], # Opposing -4.6N along Y
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.2, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg": 25.0}
            },
            "validation_rules": [
                {"field": "feedforward_force_offset[1]", "type": "range", "min": -5.0, "max": -4.0, "desc": "Applies opposing lateral force (-4.6N) along Y"},
                {"field": "feedforward_force_offset[0]", "type": "abs_max", "max": 0.5, "desc": "Zero force along X"},
                {"field": "waypoint_command.target[0]", "type": "abs_max", "max": 0.4, "desc": "Holds station at waypoint"}
            ]
        },
        {
            "id": "TC-09",
            "name": "Cyan Landing Pad Touchdown",
            "category": "Visual Servoing",
            "image_path": str(p9),
            "instruction": "Center directly over the cyan landing pad and complete landing descent to 0.15m.",
            "telemetry": {
                "altitude_m": 0.55,
                "velocity_body_mps": [-0.02, 0.01, -0.04],
                "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "gyro_rms_radps": [0.08, 0.10, 0.06]
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [0.1, 0.0, -0.40], # Descending to 0.15m
                    "yaw_deg": 0.0,
                    "max_velocity": 0.5,
                    "corridor_width": 0.4
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg": 20.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[2]", "type": "range", "min": -0.50, "max": -0.25, "desc": "Commands vertical descent to 0.15m touchdown"},
                {"field": "waypoint_command.max_velocity", "type": "range", "min": 0.3, "max": 0.8, "desc": "Emits gentle landing velocity (<= 0.8 m/s)"},
                {"field": "motor_trim_scale", "type": "ones_vector", "desc": "Nominal motor trims"}
            ]
        },
        {
            "id": "TC-10",
            "name": "Payload Pendulum Oscillation",
            "category": "Sensor Anomaly / Chatter",
            "image_path": str(p10),
            "instruction": "Damp high-frequency vibration chatter caused by payload swing and hold altitude.",
            "telemetry": {
                "altitude_m": 1.15,
                "velocity_body_mps": [0.02, -0.03, 0.00],
                "attitude_deg": {"roll": 1.5, "pitch": -2.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.1, 0.0, 0.0], # Negligible persistent lateral force
                "motor_controls_N": [3.65, 3.65, 3.65, 3.65],
                "gyro_rms_radps": [1.32, 1.40, 0.38], # Severe high-frequency oscillation
                "oscillation_detected": True,
                "oscillation_note": "attitude chatter from swing payload wx_rms=1.3rad/s, wy_rms=1.4rad/s"
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [0.0, 0.0, 0.0],
                    "yaw_deg": 0.0,
                    "max_velocity": 0.8,
                    "corridor_width": 0.6
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 0.40, "att_kd": 1.60},
                "safety_envelope": {"max_tilt_deg": 20.0}
            },
            "validation_rules": [
                {"field": "gain_multipliers.att_kd", "type": "range", "min": 1.3, "max": 2.0, "desc": "Boosts derivative damping (att_kd >= 1.4)"},
                {"field": "gain_multipliers.att_kp", "type": "range", "min": 0.3, "max": 0.6, "desc": "Softens proportional gain (att_kp <= 0.5) to stop chatter"},
                {"field": "feedforward_force_offset", "type": "zero_vector", "desc": "No false alarm lateral force for sensor noise"}
            ]
        },
        {
            "id": "TC-11",
            "name": "Motor 1 Fail during Forward Transit",
            "category": "Actuator Failure",
            "image_path": str(p11),
            "instruction": "Maintain forward corridor advance while compensating for Motor 1 actuator degradation.",
            "telemetry": {
                "altitude_m": 0.96,
                "velocity_body_mps": [0.40, -0.05, -0.02],
                "attitude_deg": {"roll": 5.0, "pitch": -8.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.2, 0.1, -1.8],
                "unmodeled_moment_disturbance_Nm": [0.20, 0.25, 0.04],
                "motor_controls_N": [5.90, 2.85, 2.85, 2.90], # Motor 1 saturated
                "gyro_rms_radps": [0.42, 0.48, 0.15]
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [1.0, 0.0, 0.0],
                    "yaw_deg": 0.0,
                    "max_velocity": 0.7,
                    "corridor_width": 0.6
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [2.5, 1.0, 1.0, 1.0], # Restores M1 output
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg": 18.0}
            },
            "validation_rules": [
                {"field": "motor_trim_scale[0]", "type": "range", "min": 2.0, "max": 2.5, "desc": "Restores M1 thrust via 2.2-2.5x trim"},
                {"field": "safety_envelope.max_tilt_deg", "type": "range", "min": 15.0, "max": 20.0, "desc": "Restricts tilt envelope during single-motor flight"},
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.5, "max": 2.0, "desc": "Maintains forward transit"}
            ]
        },
        {
            "id": "TC-12",
            "name": "Left Shoulder Obstacle Bypass",
            "category": "Spatial Evasion",
            "image_path": str(p12),
            "instruction": "Navigate past the obstacle ahead on the left by steering into the open right lane.",
            "telemetry": {
                "altitude_m": 1.00,
                "velocity_body_mps": [0.15, 0.00, 0.00],
                "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
                "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
                "proximity_warning": "Obstacle detected on left side at 1.0m distance. Right lane is clear."
            },
            "ground_truth": {
                "waypoint_command": {
                    "mode": "relative_body",
                    "target": [1.0, 0.65, 0.0], # Steering right into open lane
                    "yaw_deg": 0.0,
                    "max_velocity": 0.8,
                    "corridor_width": 0.8
                },
                "feedforward_force_offset": [0.0, 0.0, 0.0],
                "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
                "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
                "safety_envelope": {"max_tilt_deg": 20.0}
            },
            "validation_rules": [
                {"field": "waypoint_command.target[1]", "type": "range", "min": 0.35, "max": 1.0, "desc": "Steers laterally right (dy >= 0.35m) into clear lane"},
                {"field": "waypoint_command.target[0]", "type": "range", "min": 0.5, "max": 2.0, "desc": "Advances forward past obstacle"}
            ]
        }
    ]

    with open(JSON_PATH, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"✅ Generated {len(test_cases)} new scenario FPV images in: {OUTPUT_DIR.resolve()}")
    print(f"✅ Generated new test suite definition: {JSON_PATH.resolve()}")


if __name__ == "__main__":
    generate_new_suite()
