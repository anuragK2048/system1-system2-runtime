"""Single-Drone Target Navigation VLM Test Suite Generator.

Generates:
1. Synthetic 480x360 MuJoCo FPV camera frames for 7 critical single-drone flight scenarios:
   - SD-01: Nominal Pad Alpha Hover & Alignment
   - SD-02: Elevated Pedestal Delta Hover Acquisition
   - SD-03: Lateral Offset to Target Pad Bravo (Cyan)
   - SD-04: Lateral Offset to Target Pad Charlie (Emerald)
   - SD-05: Pylon Gate Obstacle Traversal
   - SD-06: Target Pad Alpha Station-Keeping under Wind Gust (+5.2N Headwind)
   - SD-07: Degraded Motor Hover over Target Pad (M1 40% Power)
2. Ground truth specification file: vlm/test_cases_single.json containing:
   - Mission instructions & realistic telemetry states
   - Ground truth System-2 patches (WaypointCommand, feedforward, trims, gains)
   - Strict validation rules for target accuracy, safety bounds, and disturbance rejection

Usage:
    uv run python vlm/generate_single_drone_suite.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

SCENE_PATH = Path("assets/skydio_x2/scene.xml")
OUTPUT_DIR = Path("vlm/test_cases_single")
JSON_PATH = Path("vlm/test_cases_single.json")
WIDTH = 480
HEIGHT = 360


def add_hud(
    img: Image.Image,
    top_text: str,
    bottom_text: str = "",
    reticle_color: tuple = (0, 255, 255),
    accent_color: tuple = (0, 255, 200),
) -> Image.Image:
    """Adds tactical boresight reticle, artificial horizon ticks, and HUD overlay."""
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH // 2, HEIGHT // 2

    # Central boresight crosshairs & circle
    r = 12
    draw.line([(cx - r, cy), (cx + r, cy)], fill=reticle_color, width=1)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=reticle_color, width=1)
    draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], outline=reticle_color, width=1)
    draw.ellipse([(cx - 16, cy - 16), (cx + 16, cy + 16)], outline=reticle_color, width=1)

    # Lateral pitch / horizon reference bars
    bar_w = 20
    draw.line([(cx - 50 - bar_w, cy), (cx - 50, cy)], fill=reticle_color, width=2)
    draw.line([(cx + 50, cy), (cx + 50 + bar_w, cy)], fill=reticle_color, width=2)
    draw.line([(cx - 50, cy), (cx - 50, cy + 6)], fill=reticle_color, width=1)
    draw.line([(cx + 50, cy), (cx + 50, cy + 6)], fill=reticle_color, width=1)

    # Corner bracket indicators
    corner_len = 14
    margin = 8
    # Top-left corner
    draw.line([(margin, margin), (margin + corner_len, margin)], fill=accent_color, width=1)
    draw.line([(margin, margin), (margin, margin + corner_len)], fill=accent_color, width=1)
    # Top-right corner
    draw.line([(WIDTH - margin - corner_len, margin), (WIDTH - margin, margin)], fill=accent_color, width=1)
    draw.line([(WIDTH - margin, margin), (WIDTH - margin, margin + corner_len)], fill=accent_color, width=1)
    # Bottom-left corner
    draw.line([(margin, HEIGHT - margin), (margin + corner_len, HEIGHT - margin)], fill=accent_color, width=1)
    draw.line([(margin, HEIGHT - margin - corner_len), (margin, HEIGHT - margin)], fill=accent_color, width=1)
    # Bottom-right corner
    draw.line([(WIDTH - margin - corner_len, HEIGHT - margin), (WIDTH - margin, HEIGHT - margin)], fill=accent_color, width=1)
    draw.line([(WIDTH - margin, HEIGHT - margin - corner_len), (WIDTH - margin, HEIGHT - margin)], fill=accent_color, width=1)

    # Top telemetry string
    draw.text((16, 12), top_text, fill=accent_color)
    if bottom_text:
        draw.text((16, HEIGHT - 24), bottom_text, fill=(255, 220, 80))

    return img


def generate_suite():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"MuJoCo scene XML not found at {SCENE_PATH.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    test_cases: List[Dict[str, Any]] = []

    # ==========================================================================
    # SD-01: Nominal Pad Alpha Hover & Alignment
    # Drone at (0.0, 0.0, 1.0), heading +X, looking directly at Target Pad Alpha
    # ==========================================================================
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img1 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: 000° | SYS: NOMINAL",
        bottom_text="TARGET: PAD ALPHA (ORANGE) | DIST: 5.5m | BEARING: 000°",
        reticle_color=(0, 255, 255),
    )
    p1 = OUTPUT_DIR / "sd_01_nominal_pad_alpha.jpg"
    img1.save(p1, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-01",
        "name": "Nominal Pad Alpha Hover & Alignment",
        "category": "Visual Navigation & Alignment",
        "image_path": str(p1),
        "instruction": "Maintain nominal flight alignment toward Target Pad Alpha along the forward corridor, hold 1.0m altitude, and advance smoothly.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.02, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.10, 0.06],
        },
        "ground_truth": {
            "diagnosis": "Nominal flight aligned with Target Pad Alpha along forward corridor",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 1.0,
                "corridor_width": 0.6,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 20.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.0,
                "max": 3.0,
                "desc": "Commands forward progress toward Target Pad Alpha",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.3,
                "desc": "Maintains centered corridor lane",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "abs_max",
                "max": 0.2,
                "desc": "Maintains level 1.0m hover altitude",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward force during calm nominal flight",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims during unperturbed flight",
            },
        ],
    })

    # ==========================================================================
    # SD-02: Elevated Pedestal Delta Hover Acquisition
    # Drone approaching elevated platform at (8.0, 0.0, 0.70), positioned at (5.5, 0.0, 1.05)
    # ==========================================================================
    data.qpos[0:3] = [5.5, 0.0, 1.05]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img2 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.05m | HDG: 000° | APPROACH: PEDESTAL DELTA",
        bottom_text="TARGET: ELEVATED PEDESTAL DELTA | TOP Z: 0.70m | DIST: 2.5m",
        reticle_color=(255, 230, 50),
    )
    p2 = OUTPUT_DIR / "sd_02_elevated_pedestal_delta.jpg"
    img2.save(p2, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-02",
        "name": "Elevated Pedestal Delta Hover Acquisition",
        "category": "Elevated Target Acquisition",
        "image_path": str(p2),
        "instruction": "Approach Elevated Pedestal Delta and command hover waypoint directly over the platform disc (clearance 0.35m above platform top).",
        "telemetry": {
            "altitude_m": 1.05,
            "velocity_body_mps": [0.08, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": -0.5, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.26, 3.26, 3.24, 3.24],
            "gyro_rms_radps": [0.10, 0.12, 0.07],
        },
        "ground_truth": {
            "diagnosis": "Approaching Elevated Pedestal Delta platform, commanding elevated target hover",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.5, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.8,
                "corridor_width": 0.5,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 18.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.5,
                "max": 3.5,
                "desc": "Advances forward onto Elevated Pedestal Delta platform",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.35,
                "desc": "Remains aligned with center of elevated pedestal disc",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "range",
                "min": -0.3,
                "max": 0.3,
                "desc": "Maintains safe elevated clearance above platform",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward force during calm elevated acquisition",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims during platform acquisition",
            },
        ],
    })

    # ==========================================================================
    # SD-03: Lateral Offset to Target Pad Bravo
    # Drone at (8.0, 0.0, 1.0), aiming toward cyan pad at (11.0, 2.2)
    # ==========================================================================
    yaw3 = np.radians(35)
    data.qpos[0:3] = [8.0, 0.0, 1.0]
    data.qpos[3:7] = [np.cos(yaw3 / 2), 0.0, 0.0, np.sin(yaw3 / 2)]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img3 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: +035° | ROUTE: PAD BRAVO",
        bottom_text="TARGET: PAD BRAVO (CYAN) | BEARING: +035° (LEFT)",
        reticle_color=(0, 220, 255),
    )
    p3 = OUTPUT_DIR / "sd_03_lateral_offset_pad_bravo.jpg"
    img3.save(p3, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-03",
        "name": "Lateral Offset to Target Pad Bravo",
        "category": "Lateral Target Tracking",
        "image_path": str(p3),
        "instruction": "Navigate toward Target Pad Bravo on the forward-left flank (Cyan / Blue multi-ring pad), adjusting heading and commanding left-lateral waypoint tracking.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.04, 0.01, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 35.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.09, 0.11, 0.08],
        },
        "ground_truth": {
            "diagnosis": "Visual acquisition of cyan Target Pad Bravo, vectoring forward-left",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.2, 0.8, 0.0],
                "yaw_deg": 15.0,
                "max_velocity": 1.0,
                "corridor_width": 0.7,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 22.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.0,
                "max": 3.5,
                "desc": "Advances forward toward Pad Bravo",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "range",
                "min": 0.2,
                "max": 2.0,
                "desc": "Vectors laterally left toward Pad Bravo (+Y)",
            },
            {
                "field": "waypoint_command.yaw_deg",
                "type": "range",
                "min": 0.0,
                "max": 35.0,
                "desc": "Aligns heading counter-clockwise toward Target Pad Bravo",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "No feedforward offset during unforced turn",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
        ],
    })

    # ==========================================================================
    # SD-04: Lateral Offset to Target Pad Charlie
    # Drone at (8.0, 0.0, 1.0), aiming toward emerald pad at (11.0, -2.2)
    # ==========================================================================
    yaw4 = np.radians(-35)
    data.qpos[0:3] = [8.0, 0.0, 1.0]
    data.qpos[3:7] = [np.cos(yaw4 / 2), 0.0, 0.0, np.sin(yaw4 / 2)]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img4 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: -035° | ROUTE: PAD CHARLIE",
        bottom_text="TARGET: PAD CHARLIE (EMERALD) | BEARING: -035° (RIGHT)",
        reticle_color=(50, 255, 120),
    )
    p4 = OUTPUT_DIR / "sd_04_lateral_offset_pad_charlie.jpg"
    img4.save(p4, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-04",
        "name": "Lateral Offset to Target Pad Charlie",
        "category": "Lateral Target Tracking",
        "image_path": str(p4),
        "instruction": "Navigate toward Target Pad Charlie on the forward-right flank (Emerald green pad), adjusting heading and commanding right-lateral waypoint tracking.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.04, -0.01, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": -35.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.09, 0.11, 0.08],
        },
        "ground_truth": {
            "diagnosis": "Visual acquisition of emerald Target Pad Charlie, vectoring forward-right",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.2, -0.8, 0.0],
                "yaw_deg": -15.0,
                "max_velocity": 1.0,
                "corridor_width": 0.7,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 22.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.0,
                "max": 3.5,
                "desc": "Advances forward toward Pad Charlie",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "range",
                "min": -2.0,
                "max": -0.2,
                "desc": "Vectors laterally right toward Pad Charlie (-Y)",
            },
            {
                "field": "waypoint_command.yaw_deg",
                "type": "range",
                "min": -35.0,
                "max": 0.0,
                "desc": "Aligns heading clockwise toward Target Pad Charlie",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "No feedforward offset during unforced turn",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
        ],
    })

    # ==========================================================================
    # SD-05: Pylon Gate Obstacle Traversal
    # Drone facing narrow passage between pylons at (3.5, +/-0.9), drone at (1.8, 0.0, 1.0)
    # ==========================================================================
    data.qpos[0:3] = [1.8, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img5 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: 000° | CAUTION: OBSTACLE GATE",
        bottom_text="PYLON OBSTACLES DETECTED | CORRIDOR WIDTH: 1.8m | STAY CENTERED",
        reticle_color=(255, 80, 80),
    )
    p5 = OUTPUT_DIR / "sd_05_pylon_gate_traversal.jpg"
    img5.save(p5, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-05",
        "name": "Pylon Gate Obstacle Traversal",
        "category": "Obstacle Traversal",
        "image_path": str(p5),
        "instruction": "Traverse through the center of the pylon obstacle gate, hold zero lateral drift (stay centered between left and right pylons), and maintain 1.0m altitude.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.03, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.09, 0.10, 0.07],
        },
        "ground_truth": {
            "diagnosis": "Approaching pylon gate obstacle; maintaining centered lane through passage",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.8,
                "corridor_width": 0.5,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 18.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.2,
                "max": 2.5,
                "desc": "Commands forward progress through gate passage",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Maintains strict lateral centering to avoid striking pylons",
            },
            {
                "field": "waypoint_command.corridor_width",
                "type": "range",
                "min": 0.2,
                "max": 0.6,
                "desc": "Restricts safety corridor width within pylon clearance",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward force during calm corridor flight",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
        ],
    })

    # ==========================================================================
    # SD-06: Target Pad Alpha Station-Keeping under Wind Gust
    # +5.2N wind disturbance while approaching Alpha pad, requires feedforward: [-5.2, 0, 0]
    # Pitch angle ~ -14 deg nose-down into wind
    # ==========================================================================
    pitch6 = np.radians(-14)
    data.qpos[0:3] = [5.0, 0.0, 1.0]
    data.qpos[3:7] = [np.cos(pitch6 / 2), 0.0, np.sin(pitch6 / 2), 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img6 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | PITCH: -14.0° | WIND DISTURBANCE",
        bottom_text="HEADWIND DETECTED: +5.2N | COMMANDING COUNTER-THRUST FEEDFORWARD",
        reticle_color=(255, 120, 0),
    )
    p6 = OUTPUT_DIR / "sd_06_target_alpha_wind_gust.jpg"
    img6.save(p6, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-06",
        "name": "Target Pad Alpha Station-Keeping under Wind Gust",
        "category": "Disturbance Rejection",
        "image_path": str(p6),
        "instruction": "Station-keep over Target Pad Alpha against strong headwind gust (+5.2N along +X). Inject opposing feedforward force to maintain position.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.02, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": -14.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [5.20, 0.00, 0.00],
            "motor_controls_N": [3.65, 3.65, 2.85, 2.85],
            "gyro_rms_radps": [0.18, 0.22, 0.12],
        },
        "ground_truth": {
            "diagnosis": "Persistent headwind disturbance (+5.2N along +X); injecting opposing feedforward",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.5,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [-5.2, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.2, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 25.0},
        },
        "validation_rules": [
            {
                "field": "feedforward_force_offset[0]",
                "type": "range",
                "min": -5.5,
                "max": -4.8,
                "desc": "Injects opposing feedforward force dFx in [-5.5, -4.8] N to cancel +5.2N wind",
            },
            {
                "field": "feedforward_force_offset[1]",
                "type": "abs_max",
                "max": 0.3,
                "desc": "Zero lateral feedforward force",
            },
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Holds station hover rather than forward drift",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims during wind disturbance",
            },
        ],
    })

    # ==========================================================================
    # SD-07: Degraded Motor Hover over Target Pad
    # Motor 1 degradation at 40% power, requires motor_trim_scale: [2.5, 1.0, 1.0, 1.0]
    # Roll tilt ~ +6 deg from actuator degradation
    # ==========================================================================
    roll7 = np.radians(6)
    data.qpos[0:3] = [5.5, 0.0, 0.95]
    data.qpos[3:7] = [np.cos(roll7 / 2), np.sin(roll7 / 2), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img7 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 0.95m | ROLL: +6.0° | ACTUATOR ALERT",
        bottom_text="MOTOR 1 DEGRADATION (40% POWER) | APPLYING 2.50x ACTUATOR TRIM",
        reticle_color=(255, 60, 60),
    )
    p7 = OUTPUT_DIR / "sd_07_degraded_motor_hover.jpg"
    img7.save(p7, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-07",
        "name": "Degraded Motor Hover over Target Pad",
        "category": "Actuator Failure Compensation",
        "image_path": str(p7),
        "instruction": "Compensate for Motor 1 actuator degradation (40% power output) while holding hover over Target Pad Alpha. Apply motor trim scale and tighten safety tilt.",
        "telemetry": {
            "altitude_m": 0.95,
            "velocity_body_mps": [0.01, 0.01, -0.01],
            "attitude_deg": {"roll": 6.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [5.80, 2.90, 2.90, 2.90],
            "gyro_rms_radps": [0.25, 0.15, 0.12],
        },
        "ground_truth": {
            "diagnosis": "Motor 1 severely degraded at 40% efficiency; applying 2.50x motor trim scale",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.5,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [2.5, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 18.0},
        },
        "validation_rules": [
            {
                "field": "motor_trim_scale[0]",
                "type": "range",
                "min": 2.3,
                "max": 2.5,
                "desc": "Boosts Motor 1 trim scale to [2.3, 2.5] to restore 40% power loss",
            },
            {
                "field": "motor_trim_scale[1]",
                "type": "range",
                "min": 0.9,
                "max": 1.1,
                "desc": "Maintains Motor 2 nominal trim",
            },
            {
                "field": "motor_trim_scale[2]",
                "type": "range",
                "min": 0.9,
                "max": 1.1,
                "desc": "Maintains Motor 3 nominal trim",
            },
            {
                "field": "motor_trim_scale[3]",
                "type": "range",
                "min": 0.9,
                "max": 1.1,
                "desc": "Maintains Motor 4 nominal trim",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "No external wind feedforward required during motor failure",
            },
            {
                "field": "safety_envelope.max_tilt_deg",
                "type": "range",
                "min": 15.0,
                "max": 20.0,
                "desc": "Tightens maximum allowable tilt envelope during degraded actuator flight",
            },
        ],
    })

    # Save to JSON
    with open(JSON_PATH, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"✅ Generated {len(test_cases)} single-drone test frames in {OUTPUT_DIR}")
    print(f"📄 Written ground truth test suite to {JSON_PATH}")
    for tc in test_cases:
        print(f"   • [{tc['id']}] {tc['name']} -> {tc['image_path']}")


if __name__ == "__main__":
    generate_suite()
