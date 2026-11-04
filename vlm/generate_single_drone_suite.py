"""Single-Drone Target Navigation VLM Test Suite Generator.

Generates:
1. Synthetic 480x360 MuJoCo FPV camera frames for 14 critical single-drone flight scenarios:
   - SD-01: Nominal Pad Alpha Hover & Alignment
   - SD-02: Delta: Navigate to Raised Disc Platform (Pedestal Approach)
   - SD-03: Lateral Offset to Target Pad Bravo (Cyan)
   - SD-04: Lateral Offset to Target Pad Charlie (Emerald)
   - SD-05: Pylon Gate Obstacle Traversal
   - SD-06: Target Pad Alpha Station-Keeping under Wind Gust (+5.2N Headwind)
   - SD-07: Degraded Motor Hover over Target Pad (M1 40% Power)
   - SD-08: Alpha: Precision Hover Between 2 Runway Poles (Pylon Centering)
   - SD-09: Delta: Precision Station-Keeping Atop Raised Disc Platform
   - SD-10: Left Window Sector Scan (Unobservable Sector Saccade +90°)
   - SD-11: Left Window Aperture Ingress (Traversing Window at Z=1.10m)
   - SD-12: Right Arch Sector Scan (Unobservable Sector Saccade -90°)
   - SD-13: Right Arch Aperture Ingress (Traversing Arch at Z=0.90m)
   - SD-14: Delta: Platform Station-Keeping under Crosswind (+4.8N Crosswind)
2. Ground truth specification file: vlm/test_cases_single.json containing:
   - Mission instructions & realistic telemetry states
   - Ground truth System-2 patches (WaypointCommand, flight_phase, feedforward, trims, gains)
   - Strict validation rules for target accuracy, safety bounds, and disturbance rejection

Usage:
    uv run python vlm/generate_single_drone_suite.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import mujoco
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm.hud_overlay import HUDProjector, Aperture3D

SCENE_PATH = Path("assets/skydio_x2/scene.xml")
OUTPUT_DIR = Path("vlm/test_cases_single")
JSON_PATH = Path("vlm/test_cases_single.json")
WIDTH = 480
HEIGHT = 360

ARENA_APERTURES = [
    Aperture3D(
        name="WINDOW_LEFT",
        center=(0.0, 5.0, 1.10),
        width=1.40,
        height=1.20,
        yaw=math.radians(90.0),
        shape="window",
        color=(255, 190, 0),       # Amber Gold
        inner_color=(0, 255, 130),  # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    ),
    Aperture3D(
        name="ARCH_RIGHT",
        center=(0.0, -5.0, 0.90),
        width=1.50,
        height=1.85,
        yaw=math.radians(-90.0),
        shape="arch",
        color=(255, 140, 0),       # High-Vis Orange
        inner_color=(0, 255, 130),  # Neon Mint
        safety_margin=0.20,
        show_corridor=True,
        approach_distance=2.0,
    ),
]


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
    hud_projector = HUDProjector(camera_name="fpv", width=WIDTH, height=HEIGHT)

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
            "flight_phase": "CORRIDOR_APPROACH",
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
            {
                "field": "flight_phase",
                "type": "in_set",
                "values": ["CORRIDOR_APPROACH", "HOVER_HOLD"],
                "desc": "Corridor approach flight phase toward forward target",
            },
        ],
    })

    # ==========================================================================
    # SD-02: Delta: Navigate to Raised Disc Platform (Pedestal Approach)
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
        "name": "Delta: Navigate to Raised Disc Platform",
        "category": "Elevated Target Acquisition",
        "image_path": str(p2),
        "instruction": "Navigate forward onto Elevated Station Delta and command hover waypoint directly over the platform disc (clearance 0.35m above platform top).",
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
            "flight_phase": "CORRIDOR_APPROACH",
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
            {
                "field": "flight_phase",
                "type": "in_set",
                "values": ["CORRIDOR_APPROACH", "HOVER_HOLD"],
                "desc": "Corridor approach to elevated pedestal platform",
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
            "flight_phase": "CORRIDOR_APPROACH",
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
            "flight_phase": "CORRIDOR_APPROACH",
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
            "flight_phase": "CORRIDOR_APPROACH",
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
            "flight_phase": "HOVER_HOLD",
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
            "flight_phase": "HOVER_HOLD",
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

    # ==========================================================================
    # SD-08: Alpha: Precision Hover Between 2 Runway Poles
    # Drone at (3.5, 0.0, 1.0), heading 000°, centered between pylons at y=±0.9m
    # ==========================================================================
    data.qpos[0:3] = [3.5, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img8 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: 000° | STATION-KEEPING: RUNWAY POLES",
        bottom_text="BETWEEN RUNWAY POLES (Y=±0.9m) | HOLD HOVER CENTERED | DY=0.0m",
        reticle_color=(0, 255, 200),
    )
    p8 = OUTPUT_DIR / "sd_08_hover_between_runway_poles.jpg"
    img8.save(p8, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-08",
        "name": "Alpha: Precision Hover Between 2 Runway Poles",
        "category": "Runway Corridor Station-Keeping",
        "image_path": str(p8),
        "instruction": "Hold precision station-keeping hover centered between the two runway poles (pylons at Y=±0.9m) framing Pad Alpha. Maintain 1.0m altitude, keep zero lateral drift (stay equidistant between poles at dy=0.0m), and hold station.",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.01, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.09, 0.06],
        },
        "ground_truth": {
            "diagnosis": "Holding precision station-keeping hover centered between the runway poles framing Pad Alpha",
            "flight_phase": "HOVER_HOLD",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.5,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 20.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.4,
                "desc": "Maintains station-keeping hover hold between runway poles",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Strict lateral centering equidistant between both runway poles",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Maintains 1.0m hover altitude",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward force during calm station-keeping",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
            {
                "field": "flight_phase",
                "type": "in_set",
                "values": ["HOVER_HOLD", "CORRIDOR_APPROACH"],
                "desc": "Station-keeping hover hold between poles",
            },
        ],
    })

    # ==========================================================================
    # SD-09: Delta: Precision Station-Keeping Atop Raised Disc Platform
    # Drone at (8.0, 0.0, 1.10), heading 000°, positioned right over yellow disc
    # ==========================================================================
    data.qpos[0:3] = [8.0, 0.0, 1.10]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img9 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.10m | HDG: 000° | STATION-KEEPING: DELTA DISC",
        bottom_text="ELEVATED PEDESTAL DELTA | TOP Z: 0.70m | CLEARANCE: 0.40m | HOLD HOVER",
        reticle_color=(255, 230, 50),
    )
    p9 = OUTPUT_DIR / "sd_09_hover_atop_raised_disc_platform.jpg"
    img9.save(p9, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-09",
        "name": "Delta: Precision Station-Keeping Atop Raised Disc Platform",
        "category": "Elevated Platform Station-Keeping",
        "image_path": str(p9),
        "instruction": "Maintain precision station-keeping hover directly atop Elevated Pedestal Delta (centered over the yellow caution disc). Hold 1.10m altitude (0.40m clearance above platform disc) and zero horizontal drift.",
        "telemetry": {
            "altitude_m": 1.10,
            "velocity_body_mps": [0.00, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.07, 0.08, 0.06],
        },
        "ground_truth": {
            "diagnosis": "Holding precision station-keeping hover directly atop Elevated Pedestal Delta yellow disc",
            "flight_phase": "HOVER_HOLD",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.5,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 18.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.4,
                "desc": "Maintains station hover over Elevated Pedestal Delta disc",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.3,
                "desc": "Centered laterally over pedestal disc",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Maintains elevated hover clearance above platform disc",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward during calm elevated hover",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
            {
                "field": "flight_phase",
                "type": "in_set",
                "values": ["HOVER_HOLD", "CORRIDOR_APPROACH"],
                "desc": "Hover hold flight phase atop elevated pedestal disc",
            },
        ],
    })

    # ==========================================================================
    # SD-10: Left Window Sector Scan (Unobservable Sector Saccade +90°)
    # Drone at (0.0, 0.0, 1.0), heading 000°, window is at +90° flank
    # ==========================================================================
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img10 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: 000° | MISSION: LEFT WINDOW INGRESS",
        bottom_text="TARGET IN SECTOR_LEFT (+90°) UNOBSERVABLE | EXECUTE ACTIVE IN-PLACE YAW SACCADE",
        reticle_color=(0, 220, 255),
    )
    p10 = OUTPUT_DIR / "sd_10_left_window_sector_scan.jpg"
    img10.save(p10, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-10",
        "name": "Left Window Sector Scan (Unobservable Sector Saccade +90°)",
        "category": "Active Perception Saccade",
        "image_path": str(p10),
        "instruction": "Objective is Left Window located at Y=+5.0m in Sector Left. Current heading is 000° (window unobservable). Perform an active perception in-place yaw saccade to face Sector Left (+90°).",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.00, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.08, 0.06],
        },
        "ground_truth": {
            "diagnosis": "Left Window in unobservable Sector Left (+90°); executing active in-place yaw saccade",
            "flight_phase": "SECTOR_SCAN",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 90.0,
                "max_velocity": 0.5,
                "corridor_width": 0.6,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 20.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.yaw_deg",
                "type": "range",
                "min": 75.0,
                "max": 95.0,
                "desc": "Commands ~+90° yaw rotation to acquire Left Window",
            },
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Rotates in-place with zero forward translation",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Rotates in-place with zero lateral translation",
            },
            {
                "field": "flight_phase",
                "type": "exact",
                "value": "SECTOR_SCAN",
                "desc": "Sector scan active perception phase",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward during unforced yaw turn",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
        ],
    })

    # ==========================================================================
    # SD-11: Left Window Aperture Ingress (Traversing Window at Z=1.10m)
    # Drone at (0.0, 3.2, 1.10), yaw = +90° looking directly at window opening at y=5.0m
    # ==========================================================================
    yaw11 = np.radians(90)
    data.qpos[0:3] = [0.0, 3.2, 1.10]
    data.qpos[3:7] = [np.cos(yaw11 / 2), 0.0, 0.0, np.sin(yaw11 / 2)]
    mujoco.mj_forward(model, data)
    cam11 = hud_projector.get_camera_extrinsics(model, data)
    renderer.update_scene(data, camera="fpv")
    base_img11 = Image.fromarray(renderer.render())
    img11 = hud_projector.overlay_hud(
        base_img11,
        cam11,
        drone_pos=np.array([0.0, 3.2, 1.10]),
        apertures=[ARENA_APERTURES[0]],
        mode_str="APERTURE INGRESS: LEFT WINDOW",
    )
    p11 = OUTPUT_DIR / "sd_11_left_window_aperture_ingress.jpg"
    img11.save(p11, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-11",
        "name": "Left Window Aperture Ingress",
        "category": "Aperture Traversal",
        "image_path": str(p11),
        "instruction": "Traverse forward through the Left Window aperture opening at Z=1.10m. Maintain tight attitude safety tilt envelope (max 12.0°) and approach velocity (0.6 m/s) to pass safely within the inner clearance frame.",
        "telemetry": {
            "altitude_m": 1.10,
            "velocity_body_mps": [0.04, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.09, 0.07],
        },
        "ground_truth": {
            "diagnosis": "Aligned with Left Window aperture center; traversing through opening with tightened 12° tilt clamp",
            "flight_phase": "APERTURE_INGRESS",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [1.8, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.6,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 12.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.2,
                "max": 2.5,
                "desc": "Advances forward through window opening",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Remains centered within horizontal window clearance",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Maintains 1.10m aperture center elevation",
            },
            {
                "field": "safety_envelope.max_tilt_deg",
                "type": "range",
                "min": 10.0,
                "max": 14.0,
                "desc": "Tightens safety tilt envelope to ~12° for aperture passage",
            },
            {
                "field": "flight_phase",
                "type": "exact",
                "value": "APERTURE_INGRESS",
                "desc": "Aperture ingress traversal flight phase",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward during aperture flight",
            },
        ],
    })

    # ==========================================================================
    # SD-12: Right Arch Sector Scan (Unobservable Sector Saccade -90°)
    # Drone at (0.0, 0.0, 1.0), heading 000°, arch is at -90° flank
    # ==========================================================================
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img12 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.00m | HDG: 000° | MISSION: RIGHT ARCH INGRESS",
        bottom_text="TARGET IN SECTOR_RIGHT (-90°) UNOBSERVABLE | EXECUTE ACTIVE IN-PLACE YAW SACCADE",
        reticle_color=(255, 140, 0),
    )
    p12 = OUTPUT_DIR / "sd_12_right_arch_sector_scan.jpg"
    img12.save(p12, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-12",
        "name": "Right Arch Sector Scan (Unobservable Sector Saccade -90°)",
        "category": "Active Perception Saccade",
        "image_path": str(p12),
        "instruction": "Objective is Right Arch located at Y=-5.0m in Sector Right. Current heading is 000° (arch unobservable). Perform an active perception in-place yaw saccade to face Sector Right (-90°).",
        "telemetry": {
            "altitude_m": 1.00,
            "velocity_body_mps": [0.00, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.08, 0.06],
        },
        "ground_truth": {
            "diagnosis": "Right Arch in unobservable Sector Right (-90°); executing active in-place yaw saccade",
            "flight_phase": "SECTOR_SCAN",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": -90.0,
                "max_velocity": 0.5,
                "corridor_width": 0.6,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 20.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.yaw_deg",
                "type": "range",
                "min": -95.0,
                "max": -75.0,
                "desc": "Commands ~-90° yaw rotation to acquire Right Arch",
            },
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Rotates in-place with zero forward translation",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Rotates in-place with zero lateral translation",
            },
            {
                "field": "flight_phase",
                "type": "exact",
                "value": "SECTOR_SCAN",
                "desc": "Sector scan active perception phase",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward during unforced yaw turn",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims",
            },
        ],
    })

    # ==========================================================================
    # SD-13: Right Arch Aperture Ingress (Traversing Arch at Z=0.90m)
    # Drone at (0.0, -3.2, 0.90), yaw = -90° looking directly at arch opening at y=-5.0m
    # ==========================================================================
    yaw13 = np.radians(-90)
    data.qpos[0:3] = [0.0, -3.2, 0.90]
    data.qpos[3:7] = [np.cos(yaw13 / 2), 0.0, 0.0, np.sin(yaw13 / 2)]
    mujoco.mj_forward(model, data)
    cam13 = hud_projector.get_camera_extrinsics(model, data)
    renderer.update_scene(data, camera="fpv")
    base_img13 = Image.fromarray(renderer.render())
    img13 = hud_projector.overlay_hud(
        base_img13,
        cam13,
        drone_pos=np.array([0.0, -3.2, 0.90]),
        apertures=[ARENA_APERTURES[1]],
        mode_str="APERTURE INGRESS: RIGHT ARCH",
    )
    p13 = OUTPUT_DIR / "sd_13_right_arch_aperture_ingress.jpg"
    img13.save(p13, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-13",
        "name": "Right Arch Aperture Ingress",
        "category": "Aperture Traversal",
        "image_path": str(p13),
        "instruction": "Traverse forward through the Right Arch aperture opening at Z=0.90m. Maintain tight attitude safety tilt envelope (max 12.0°) and approach velocity (0.6 m/s) to pass safely within the inner clearance frame.",
        "telemetry": {
            "altitude_m": 0.90,
            "velocity_body_mps": [0.04, 0.00, 0.00],
            "attitude_deg": {"roll": 0.0, "pitch": 0.0, "yaw": -90.0},
            "unmodeled_force_disturbance_N": [0.0, 0.0, 0.0],
            "motor_controls_N": [3.25, 3.25, 3.25, 3.25],
            "gyro_rms_radps": [0.08, 0.09, 0.07],
        },
        "ground_truth": {
            "diagnosis": "Aligned with Right Arch aperture center; traversing through opening with tightened 12° tilt clamp",
            "flight_phase": "APERTURE_INGRESS",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [1.8, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.6,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 12.0},
        },
        "validation_rules": [
            {
                "field": "waypoint_command.target[0]",
                "type": "range",
                "min": 1.2,
                "max": 2.5,
                "desc": "Advances forward through arch opening",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Remains centered within arch lateral opening",
            },
            {
                "field": "waypoint_command.target[2]",
                "type": "abs_max",
                "max": 0.25,
                "desc": "Maintains 0.90m arch center elevation",
            },
            {
                "field": "safety_envelope.max_tilt_deg",
                "type": "range",
                "min": 10.0,
                "max": 14.0,
                "desc": "Tightens safety tilt envelope to ~12° for arch passage",
            },
            {
                "field": "flight_phase",
                "type": "exact",
                "value": "APERTURE_INGRESS",
                "desc": "Aperture ingress traversal flight phase",
            },
            {
                "field": "feedforward_force_offset",
                "type": "zero_vector",
                "desc": "Zero feedforward during aperture flight",
            },
        ],
    })

    # ==========================================================================
    # SD-14: Delta: Platform Station-Keeping under Crosswind
    # Drone at (8.0, 0.0, 1.10) atop Delta disc, +4.8N crosswind along +Y
    # Roll tilt ~ -12 deg banked into crosswind
    # ==========================================================================
    roll14 = np.radians(-12)
    data.qpos[0:3] = [8.0, 0.0, 1.10]
    data.qpos[3:7] = [np.cos(roll14 / 2), np.sin(roll14 / 2), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="fpv")
    img14 = add_hud(
        Image.fromarray(renderer.render()),
        top_text="FPV: fpv | ALT: 1.10m | ROLL: -12.0° | CROSSWIND DISTURBANCE",
        bottom_text="CROSSWIND +4.8N (+Y) | COUNTER-THRUST FEEDFORWARD: dFy = -4.8N",
        reticle_color=(255, 140, 0),
    )
    p14 = OUTPUT_DIR / "sd_14_platform_delta_crosswind_hover.jpg"
    img14.save(p14, format="JPEG", quality=85)

    test_cases.append({
        "id": "SD-14",
        "name": "Delta: Platform Station-Keeping under Crosswind",
        "category": "Disturbance Rejection",
        "image_path": str(p14),
        "instruction": "Maintain station-keeping hover atop Elevated Station Delta against a strong lateral crosswind disturbance (+4.8N along +Y). Inject opposing feedforward force to prevent drifting off the raised platform.",
        "telemetry": {
            "altitude_m": 1.10,
            "velocity_body_mps": [0.00, 0.02, 0.00],
            "attitude_deg": {"roll": -12.0, "pitch": 0.0, "yaw": 0.0},
            "unmodeled_force_disturbance_N": [0.00, 4.80, 0.00],
            "motor_controls_N": [3.60, 3.60, 2.90, 2.90],
            "gyro_rms_radps": [0.16, 0.14, 0.10],
        },
        "ground_truth": {
            "diagnosis": "Strong lateral crosswind disturbance (+4.8N along +Y) atop Elevated Pedestal Delta; injecting opposing feedforward",
            "flight_phase": "HOVER_HOLD",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [0.0, 0.0, 0.0],
                "yaw_deg": 0.0,
                "max_velocity": 0.5,
                "corridor_width": 0.4,
            },
            "feedforward_force_offset": [0.0, -4.8, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "gain_multipliers": {"pos_kp": 1.2, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
            "safety_envelope": {"max_tilt_deg": 25.0},
        },
        "validation_rules": [
            {
                "field": "feedforward_force_offset[1]",
                "type": "range",
                "min": -5.5,
                "max": -4.2,
                "desc": "Injects opposing lateral feedforward force dFy in [-5.5, -4.2] N to cancel +4.8N crosswind",
            },
            {
                "field": "feedforward_force_offset[0]",
                "type": "abs_max",
                "max": 0.3,
                "desc": "Zero forward feedforward force",
            },
            {
                "field": "waypoint_command.target[0]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Holds station hover rather than forward drift",
            },
            {
                "field": "waypoint_command.target[1]",
                "type": "abs_max",
                "max": 0.5,
                "desc": "Holds lateral station against crosswind",
            },
            {
                "field": "motor_trim_scale",
                "type": "ones_vector",
                "desc": "Nominal motor trims during wind disturbance",
            },
            {
                "field": "flight_phase",
                "type": "in_set",
                "values": ["HOVER_HOLD", "CORRIDOR_APPROACH"],
                "desc": "Hover hold flight phase under crosswind",
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
