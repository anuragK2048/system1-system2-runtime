"""Unit and Integration Tests for VLM Single-Drone Target Navigation & Evaluation.

Tests cover:
1. WaypointCommand schema validation, clamping, and serialization within System2Patch.
2. Ground truth test suite integrity and validation rules across all 7 scenarios (SD-01 to SD-07).
3. System-1 controller setpoint update and frame transformations under waypoint_command.
4. Evaluation metric computation (Euclidean target error, disturbance rejection, actuator trims, safety bounds).

Runnable via:
    uv run python -m unittest tests/test_vlm_target_navigation.py
"""

import json
import math
import unittest
from pathlib import Path
import numpy as np

from runtime.schemas.tier2 import (
    System2Patch,
    WaypointCommand,
    GainMultipliers,
    SafetyEnvelope,
    validate_patch,
    WAYPOINT_MIN_VELOCITY,
    WAYPOINT_MAX_VELOCITY,
    WAYPOINT_MIN_CORRIDOR,
    WAYPOINT_MAX_CORRIDOR,
)
from runtime.sim.system1_controller import System1Controller
from vlm.evaluate_target_navigation import evaluate_rules, compute_metrics


class TestWaypointCommandSchema(unittest.TestCase):
    """Tests for WaypointCommand dataclass, serialization, and validation."""

    def test_default_values(self):
        wc = WaypointCommand()
        self.assertEqual(wc.mode, "relative_body")
        self.assertEqual(wc.target, [0.0, 0.0, 0.0])
        self.assertEqual(wc.yaw_deg, 0.0)
        self.assertEqual(wc.max_velocity, 1.0)
        self.assertEqual(wc.corridor_width, 0.6)

    def test_serialization_and_roundtrip(self):
        patch_dict = {
            "diagnosis": "Navigate to Bravo",
            "waypoint_command": {
                "mode": "relative_body",
                "target": [2.2, 0.8, -0.1],
                "yaw_deg": 15.0,
                "max_velocity": 1.2,
                "corridor_width": 0.5,
            },
            "feedforward_force_offset": [0.0, 0.0, 0.0],
            "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
            "confidence": 0.95,
        }
        patch = System2Patch.from_dict(patch_dict)
        self.assertIsNotNone(patch.waypoint_command)
        self.assertEqual(patch.waypoint_command.mode, "relative_body")
        self.assertEqual(patch.waypoint_command.target, [2.2, 0.8, -0.1])
        self.assertAlmostEqual(patch.waypoint_command.yaw_deg, 15.0)
        self.assertAlmostEqual(patch.waypoint_command.max_velocity, 1.2)
        self.assertAlmostEqual(patch.waypoint_command.corridor_width, 0.5)

        # Check to_dict serialization
        dumped = patch.to_dict()
        self.assertIn("waypoint_command", dumped)
        self.assertEqual(dumped["waypoint_command"]["target"], [2.2, 0.8, -0.1])
        self.assertEqual(dumped["waypoint_command"]["mode"], "relative_body")

    def test_invalid_mode_rejected(self):
        patch = System2Patch(
            diagnosis="Bad Mode",
            waypoint_command=WaypointCommand(mode="orbit_target", target=[1.0, 0.0, 0.0]),
        )
        with self.assertRaises(ValueError) as ctx:
            validate_patch(patch)
        self.assertIn("'waypoint_command.mode'", str(ctx.exception))

    def test_invalid_target_length_rejected(self):
        patch = System2Patch(
            diagnosis="Short Target",
            waypoint_command=WaypointCommand(target=[1.0, 0.0]),
        )
        with self.assertRaises(ValueError) as ctx:
            validate_patch(patch)
        self.assertIn("'waypoint_command.target' must have 3 elements", str(ctx.exception))

    def test_invalid_target_types_rejected(self):
        patch = System2Patch(
            diagnosis="Non-numeric target",
            waypoint_command=WaypointCommand(target=["north", 0.0, 0.0]),
        )
        with self.assertRaises(ValueError):
            validate_patch(patch)

    def test_velocity_clamping(self):
        # Over maximum velocity
        patch_high = System2Patch(
            diagnosis="Too fast",
            waypoint_command=WaypointCommand(max_velocity=12.0),
        )
        validated_high = validate_patch(patch_high)
        self.assertAlmostEqual(validated_high.waypoint_command.max_velocity, WAYPOINT_MAX_VELOCITY)

        # Under minimum velocity
        patch_low = System2Patch(
            diagnosis="Too slow",
            waypoint_command=WaypointCommand(max_velocity=0.01),
        )
        validated_low = validate_patch(patch_low)
        self.assertAlmostEqual(validated_low.waypoint_command.max_velocity, WAYPOINT_MIN_VELOCITY)

    def test_corridor_width_clamping(self):
        patch_wide = System2Patch(
            diagnosis="Too wide",
            waypoint_command=WaypointCommand(corridor_width=8.0),
        )
        validated_wide = validate_patch(patch_wide)
        self.assertAlmostEqual(validated_wide.waypoint_command.corridor_width, WAYPOINT_MAX_CORRIDOR)

        patch_narrow = System2Patch(
            diagnosis="Too narrow",
            waypoint_command=WaypointCommand(corridor_width=0.02),
        )
        validated_narrow = validate_patch(patch_narrow)
        self.assertAlmostEqual(validated_narrow.waypoint_command.corridor_width, WAYPOINT_MIN_CORRIDOR)


class TestSingleDroneTestSuiteGroundTruth(unittest.TestCase):
    """Tests that all 7 test cases in vlm/test_cases_single.json pass ground truth validation."""

    SUITE_PATH = Path("vlm/test_cases_single.json")

    def setUp(self):
        if not self.SUITE_PATH.exists():
            from vlm.generate_single_drone_suite import generate_suite
            generate_suite()

        with open(self.SUITE_PATH) as f:
            self.test_cases = json.load(f)

    def test_suite_contains_all_seven_scenarios(self):
        expected_ids = {"SD-01", "SD-02", "SD-03", "SD-04", "SD-05", "SD-06", "SD-07"}
        actual_ids = {tc["id"] for tc in self.test_cases}
        self.assertEqual(expected_ids, actual_ids, f"Suite missing expected scenario IDs: {expected_ids - actual_ids}")

    def test_rendered_images_exist_and_nonempty(self):
        for tc in self.test_cases:
            img_path = Path(tc["image_path"])
            self.assertTrue(img_path.exists(), f"Rendered image missing for {tc['id']}: {img_path}")
            self.assertGreater(img_path.stat().st_size, 5000, f"Image file too small or empty: {img_path}")

    def test_ground_truth_answers_pass_validation_rules(self):
        for tc in self.test_cases:
            tc_id = tc["id"]
            gt = tc["ground_truth"]
            rules = tc["validation_rules"]

            is_valid, reasons = evaluate_rules(gt, rules)
            self.assertTrue(
                is_valid,
                f"Ground truth for {tc_id} ({tc['name']}) failed validation rules: {reasons}",
            )


class TestSystem1WaypointTracking(unittest.TestCase):
    """Tests System-1 controller response to System-2 WaypointCommands."""

    def setUp(self):
        self.controller = System1Controller(mode="l1_adaptive", target_pos=np.array([0.0, 0.0, 1.0]))

    def test_world_mode_setpoint_update(self):
        patch = System2Patch(
            diagnosis="Command World Waypoint",
            waypoint_command=WaypointCommand(
                mode="world",
                target=[5.5, 0.0, 1.2],
            ),
        )
        self.controller.bridge.apply_system2_patch(patch)

        # Resolve active patch
        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.1,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(resolved_target, np.array([5.5, 0.0, 1.2], dtype=np.float32), atol=1e-5)

    def test_relative_body_mode_zero_yaw(self):
        patch = System2Patch(
            diagnosis="Advance Forward 2m",
            waypoint_command=WaypointCommand(
                mode="relative_body",
                target=[2.0, 0.0, 0.0],
            ),
        )
        self.controller.bridge.apply_system2_patch(patch)

        # Drone heading +X (yaw = 0): quat = [1, 0, 0, 0]
        quat_zero = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.1,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=base_pos,
            quat=quat_zero,
        )
        # World target should advance by +2.0m along X
        np.testing.assert_allclose(resolved_target, np.array([3.0, 0.0, 1.0], dtype=np.float32), atol=1e-5)

    def test_relative_body_mode_rotated_yaw(self):
        patch = System2Patch(
            diagnosis="Advance Forward in Turned Body Frame",
            waypoint_command=WaypointCommand(
                mode="relative_body",
                target=[2.0, 0.0, 0.0],
            ),
        )
        self.controller.bridge.apply_system2_patch(patch)

        # Drone yaw = 90 deg (+Y direction): quat = [cos(45°), 0, 0, sin(45°)]
        angle = math.pi / 2.0
        quat_90 = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float32)
        base_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        _, _, _, _, _, _, resolved_target, _ = self.controller._resolve_active_patch(
            sim_time=0.1,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=base_pos,
            quat=quat_90,
        )
        # Forward in body frame (dx=2.0) with yaw=90° should become dy=+2.0 in world frame
        np.testing.assert_allclose(resolved_target, np.array([0.0, 2.0, 1.0], dtype=np.float32), atol=1e-4)

    def test_combined_waypoint_and_safety_envelope(self):
        patch = System2Patch(
            diagnosis="Waypoint with Safety Offset",
            waypoint_command=WaypointCommand(
                mode="world",
                target=[4.0, 1.0, 1.0],
            ),
            safety_envelope=SafetyEnvelope(
                max_tilt_deg=18.0,
                temp_setpoint_offset=[0.0, 0.0, 0.2],
            ),
        )
        self.controller.bridge.apply_system2_patch(patch)

        _, _, _, _, _, _, resolved_target, max_tilt = self.controller._resolve_active_patch(
            sim_time=0.1,
            explicit_ff=None,
            explicit_trim=None,
            target_pos=np.array([0.0, 0.0, 1.0]),
        )
        # Target should be world target [4.0, 1.0, 1.0] + safety offset [0, 0, 0.2] = [4.0, 1.0, 1.2]
        np.testing.assert_allclose(resolved_target, np.array([4.0, 1.0, 1.2], dtype=np.float32), atol=1e-4)
        self.assertAlmostEqual(max_tilt, 18.0)

    def test_compute_motor_thrusts_execution(self):
        patch = System2Patch(
            diagnosis="Nominal Flight",
            waypoint_command=WaypointCommand(target=[2.0, 0.0, 0.0]),
            feedforward_force_offset=[0.0, 0.0, 0.0],
            motor_trim_scale=[1.0, 1.0, 1.0, 1.0],
        )
        self.controller.bridge.apply_system2_patch(patch)

        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        omega = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        thrusts, diag = self.controller.compute_motor_thrusts(
            sim_time=0.01,
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
        )
        self.assertEqual(len(thrusts), 4)
        for t in thrusts:
            self.assertGreaterEqual(t, self.controller.drone_cfg.min_thrust)
            self.assertLessEqual(t, self.controller.drone_cfg.max_thrust)


class TestEvaluationMetricsComputation(unittest.TestCase):
    """Tests evaluation metrics: Euclidean error, disturbance rejection, and actuator trims."""

    def test_target_euclidean_error_perfect_match(self):
        parsed = {"waypoint_command": {"target": [2.0, 0.0, 0.0]}}
        gt = {"waypoint_command": {"target": [2.0, 0.0, 0.0]}}
        metrics = compute_metrics(parsed, gt)
        self.assertAlmostEqual(metrics["target_error_euclidean_m"], 0.0)

    def test_target_euclidean_error_offset(self):
        # 3-4-0 Pythagorean triangle -> Euclidean error is 5.0
        parsed = {"waypoint_command": {"target": [5.0, 4.0, 1.0]}}
        gt = {"waypoint_command": {"target": [2.0, 0.0, 1.0]}}
        metrics = compute_metrics(parsed, gt)
        self.assertAlmostEqual(metrics["target_error_euclidean_m"], 5.0)
        self.assertAlmostEqual(metrics["target_dx_err_m"], 3.0)
        self.assertAlmostEqual(metrics["target_dy_err_m"], 4.0)
        self.assertAlmostEqual(metrics["target_dz_err_m"], 0.0)

    def test_disturbance_rejection_feedforward_metric(self):
        parsed = {"feedforward_force_offset": [-5.2, 0.0, 0.0]}
        gt = {"feedforward_force_offset": [-5.2, 0.0, 0.0]}
        metrics = compute_metrics(parsed, gt)
        self.assertAlmostEqual(metrics["feedforward_error_N"], 0.0)

        parsed_imperfect = {"feedforward_force_offset": [-4.7, 0.2, 0.0]}
        metrics_imperfect = compute_metrics(parsed_imperfect, gt)
        # error: dx = 0.5, dy = 0.2 -> norm = sqrt(0.25 + 0.04) = sqrt(0.29) ≈ 0.5385
        self.assertAlmostEqual(metrics_imperfect["feedforward_error_N"], math.sqrt(0.29), places=4)

    def test_actuator_trim_error_metric(self):
        parsed = {"motor_trim_scale": [2.5, 1.0, 1.0, 1.0]}
        gt = {"motor_trim_scale": [2.5, 1.0, 1.0, 1.0]}
        metrics = compute_metrics(parsed, gt)
        self.assertAlmostEqual(metrics["trim_error_norm"], 0.0)

        parsed_partial = {"motor_trim_scale": [2.0, 1.0, 1.0, 1.0]}
        metrics_partial = compute_metrics(parsed_partial, gt)
        self.assertAlmostEqual(metrics_partial["trim_error_norm"], 0.5)


if __name__ == "__main__":
    unittest.main()
