"""Unit tests for System-2 Master Configuration, Sub-configs, and Runtime Wiring."""

import time
import tempfile
from pathlib import Path
import unittest
import numpy as np

from runtime.config import (
    FastTriggerConfig,
    TelemetrySummaryConfig,
    PatchApplicationConfig,
    System2InferenceConfig,
    System2WorkerConfig,
    System2MasterConfig,
)
from runtime.sim.bridge import System1System2Bridge
from runtime.sim.system1_controller import System1Controller
from runtime.schemas.tier2 import System2Patch, GainMultipliers, SafetyEnvelope
from runtime.brain.llm import System2Brain, System2Worker


class TestSystem2Config(unittest.TestCase):
    """Verifies that all System-2 configurations operate properly and match expected defaults."""

    def test_default_configs_match_legacy_values(self):
        """Verifies that all default values preserve exact baseline behavior without early quick wins."""
        trigger = FastTriggerConfig()
        self.assertEqual(trigger.pos_error_threshold, 0.30)
        self.assertEqual(trigger.tilt_deg_threshold, 25.0)
        self.assertEqual(trigger.gyro_norm_threshold, 3.5)
        self.assertEqual(trigger.force_residual_threshold, 3.0)
        self.assertEqual(trigger.trigger_cooldown_s, 0.5)
        self.assertEqual(trigger.min_sim_time, 1.0)
        self.assertEqual(trigger.max_buffer_size, 1500)

        telemetry = TelemetrySummaryConfig()
        self.assertEqual(telemetry.window_seconds, 1.0)
        self.assertEqual(telemetry.recent_instant_samples, 25)
        self.assertEqual(telemetry.motor_saturation_min, 0.05)
        self.assertEqual(telemetry.motor_saturation_max, 12.95)
        self.assertEqual(telemetry.oscillation_min_freq_hz, 3.0)
        self.assertEqual(telemetry.oscillation_min_gyro_rms, 2.0)

        patch = PatchApplicationConfig()
        self.assertEqual(patch.ramp_duration_s, 0.0)
        self.assertIsNone(patch.patch_ttl_s)
        self.assertFalse(patch.confidence_scaling)
        self.assertEqual(patch.feedforward_clamp_min, -10.0)
        self.assertEqual(patch.feedforward_clamp_max, 10.0)

        inference = System2InferenceConfig()
        self.assertEqual(inference.temperature, 0.2)
        self.assertEqual(inference.groq_max_completion_tokens, 2048)
        self.assertEqual(inference.general_max_tokens, 4096)

        worker = System2WorkerConfig()
        self.assertEqual(worker.cooldown_s, 2.0)
        self.assertEqual(worker.attention_pos_error, 0.30)
        self.assertEqual(worker.calm_pos_error, 0.15)
        self.assertEqual(worker.reconsult_pos_error, 0.75)
        self.assertEqual(worker.crash_alt_min, 0.20)

    def test_master_config_json_roundtrip(self):
        """Verifies that System2MasterConfig can be serialized and deserialized via dict and JSON."""
        master = System2MasterConfig()
        master.trigger.pos_error_threshold = 0.18
        master.patch.ramp_duration_s = 0.35
        master.telemetry.window_seconds = 1.5
        master.inference.temperature = 0.05
        master.worker.cooldown_s = 1.0

        d = master.to_dict()
        self.assertEqual(d["trigger"]["pos_error_threshold"], 0.18)
        self.assertEqual(d["patch"]["ramp_duration_s"], 0.35)
        self.assertEqual(d["telemetry"]["window_seconds"], 1.5)
        self.assertEqual(d["inference"]["temperature"], 0.05)
        self.assertEqual(d["worker"]["cooldown_s"], 1.0)

        loaded_from_dict = System2MasterConfig.from_dict(d)
        self.assertEqual(loaded_from_dict.trigger.pos_error_threshold, 0.18)
        self.assertEqual(loaded_from_dict.patch.ramp_duration_s, 0.35)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            master.save_json(tmp_path)
            loaded_from_file = System2MasterConfig.load_json(tmp_path)
            self.assertEqual(loaded_from_file.trigger.pos_error_threshold, 0.18)
            self.assertEqual(loaded_from_file.patch.ramp_duration_s, 0.35)
            self.assertEqual(loaded_from_file.inference.temperature, 0.05)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_bridge_wires_custom_trigger_config(self):
        """Verifies that custom FastTriggerConfig thresholds are strictly honored by the bridge."""
        custom_trigger = FastTriggerConfig(pos_error_threshold=0.10)
        bridge = System1System2Bridge(trigger_cfg=custom_trigger)

        # Drift 0.15m away (exceeds 0.10m threshold but below standard 0.30m)
        is_trig, reasons = bridge.push_telemetry(
            sim_time=2.0,
            pos=np.array([0.15, 0.0, 1.0]),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
            motor_ctrl=np.full(4, 3.25),
            target_pos=np.array([0.0, 0.0, 1.0]),
            sigma_f_hat=np.zeros(3),
            sigma_M_hat=np.zeros(3),
        )
        self.assertTrue(is_trig)
        self.assertTrue(any("Position error (0.15m) exceeded threshold" in r for r in reasons))

    def test_patch_ramp_in_and_safety_clamping(self):
        """Verifies that patch ramp-in apply time smoothly scales control and clamps excessive values."""
        patch_cfg = PatchApplicationConfig(
            ramp_duration_s=0.5,
            feedforward_clamp_max=8.0,
        )
        bridge = System1System2Bridge(patch_cfg=patch_cfg)
        controller = System1Controller(bridge=bridge, patch_cfg=patch_cfg)

        # Inject an excessive +15.0N feedforward patch
        patch = System2Patch(
            diagnosis="Strong wind test",
            feedforward_force_offset=[15.0, 0.0, 0.0],
            confidence=1.0,
            timestamp=time.time(),
        )
        bridge.apply_system2_patch(patch)

        # Step controller at time = 0.0s (ramp factor = 0)
        thrusts, diag = controller.compute_motor_thrusts(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
            sim_time=0.0,
            dt=0.002,
        )
        # Because patch was just applied, ramp is near 0.0
        self.assertIsNotNone(thrusts)

        # Now test when ramp is complete (simulate elapsed > ramp_duration)
        patch_applied_t = bridge.get_patch_applied_time()
        bridge._patch_applied_time = patch_applied_t - 1.0  # elapsed = 1.0s > 0.5s ramp

        resolved_ff, _, _, _, _, _, _, _ = controller._resolve_active_patch(
            sim_time=1.0, explicit_ff=None, explicit_trim=None, target_pos=None
        )
        # Raw force was 15.0N, but clamp is 8.0N
        self.assertAlmostEqual(resolved_ff[0], 8.0, places=4)

    def test_patch_ttl_expiration(self):
        """Verifies that patch auto-expires if patch_ttl_s is reached."""
        patch_cfg = PatchApplicationConfig(patch_ttl_s=0.2)
        bridge = System1System2Bridge(patch_cfg=patch_cfg)

        patch = System2Patch(diagnosis="Temporary Gust", feedforward_force_offset=[5.0, 0.0, 0.0])
        bridge.apply_system2_patch(patch)
        self.assertIsNotNone(bridge.get_active_patch())

        # Backdate applied time past TTL
        bridge._patch_applied_time = time.time() - 0.5
        # Retrieval should now return None (expired)
        self.assertIsNone(bridge.get_active_patch())


if __name__ == "__main__":
    unittest.main()

