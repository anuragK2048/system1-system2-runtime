"""Unit Tests for System 2 LLMLogger and Audit Tracking."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from runtime.logging.llm_logger import LLMLogger, LLMCallRecord


class TestLLMLogger(unittest.TestCase):
    """Verifies file generation, token tracking, and structured serialization of LLM calls."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.logger = LLMLogger(log_dir=self.log_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_successful_call(self):
        """Verify successful LLM call is properly written to jsonl and text logs."""
        record = self.logger.log_call(
            duration_s=2.45,
            model="meta-models/muse-spark-open-1026",
            system_prompt="You are System 2 flight reasoner.",
            user_prompt="TELEMETRY SUMMARY:\n- Position Error: dx=0.5m",
            raw_output='{"diagnosis": "lateral wind", "feedforward_force_offset": [-5.0, 0, 0]}',
            prompt_tokens=180,
            completion_tokens=45,
            total_tokens=225,
            reasoning_tokens=12,
            reasoning_content="Reasoning about lateral wind drift...",
            parsed_patch={"diagnosis": "lateral wind", "feedforward_force_offset": [-5.0, 0, 0]},
            status="success",
            metadata={"scenario": "wind_gust"},
        )

        self.assertTrue(record.call_id.startswith("call_"))
        self.assertEqual(record.status, "success")
        self.assertEqual(record.prompt_tokens, 180)
        self.assertEqual(record.completion_tokens, 45)
        self.assertEqual(record.total_tokens, 225)
        self.assertEqual(record.duration_s, 2.45)

        # Check JSONL file content
        self.assertTrue(self.logger.jsonl_path.exists())
        with open(self.logger.jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 1)
        logged_json = lines[0]
        self.assertEqual(logged_json["call_id"], record.call_id)
        self.assertEqual(logged_json["prompt_tokens"], 180)
        self.assertEqual(logged_json["completion_tokens"], 45)
        self.assertEqual(logged_json["parsed_patch"]["diagnosis"], "lateral wind")

        # Check Text file content
        self.assertTrue(self.logger.text_path.exists())
        with open(self.logger.text_path, "r", encoding="utf-8") as f:
            text_content = f.read()

        self.assertIn("LLM CALL RECORD", text_content)
        self.assertIn("Prompt=180", text_content)
        self.assertIn("Completion=45", text_content)
        self.assertIn("lateral wind", text_content)

    def test_log_error_call(self):
        """Verify failed LLM calls record the error message and status."""
        record = self.logger.log_call(
            duration_s=0.85,
            model="meta-models/muse-spark-open-1026",
            system_prompt="System prompt",
            user_prompt="User prompt",
            raw_output=None,
            prompt_tokens=150,
            completion_tokens=0,
            total_tokens=150,
            status="error",
            error_message="API connection timeout",
        )

        self.assertEqual(record.status, "error")
        self.assertEqual(record.error_message, "API connection timeout")

        # Verify summary statistics
        stats = self.logger.get_summary_statistics()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["successful_calls"], 0)
        self.assertEqual(stats["failed_calls"], 1)
        self.assertEqual(stats["total_prompt_tokens"], 150)
        self.assertEqual(stats["total_completion_tokens"], 0)

    def test_summary_statistics_aggregation(self):
        """Verify token and latency aggregation over multiple calls."""
        for i in range(5):
            self.logger.log_call(
                duration_s=1.0 + i * 0.5,
                model="test-model",
                system_prompt="sys",
                user_prompt=f"user_{i}",
                raw_output="{}",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )

        stats = self.logger.get_summary_statistics()
        self.assertEqual(stats["total_calls"], 5)
        self.assertEqual(stats["successful_calls"], 5)
        self.assertEqual(stats["total_prompt_tokens"], 500)
        self.assertEqual(stats["total_completion_tokens"], 250)
        self.assertEqual(stats["total_tokens"], 750)
        self.assertAlmostEqual(stats["avg_duration_s"], 2.0, delta=0.01)

        recent = self.logger.get_recent_calls(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].user_prompt, "user_4")


if __name__ == "__main__":
    unittest.main()
