"""Unit and integration tests for System 2 Natural Language Intent Parsing & Action Mapping."""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from runtime.brain.intent_parser import NaturalLanguageIntentParser, ARENA_LANDMARKS
from runtime.server.navigation_server import create_navigation_app
from runtime.server.navigation_state import nav_coordinator


class TestNaturalLanguageIntentParser(unittest.TestCase):
    """Verifies LLM intent parsing and rule-based heuristic fallback."""

    def setUp(self):
        self.parser = NaturalLanguageIntentParser(preferred_provider="groq")

    def test_canonical_user_prompts(self):
        """Verifies mapping of all user-requested natural language prompts."""
        test_cases = [
            ("Find two poles and hover between them", "poles"),
            ("Find the window and go through the window", "window"),
            ("locate the arch and go through it", "arch"),
            ("Go to the orange pad", "alpha"),
            ("Go to the elevated yellow pad", "delta"),
            ("Return to launch position", "home"),
            ("Inspect all arena landmarks in sequence", "grand_tour"),
        ]

        for prompt, expected_action in test_cases:
            intent = self.parser.parse_instruction(prompt)
            self.assertEqual(
                intent.target_action,
                expected_action,
                f"Prompt '{prompt}' failed to map to '{expected_action}', got '{intent.target_action}'",
            )
            self.assertGreaterEqual(intent.confidence, 0.80)
            self.assertIn(intent.target_action, ARENA_LANDMARKS)
            self.assertTrue(len(intent.semantic_label) > 0)
            self.assertTrue(len(intent.visual_features) > 0)
            self.assertTrue(len(intent.intent_reasoning) > 0)

    def test_rule_based_fallback_mode(self):
        """Verifies 100% resilient offline operation without any LLM network call."""
        intent = self.parser.parse_instruction(
            "locate the arch and go through it",
            use_llm=False,
        )
        self.assertEqual(intent.target_action, "arch")
        self.assertEqual(intent.source, "heuristic_fallback")
        self.assertEqual(intent.semantic_label, "Arched Doorway Gate")
        self.assertGreaterEqual(intent.confidence, 0.90)

    def test_llm_exception_fallback(self):
        """Verifies that an unexpected LLM network error gracefully falls back to local heuristic."""
        with patch.object(self.parser, "_parse_with_groq", side_effect=RuntimeError("Groq offline")):
            with patch.object(self.parser, "_parse_with_gemini", side_effect=RuntimeError("Gemini offline")):
                intent = self.parser.parse_instruction("Go to the elevated yellow pad", use_llm=True)
                self.assertEqual(intent.target_action, "delta")
                self.assertEqual(intent.source, "heuristic_fallback")
                self.assertEqual(intent.semantic_label, "Elevated Yellow Platform")


class TestNavigationInstructionAPI(unittest.TestCase):
    """Verifies FastAPI REST API endpoint and state coordination."""

    def setUp(self):
        self.app = create_navigation_app()
        self.client = TestClient(self.app)

    def test_instruction_endpoint(self):
        """Verifies POST /api/navigation/instruction maps intent and dispatches flight."""
        res = self.client.post(
            "/api/navigation/instruction",
            json={"instruction": "Find the window and go through the window", "use_llm": False},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "executed")
        self.assertEqual(data["target_action"], "window")
        self.assertEqual(data["semantic_label"], "Wall Window Aperture")
        self.assertIn("visual_features", data)
        self.assertIn("intent_reasoning", data)

        # Verify telemetry snapshot includes semantic target and instruction
        snap = nav_coordinator.get_telemetry_snapshot()
        self.assertEqual(snap["semantic_target"], "Wall Window Aperture")
        self.assertEqual(snap["active_instruction"], "Find the window and go through the window")

        # Verify event was recorded in sliding event feed
        event_types = [e["type"] for e in snap["events"]]
        self.assertIn("SYSTEM2_INTENT", event_types)


if __name__ == "__main__":
    unittest.main()
