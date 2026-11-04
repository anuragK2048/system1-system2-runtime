"""Unit Tests for Gemini Voice Transcriber and Speech Recognition Module.

Validates:
1. Text sanitation and prompt extraction.
2. VoiceTranscriptionResult structured schemas.
3. Resilient fallback to browser preview when Gemini is unavailable.
4. Error state propagation.
5. End-to-end simulated audio transcription flow.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from runtime.brain.voice_transcriber import (
    GeminiVoiceTranscriber,
    VoiceTranscriptionResult,
    clean_flight_instruction,
)


class TestVoiceTranscriber(unittest.TestCase):
    """Test suite for voice transcription and sanitation."""

    def test_clean_flight_instruction(self):
        """Verifies text cleaner removes conversational fluff, quotes, and punctuation."""
        cases = [
            ('"Find two poles and hover between them."', "Find two poles and hover between them"),
            ('Command: Go to the orange pad.', "Go to the orange pad"),
            ('User said: "Find the window and go through the window"', "Find the window and go through the window"),
            ("```\nlocate the arch and go through it\n```", "locate the arch and go through it"),
            ("flight command: Go to the elevated yellow pad", "Go to the elevated yellow pad"),
            ("   Return to launch position.   ", "Return to launch position"),
        ]
        for raw, expected in cases:
            cleaned = clean_flight_instruction(raw)
            self.assertEqual(cleaned, expected, f"Failed cleaning: {raw}")

    def test_empty_audio_with_client_fallback(self):
        """Verifies client fallback is returned when audio buffer is empty."""
        transcriber = GeminiVoiceTranscriber(api_key="test-key")
        result = transcriber.transcribe_audio(
            audio_bytes=b"",
            mime_type="audio/webm",
            client_fallback="Find the window and go through it",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Find the window and go through it")
        self.assertEqual(result.provider, "browser_speech")
        self.assertGreater(result.confidence, 0.8)

    def test_empty_audio_without_fallback_returns_error(self):
        """Verifies clear error response when both audio and fallback are absent."""
        transcriber = GeminiVoiceTranscriber(api_key="test-key")
        result = transcriber.transcribe_audio(
            audio_bytes=b"",
            mime_type="audio/webm",
            client_fallback=None,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.text, "")
        self.assertIsNotNone(result.error)

    def test_gemini_client_mock_success(self):
        """Verifies successful Gemini API response transcription."""
        transcriber = GeminiVoiceTranscriber(api_key="mock-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '  "Go to the orange pad."  '
        mock_client.models.generate_content.return_value = mock_response
        transcriber._client = mock_client

        dummy_audio = b"\x00" * 128
        result = transcriber.transcribe_audio(
            audio_bytes=dummy_audio,
            mime_type="audio/webm",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.text, "Go to the orange pad")
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, "gemini-3.5-flash")
        self.assertGreaterEqual(result.confidence, 0.95)

    def test_gemini_failure_falls_back_to_client_preview(self):
        """Verifies resilient fallback to client preview when Gemini raises an exception."""
        transcriber = GeminiVoiceTranscriber(api_key="mock-key")
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Gemini API Network Timeout")
        transcriber._client = mock_client

        dummy_audio = b"\x00" * 128
        result = transcriber.transcribe_audio(
            audio_bytes=dummy_audio,
            mime_type="audio/webm",
            client_fallback="locate the arch and go through it",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.text, "locate the arch and go through it")
        self.assertEqual(result.provider, "browser_speech")

    def test_structured_dict_serialization(self):
        """Verifies to_dict returns serializable dict matching requirements."""
        res = VoiceTranscriptionResult(
            text="Go to the orange pad",
            raw_text="Go to the orange pad.",
            provider="gemini",
            model="gemini-3.5-flash",
            confidence=0.98,
            latency_ms=45.2,
            success=True,
        )
        d = res.to_dict()
        self.assertEqual(d["text"], "Go to the orange pad")
        self.assertEqual(d["provider"], "gemini")
        self.assertEqual(d["latency_ms"], 45.2)


if __name__ == "__main__":
    unittest.main()
