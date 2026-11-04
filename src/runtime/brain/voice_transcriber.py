"""Gemini Voice Transcriber and Speech-to-Intent Module.

Transcribes spoken natural language flight instructions from browser audio streams
into actionable text commands using Google Gemini (via google-genai SDK) with resilient
zero-downtime client-side speech recognition fallback.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class VoiceTranscriptionResult(BaseModel):
    """Structured result of a voice transcription request."""

    text: str = Field(..., description="Cleaned transcribed flight instruction text")
    raw_text: str = Field("", description="Raw response text from Gemini or fallback")
    provider: str = Field("gemini", description="Transcription provider: 'gemini' | 'browser_speech' | 'heuristic'")
    model: str = Field("gemini-3.5-flash", description="Underlying model name used")
    confidence: float = Field(0.95, ge=0.0, le=1.0, description="Transcription confidence score")
    latency_ms: float = Field(0.0, description="End-to-end transcription latency in milliseconds")
    success: bool = Field(True, description="Whether transcription succeeded")
    error: Optional[str] = Field(None, description="Error message if transcription failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "provider": self.provider,
            "model": self.model,
            "confidence": round(self.confidence, 2),
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "error": self.error,
        }


def clean_flight_instruction(raw_text: str) -> str:
    """Sanitizes raw model output into a concise natural language flight instruction."""
    if not raw_text:
        return ""
    
    text = raw_text.strip()
    # Strip markdown block formatting
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = text.replace("```", "").strip()

    # Strip surrounding quotes
    text = re.sub(r'^["\'\s]+|["\'\s]+$', "", text)

    # Strip common conversational prefixes models might accidentally output
    prefixes = [
        r"^(?:transcribed|transcription|instruction|command|flight command|user said|user says):\s*",
        r"^(?:here is the transcription|the user wants to|the command is):\s*",
        r"^(?:audio:\s*)",
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes again if present
    text = re.sub(r'^["\'\s]+|["\'\s]+$', "", text)

    # Strip final punctuation like trailing periods if single short command
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].strip()

    return text


VOICE_TRANSCRIBE_PROMPT = """You are an onboard real-time speech recognition system for an autonomous quadrotor drone.
Listen to the user's spoken audio instruction.
Transcribe the user's flight navigation command verbatim into clear natural text.
Examples of typical instructions:
- "Find two poles and hover between them"
- "Go to the orange pad"
- "Find the window and go through the window"
- "locate the arch and go through it"
- "Go to the elevated yellow pad"
- "Return to launch position"
- "Inspect all arena landmarks in sequence"

RULES:
1. Output ONLY the raw flight command text.
2. Do NOT include conversational greetings, explanations, notes, metadata, or markdown.
3. If the audio is silent or unintelligible, output "SILENCE".
"""


class GeminiVoiceTranscriber:
    """High-resilience speech transcriber using Google Gemini with browser fallback."""

    CANDIDATE_MODELS = [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-flash-latest",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY_2")
            or os.getenv("GOOGLE_API_KEY")
        )
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initializes genai client if key is available."""
        if not self.api_key:
            return
        try:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=10000),
            )
        except Exception as err:
            print(f"⚠️ [VOICE TRANSCRIBER] Failed to initialize Gemini client: {err}")
            self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if Gemini client is initialized."""
        return self._client is not None

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        client_fallback: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        """Transcribes audio bytes into text with Gemini and fallback support."""
        t_start = time.time()
        clean_fallback = (client_fallback or "").strip()

        # Clean mime type
        clean_mime = mime_type.split(";")[0].strip() or "audio/webm"

        # Check if we have audio or fallback
        if not audio_bytes or len(audio_bytes) < 64:
            if clean_fallback:
                return VoiceTranscriptionResult(
                    text=clean_flight_instruction(clean_fallback),
                    raw_text=clean_fallback,
                    provider="browser_speech",
                    model="browser_webkit_speech",
                    confidence=0.92,
                    latency_ms=(time.time() - t_start) * 1000,
                    success=True,
                )
            return VoiceTranscriptionResult(
                text="",
                raw_text="",
                provider="gemini",
                model="none",
                confidence=0.0,
                latency_ms=(time.time() - t_start) * 1000,
                success=False,
                error="Audio buffer is empty and no client fallback provided",
            )

        # 1. Try Gemini models
        if self._client:
            from google.genai import types

            part = types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime)
            models_to_try = [model_override] if model_override else self.CANDIDATE_MODELS

            for model_name in models_to_try:
                if not model_name:
                    continue
                try:
                    resp = self._client.models.generate_content(
                        model=model_name,
                        contents=[part, VOICE_TRANSCRIBE_PROMPT],
                    )
                    raw_text = (resp.text or "").strip()
                    if raw_text and raw_text.upper() != "SILENCE":
                        clean_cmd = clean_flight_instruction(raw_text)
                        if clean_cmd:
                            elapsed_ms = (time.time() - t_start) * 1000
                            return VoiceTranscriptionResult(
                                text=clean_cmd,
                                raw_text=raw_text,
                                provider="gemini",
                                model=model_name,
                                confidence=0.98,
                                latency_ms=elapsed_ms,
                                success=True,
                            )
                except Exception as err:
                    print(f"⚠️ [VOICE TRANSCRIBER] Gemini model '{model_name}' failed: {err}")
                    continue

        # 2. Resilient Browser Speech Recognition Fallback
        if clean_fallback:
            elapsed_ms = (time.time() - t_start) * 1000
            print(f"🔄 [VOICE TRANSCRIBER] Using client speech preview fallback: \"{clean_fallback}\"")
            return VoiceTranscriptionResult(
                text=clean_flight_instruction(clean_fallback),
                raw_text=clean_fallback,
                provider="browser_speech",
                model="browser_webkit_speech",
                confidence=0.90,
                latency_ms=elapsed_ms,
                success=True,
            )

        # 3. Complete failure
        elapsed_ms = (time.time() - t_start) * 1000
        return VoiceTranscriptionResult(
            text="",
            raw_text="",
            provider="gemini",
            model="none",
            confidence=0.0,
            latency_ms=elapsed_ms,
            success=False,
            error="Speech transcription unavailable (Gemini service timed out and no client audio transcript available)",
        )

    async def transcribe_audio_async(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/webm",
        client_fallback: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> VoiceTranscriptionResult:
        """Asynchronously transcribes audio without blocking the FastAPI event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.transcribe_audio,
            audio_bytes,
            mime_type,
            client_fallback,
            model_override,
        )


# Global singleton instance
voice_transcriber = GeminiVoiceTranscriber()
