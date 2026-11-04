"""System 2 Natural Language Intent Parser and Semantic Action Mapper.

Translates human-authored natural language instructions (e.g. "Find two poles and hover
between them", "Find the window and go through the window", "Go to the orange pad")
into structured quadrotor navigation targets, visual landmark descriptions, and
spatial guidance plans using Large Language Models (Groq, Gemini) with a resilient
offline heuristic classifier fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Canonical Arena Landmarks and Semantic Metadata
ARENA_LANDMARKS: Dict[str, Dict[str, str]] = {
    "poles": {
        "action": "poles",
        "semantic_label": "Two Corridor Pylons",
        "visual_features": "Dual red and amber vertical boundary posts framing the runway corridor entrance",
        "default_reasoning": "Located the dual vertical pylon gate at corridor entrance; aligning along central glidepath between poles.",
    },
    "alpha": {
        "action": "alpha",
        "semantic_label": "Orange Landing Pad",
        "visual_features": "Concentric amber and white circular hover bullseye on the arena floor plane",
        "default_reasoning": "Identified high-contrast orange landing pad on the ground; executing visual descent glidepath.",
    },
    "delta": {
        "action": "delta",
        "semantic_label": "Elevated Yellow Platform",
        "visual_features": "Raised circular caution pedestal platform elevated at Z=0.70m above ground",
        "default_reasoning": "Detected elevated yellow platform pedestal; climbing to clear rim and establishing precision hover.",
    },
    "window": {
        "action": "window",
        "semantic_label": "Wall Window Aperture",
        "visual_features": "Elevated rectangular aperture cut into the left partition wall at 90° azimuth",
        "default_reasoning": "Framed rectangular window aperture in FPV camera; executing 90° yaw saccade and aperture ingress.",
    },
    "arch": {
        "action": "arch",
        "semantic_label": "Arched Doorway Gate",
        "visual_features": "High-clearance curved archway passage through the right boundary partition at -90° azimuth",
        "default_reasoning": "Detected high-clearance arched doorway; executing -90° yaw saccade and traversal corridor approach.",
    },
    "home": {
        "action": "home",
        "semantic_label": "Launch Pad / Home",
        "visual_features": "Origin takeoff station and primary base landing zone at X=0, Y=0",
        "default_reasoning": "Returning along runway centerline to base launch station for stabilized recovery.",
    },
    "grand_tour": {
        "action": "grand_tour",
        "semantic_label": "Full Arena Inspection Sequence",
        "visual_features": "Multi-landmark inspection loop covering poles, orange pad, yellow platform, window, and arch",
        "default_reasoning": "Initiating comprehensive visual inspection flight across all detected arena landmarks in sequence.",
    },
}


class NaturalLanguageIntent(BaseModel):
    """Structured representation of a parsed natural language navigation instruction."""

    raw_instruction: str = Field(..., description="Original user prompt or command string")
    target_action: str = Field(..., description="Action target: poles | alpha | delta | window | arch | home | grand_tour")
    semantic_label: str = Field(..., description="Human-readable natural name of visual landmark")
    visual_features: str = Field(..., description="Identified visual descriptors in camera view")
    intent_reasoning: str = Field(..., description="Step-by-step System 2 spatial reasoner summary")
    confidence: float = Field(0.95, ge=0.0, le=1.0, description="Semantic classification confidence score")
    source: str = Field("llm", description="Origin of interpretation: 'llm_groq' | 'llm_gemini' | 'heuristic_fallback'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_instruction": self.raw_instruction,
            "target_action": self.target_action,
            "semantic_label": self.semantic_label,
            "visual_features": self.visual_features,
            "intent_reasoning": self.intent_reasoning,
            "confidence": round(self.confidence, 2),
            "source": self.source,
        }


SYSTEM2_INTENT_PROMPT = """You are the System 2 Vision-Language-Navigation (VLN) Meta-Controller for an autonomous aerial drone.
The quadrotor operates in a testing arena containing the following physical landmarks observed through onboard cameras:
1. 'poles' (Two Corridor Pylons): Dual red and amber vertical boundary pylons framing the corridor entrance. Keywords: poles, pylons, two posts, between poles, corridor entrance.
2. 'alpha' (Orange Landing Pad): Concentric amber and white circular hover pad on the arena floor. Keywords: orange pad, amber pad, bullseye, circular target, ground pad.
3. 'delta' (Elevated Yellow Platform): Raised yellow circular caution pedestal platform elevated above the floor. Keywords: yellow pad, elevated platform, raised disc, pedestal, yellow platform.
4. 'window' (Wall Window Aperture): Rectangular aperture cut into the left partition wall (90° left). Keywords: window, left window, wall opening, rectangular aperture, go through window.
5. 'arch' (Arched Doorway Gate): Curved high-clearance arched doorway through the right partition wall (-90° right). Keywords: arch, right arch, doorway, archway, arched gate, go through arch.
6. 'home' (Launch Pad / Home): Origin starting pad / base. Keywords: home, base, origin, starting point, launch pad, return, land.
7. 'grand_tour' (Full Arena Inspection Sequence): Comprehensive multi-landmark visual inspection sequence. Keywords: tour, grand tour, all targets, inspect all, full sequence, entire arena.

Given the user's natural language instruction, parse their intent into the target landmark.
You MUST respond with a STRICT, valid JSON object matching this exact schema:
{
  "target_action": "poles" | "alpha" | "delta" | "window" | "arch" | "home" | "grand_tour",
  "semantic_label": "<natural human-readable object name>",
  "visual_features": "<visual description of landmark seen by camera>",
  "intent_reasoning": "<1-sentence System 2 reasoning explaining how prompt maps to visual landmark>",
  "confidence": 0.95
}
"""


class NaturalLanguageIntentParser:
    """Parses free-form flight commands into System 2 navigation actions with resilient fallbacks."""

    def __init__(self, preferred_provider: str = "groq") -> None:
        self.preferred_provider = preferred_provider.lower().strip()
        self._groq_client = None
        self._gemini_client = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Lazily initializes LLM API clients from environment variables."""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_key)
            except Exception:
                pass

        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_key)
            except Exception:
                pass

    def parse_instruction(
        self,
        instruction: str,
        use_llm: bool = True,
        model_override: Optional[str] = None,
    ) -> NaturalLanguageIntent:
        """Synchronously parses a natural language flight instruction into a navigation intent."""
        clean_text = instruction.strip()
        if not clean_text:
            return self._heuristic_fallback("home", clean_text)

        if use_llm:
            # 1. Try Groq (ultra-low latency, sub-second response)
            if self.preferred_provider == "groq" and self._groq_client:
                try:
                    return self._parse_with_groq(clean_text, model_override)
                except Exception as err:
                    print(f"⚠️ [INTENT PARSER] Groq LLM call failed ({err}), trying fallback...")

            # 2. Try Gemini 3.8 Flash
            if self._gemini_client:
                try:
                    return self._parse_with_gemini(clean_text, model_override)
                except Exception as err:
                    print(f"⚠️ [INTENT PARSER] Gemini LLM call failed ({err}), using local heuristic...")

            # 3. Try Groq if Gemini was preferred but failed
            if self.preferred_provider != "groq" and self._groq_client:
                try:
                    return self._parse_with_groq(clean_text, model_override)
                except Exception:
                    pass

        # 4. Deterministic Local Heuristic Fallback (zero latency, zero network dependency)
        return self._heuristic_fallback(self._classify_rule_based(clean_text), clean_text)

    async def parse_instruction_async(
        self,
        instruction: str,
        use_llm: bool = True,
        model_override: Optional[str] = None,
    ) -> NaturalLanguageIntent:
        """Asynchronously parses natural language instruction without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.parse_instruction,
            instruction,
            use_llm,
            model_override,
        )

    def _parse_with_groq(self, instruction: str, model_override: Optional[str]) -> NaturalLanguageIntent:
        """Queries Groq Cloud LLM for structured intent parsing."""
        model = model_override or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        res = self._groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM2_INTENT_PROMPT},
                {"role": "user", "content": instruction},
            ],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0.1,
            timeout=3.0,
        )
        content = res.choices[0].message.content
        data = json.loads(content)
        action = data.get("target_action", "alpha").lower().strip()
        if action not in ARENA_LANDMARKS:
            action = self._classify_rule_based(instruction)

        meta = ARENA_LANDMARKS[action]
        return NaturalLanguageIntent(
            raw_instruction=instruction,
            target_action=action,
            semantic_label=data.get("semantic_label") or meta["semantic_label"],
            visual_features=data.get("visual_features") or meta["visual_features"],
            intent_reasoning=data.get("intent_reasoning") or meta["default_reasoning"],
            confidence=float(data.get("confidence", 0.95)),
            source="llm_groq",
        )

    def _parse_with_gemini(self, instruction: str, model_override: Optional[str]) -> NaturalLanguageIntent:
        """Queries Google Gemini 3.8 Flash for structured intent parsing."""
        model = model_override or "gemini-3.8-flash"
        prompt = f"{SYSTEM2_INTENT_PROMPT}\n\nUSER INSTRUCTION:\n{instruction}"
        resp = self._gemini_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = resp.text.strip()
        # Extract JSON from markdown fences if present
        if "```" in text:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if m:
                text = m.group(1).strip()

        data = json.loads(text)
        action = data.get("target_action", "alpha").lower().strip()
        if action not in ARENA_LANDMARKS:
            action = self._classify_rule_based(instruction)

        meta = ARENA_LANDMARKS[action]
        return NaturalLanguageIntent(
            raw_instruction=instruction,
            target_action=action,
            semantic_label=data.get("semantic_label") or meta["semantic_label"],
            visual_features=data.get("visual_features") or meta["visual_features"],
            intent_reasoning=data.get("intent_reasoning") or meta["default_reasoning"],
            confidence=float(data.get("confidence", 0.95)),
            source="llm_gemini",
        )

    def _classify_rule_based(self, text: str) -> str:
        """Deterministic keyword-based classifier ensuring 100% reliable offline operation."""
        low = text.lower().strip()

        # Grand tour / all
        if any(w in low for w in ("tour", "grand", "all", "sequence", "inspect all", "every", "full")):
            return "grand_tour"

        # Window
        if any(w in low for w in ("window", "aperture", "wall opening", "left opening", "ingress window")):
            return "window"

        # Arch
        if any(w in low for w in ("arch", "archway", "doorway", "curved gate", "right arch")):
            return "arch"

        # Yellow platform / delta
        if any(w in low for w in ("yellow", "elevated", "platform", "pedestal", "disc", "delta", "raised")):
            return "delta"

        # Poles / pylons
        if any(w in low for w in ("pole", "poles", "pylon", "pylons", "post", "posts", "between", "gate")):
            return "poles"

        # Orange pad / alpha
        if any(w in low for w in ("orange", "amber", "bullseye", "concentric", "alpha", "landing pad", "circle")):
            return "alpha"

        # Home / base / return
        if any(w in low for w in ("home", "base", "origin", "launch", "start", "land", "return")):
            return "home"

        # Default fallback
        return "alpha"

    def _heuristic_fallback(self, action: str, raw_text: str) -> NaturalLanguageIntent:
        """Builds a structured intent from the deterministic rule classifier."""
        if action not in ARENA_LANDMARKS:
            action = "alpha"
        meta = ARENA_LANDMARKS[action]
        return NaturalLanguageIntent(
            raw_instruction=raw_text,
            target_action=action,
            semantic_label=meta["semantic_label"],
            visual_features=meta["visual_features"],
            intent_reasoning=meta["default_reasoning"],
            confidence=0.92,
            source="heuristic_fallback",
        )


# Global singleton instance
intent_parser = NaturalLanguageIntentParser(preferred_provider="groq")
