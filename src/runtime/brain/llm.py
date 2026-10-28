"""System 2 (LLM Meta-Controller) Coordination Engine.

This module provides the high-level reasoning brain that coordinates with System 1
via the System1System2Bridge.

Architecture:
-------------
- System 1 (500 Hz reflex controller) stabilizes the quadrotor and detects fast anomalies
  (pitch/roll excursions, gyro oscillations, unmodeled disturbance forces from L1).
- System 2 (this module, running asynchronously at 1-5 Hz in a background thread)
  consumes structured 1.0s sliding-window telemetry summaries, reasons about physical
  causes, and applies compensatory patches:
    * feedforward_force_offset: [Fx, Fy, Fz] (N)
    * gain_multipliers: {pos_kp, pos_kd, att_kp, att_kd}
    * motor_trim_scale: [s1, s2, s3, s4]
    * safety_envelope: {max_tilt_deg}

Time basis (unified): System2Worker cooldowns, patch ages, and trigger expiry all
use MuJoCo SIM time from TelemetrySummary, matching the bridge. Wall time is used
only for thread sleeps and measuring LLM call latency.
"""

import os
import re
import json
import time
import threading
from typing import Optional, Dict, Any
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from runtime.schemas.tier2 import System2Patch, GainMultipliers, SafetyEnvelope
from runtime.schemas.telemetry import TelemetrySummary, FastTriggerEvent
from runtime.sim.bridge import System1System2Bridge
from runtime.logging import LLMLogger, get_llm_logger

# Load environment variables (.env)
load_dotenv()


# F1 motor-trim physics: motor 1 at 40% power produces 0.40 x commanded thrust.
# Full compensation needs a 1/0.40 = 2.5x trim on M1 (0.40 x 2.50 = 1.00, i.e.
# hover 3.25 N x 2.50 = 8.1 N effective, under the 13 N per-motor limit), or
# equivalently ~1.20x on the three healthy motors (missing 1.95 N / 3 = 0.65 N
# each: 3.25 -> 3.90 N). The old prompt value 1.60x only restored 64% of M1.
SYSTEM_PROMPT = """You are the System 2 Autonomous Flight Reasoner for a Skydio X2 quadrotor.
CRITICAL INSTRUCTION: Output valid JSON immediately. Keep reasoning under 50 words.

Physical Specifications:
- Mass: 1.325 kg, total hover weight W = mg ≈ 13.0 N.
- Nominal hover thrust per motor: ~3.25 N (Motors 1-4, max thrust 13.0 N/rotor).
- Motor layout: M1=Rear-Left, M2=Rear-Right, M3=Front-Right, M4=Front-Left.

Guidance on Control Levers:
- feedforward_force_offset:
  * Lateral wind: If external wind disturbance force is detected along X (Fx ≈ +5.0N or Fx ≈ -5.0N) or Y, or if position is drifting downwind, command opposing feedforward force offset: dFx = -Fx (e.g. if Fx ≈ +5.0N, command dFx = -5.0; if Fx ≈ -5.0N, command dFx = +5.0). This tilts the drone thrust vector into the wind to hold position at 0.0m. Do NOT dismiss persistent force as a false alarm.
  * Vertical: Keep dFz = 0.0 unless there is significant altitude loss (|dz| > 0.15m).
- motor_trim_scale:
  * Motor 1 Failure: If Motor 1 thrust is elevated (M1 > 5.0N vs others ~3.2N) or Motor 1 is degraded (40% power / moment disturbance), set motor_trim_scale: [2.50, 1.0, 1.0, 1.0] to restore M1 effective thrust (0.40 x 2.50 = 1.00; hover 3.25N x 2.50 = 8.1N, still under the 13N limit). Alternative: [1.0, 1.20, 1.20, 1.20] spreads the missing ~1.95N over the three healthy motors (3.25 -> 3.90N each).
- gain_multipliers:
  * Sensor Noise / Gyro Chatter: If gyro RMS > 1.0 rad/s, angular rate spikes, or oscillation is detected, set gain_multipliers: {"att_kp": 0.40, "att_kd": 1.6, "pos_kp": 1.0, "pos_kd": 1.0} to damp high-frequency vibration chatter.
  * If drifting sluggishly against wind, set pos_kp = 1.2 to 1.3.

Required JSON Structure:
{
  "diagnosis": "Brief 1-sentence physical explanation.",
  "feedforward_force_offset": [dFx, dFy, dFz],
  "gain_multipliers": {"pos_kp": 1.0, "pos_kd": 1.0, "att_kp": 1.0, "att_kd": 1.0},
  "motor_trim_scale": [1.0, 1.0, 1.0, 1.0],
  "safety_envelope": {"max_tilt_deg": 20.0}
}
OUTPUT ONLY RAW VALID JSON. No markdown codeblocks, no explanations outside JSON."""


def _extract_json_patch(text: str) -> Optional[dict]:
    """Robustly extracts a valid System 2 JSON patch from model output or reasoning."""
    if not text:
        return None
    cleaned = text.strip()

    # 1. Try stripping markdown codeblock if present
    if "```" in cleaned:
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if code_match:
            try:
                d = json.loads(code_match.group(1).strip())
                if isinstance(d, dict) and "diagnosis" in d:
                    return d
            except Exception:
                pass

    # 2. Try raw direct JSON parse
    try:
        d = json.loads(cleaned)
        if isinstance(d, dict) and "diagnosis" in d:
            return d
    except Exception:
        pass

    # 3. Search backwards for valid JSON dictionary containing 'diagnosis'
    for start in range(len(cleaned) - 1, -1, -1):
        if cleaned[start] == '{':
            for end in range(len(cleaned), start, -1):
                if cleaned[end - 1] == '}':
                    candidate = cleaned[start:end]
                    try:
                        d = json.loads(candidate)
                        if isinstance(d, dict) and "diagnosis" in d:
                            return d
                    except Exception:
                        continue
    return None


class System2Brain:
    """Pluggable LLM client interface for generating System 2 control patches.

    Supported providers:
      - 'groq': Ultra-fast Groq Cloud (~1.0s latency, default model: 'openai/gpt-oss-120b')
      - 'baseten': Baseten endpoint (default model: 'meta-models/muse-spark-open-1026')
      - 'openai': OpenAI or custom OpenAI-compatible endpoint
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        logger: Optional[LLMLogger] = None,
    ) -> None:
        # Determine provider: parameter -> LLM_PROVIDER env -> auto-detect -> default 'groq'
        env_provider = os.getenv("LLM_PROVIDER")
        self.provider = (provider or env_provider or "").lower().strip()
        if not self.provider:
            if os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.getenv("BASETEN_API_KEY"):
                self.provider = "baseten"
            else:
                self.provider = "groq"

        self.logger = logger or get_llm_logger()
        self.base_url = base_url

        if self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in environment or .env file! "
                    "Please configure GROQ_API_KEY to enable System 2 LLM reasoning."
                )
            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            from groq import Groq
            self.client = Groq(api_key=self.api_key)

        elif self.provider == "baseten":
            self.api_key = api_key or os.getenv("BASETEN_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "BASETEN_API_KEY not found in environment or .env file! "
                    "Please configure BASETEN_API_KEY to enable System 2 LLM reasoning."
                )
            self.base_url = self.base_url or os.getenv(
                "BASETEN_BASE_URL", "https://inference.baseten.co/v1"
            )
            self.model = model or os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file!")
            self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)

        else:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Supported providers: 'groq', 'baseten', 'openai'."
            )

    def analyze_and_patch(
        self,
        summary: TelemetrySummary,
        trigger_event: Optional[FastTriggerEvent] = None,
    ) -> Optional[System2Patch]:
        """Calls the LLM with the latest telemetry summary and parses the resulting patch.

        Args:
            summary: 1.0s pre-processed TelemetrySummary from the bridge.
            trigger_event: Optional trigger event with instant trigger reasons.

        Returns:
            System2Patch containing structured control adjustments, or None if call fails.
        """
        trigger_header = trigger_event.format_trigger_header() if trigger_event else None
        user_prompt = summary.format_llm_text(trigger_header=trigger_header)

        start_t = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = None
        raw_text = None
        reasoning = None

        # Timing metadata recorded into logs/llm_calls.jsonl for every call.
        timing_metadata: Dict[str, Any] = {
            "fault_type": trigger_event.fault_type if trigger_event else None,
            "time_to_trigger_s": trigger_event.time_to_trigger_s if trigger_event else None,
            "trigger_sim_time": trigger_event.timestamp if trigger_event else None,
            "summary_sim_time": summary.timestamp,
        }

        try:
            call_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            if self.provider == "groq":
                call_kwargs["max_completion_tokens"] = 2048
            else:
                call_kwargs["max_tokens"] = 4096

            response = self.client.chat.completions.create(**call_kwargs)
            elapsed_s = time.time() - start_t

            # Extract token usage if returned by endpoint
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or (prompt_tokens + completion_tokens)
                if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", None)

            msg = response.choices[0].message
            raw_text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""

            # Check if JSON is in content or reasoning_content fallback
            data = _extract_json_patch(raw_text)
            if not data and reasoning:
                data = _extract_json_patch(reasoning)

            if not data:
                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                raise ValueError(
                    f"Model returned invalid or empty JSON payload (finish_reason: {finish_reason})"
                )

            data["timestamp"] = time.time()
            patch = System2Patch.from_dict(data)

            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")

            # Log successful transaction
            self.logger.log_call(
                duration_s=elapsed_s,
                model=f"{self.provider}:{self.model}",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_output=raw_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                reasoning_content=reasoning,
                parsed_patch=patch.to_dict(),
                status="success",
                metadata={
                    "provider": self.provider,
                    "trigger_header": trigger_header,
                    "finish_reason": finish_reason,
                    **timing_metadata,
                },
            )

            self._print_reasoning_banner(
                trigger_header, patch, elapsed_s, prompt_tokens, completion_tokens, total_tokens
            )
            return patch

        except Exception as e:
            elapsed_s = time.time() - start_t
            print(f"⚠️ [SYSTEM 2 LLM ERROR - {self.provider.upper()}] Failed to generate control patch: {e}")

            # Log failed transaction
            self.logger.log_call(
                duration_s=elapsed_s,
                model=f"{self.provider}:{self.model}",
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                raw_output=raw_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                reasoning_content=reasoning,
                parsed_patch=None,
                status="error",
                error_message=str(e),
                metadata={"provider": self.provider, "trigger_header": trigger_header, **timing_metadata},
            )
            return None

    def _print_reasoning_banner(
        self,
        trigger_header: Optional[str],
        patch: System2Patch,
        elapsed_s: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Prints a human-readable banner summarizing LLM reasoning and applied levers."""
        print("\n" + "═" * 74)
        print(f"🧠 [SYSTEM 2 LLM META-CONTROLLER - {self.provider.upper()}: {self.model}] Reasoning Complete in {elapsed_s:.2f}s")
        print(f"📊 Tokens: Input={prompt_tokens} | Output={completion_tokens} | Total={total_tokens}")
        print("═" * 74)
        if trigger_header:
            print(f"🚨 Cause: {trigger_header}")
        print(f"📋 Diagnosis: {patch.diagnosis}")
        print("-" * 74)
        print("🎯 Control Patches Injected into System 1 (500 Hz Reflex Loop):")
        if patch.feedforward_force_offset:
            fx, fy, fz = patch.feedforward_force_offset
            print(f"  • Feedforward Force Offset: [Fx={fx:+.2f}N, Fy={fy:+.2f}N, Fz={fz:+.2f}N]")
        if patch.gain_multipliers:
            gm = patch.gain_multipliers
            print(
                f"  • Gain Multipliers: Kp_pos={gm.pos_kp:.2f}x | Kd_pos={gm.pos_kd:.2f}x | "
                f"Kp_att={gm.att_kp:.2f}x | Kd_att={gm.att_kd:.2f}x"
            )
        if patch.motor_trim_scale:
            m = patch.motor_trim_scale
            print(f"  • Motor Trim Scale: M1={m[0]:.2f}x | M2={m[1]:.2f}x | M3={m[2]:.2f}x | M4={m[3]:.2f}x")
        if patch.safety_envelope:
            se = patch.safety_envelope
            print(f"  • Safety Envelope: Max Tilt Clamped to {se.max_tilt_deg:.1f}°")
        print(f"💾 Transaction logged to: {self.logger.jsonl_path}")
        print("═" * 74 + "\n")


class System2Worker:
    """Asynchronous background worker that monitors the bridge and triggers LLM reasoning.

    Runs in a separate daemon thread to ensure the 500 Hz physics simulation
    never experiences any latency or blocking. All cooldowns and patch ages use
    sim time from TelemetrySummary (unified basis with the bridge).
    """

    def __init__(
        self,
        bridge: System1System2Bridge,
        brain: Optional[System2Brain] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        cooldown_s: float = 2.0,
        drone_name: str = "Drone",
    ) -> None:
        self.bridge = bridge
        self.brain = brain or System2Brain(provider=provider, model=model)
        self.cooldown_s = cooldown_s  # sim seconds between LLM calls
        self.drone_name = drone_name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_call_sim_time = 0.0
        self._patch_applied_sim_time = 0.0
        self._patch_notified_stable = False

    def start(self) -> None:
        """Launches the System 2 background monitoring thread."""
        if self._running:
            return
        self._running = True
        thread_name = f"System2Worker_{self.drone_name.replace(' ', '_')}"
        self._thread = threading.Thread(target=self._run_loop, name=thread_name, daemon=True)
        self._thread.start()
        print(f"🚀 [SYSTEM 2 - {self.drone_name.upper()}] Async Meta-Controller thread started. Monitoring System 1 bridge...")

    def stop(self) -> None:
        """Stops the System 2 background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print(f"🛑 [SYSTEM 2 - {self.drone_name.upper()}] Async Meta-Controller stopped.")

    def _print_timing(
        self,
        trigger_event: Optional[FastTriggerEvent],
        patch_sim_time: float,
        llm_wall_s: float,
    ) -> None:
        """Prints one timing line per applied patch: fault, trigger, and patch latency."""
        if trigger_event is not None:
            time_to_patch = patch_sim_time - trigger_event.timestamp
            fault = trigger_event.fault_type or "unknown"
            ttt = trigger_event.time_to_trigger_s
            ttt_str = f"{ttt:.2f}s" if ttt is not None else "unknown"
            print(
                f"⏱️ [TIMING - {self.drone_name.upper()}] fault={fault} "
                f"time-to-trigger={ttt_str} time-to-patch={time_to_patch:.2f}s "
                f"(LLM call took {llm_wall_s:.2f}s wall)"
            )
        else:
            print(
                f"⏱️ [TIMING - {self.drone_name.upper()}] fault=unknown (no trigger event) "
                f"time-to-patch=n/a (LLM call took {llm_wall_s:.2f}s wall)"
            )

    def _run_loop(self) -> None:
        """Main loop executed in background thread."""
        while self._running:
            try:
                # The 1.0 s summary is kept for the LLM prompt, but the decision
                # to call uses the instant (50 ms) force estimate as well, so a
                # fast fault does not wait for a full 1.0 s mean to build up.
                summary = self.bridge.get_telemetry_summary(window_seconds=1.0)

                # Skip initial spawn warmup (first 1.0s of physics)
                if summary.timestamp < 1.0 or summary.samples_count < 200:
                    time.sleep(0.1)
                    continue

                # Cooldown between LLM calls, in sim seconds (unified basis).
                if (summary.timestamp - self._last_call_sim_time) < self.cooldown_s:
                    time.sleep(0.1)
                    continue

                pos_err_norm = (summary.pos_error[0]**2 + summary.pos_error[1]**2 + summary.pos_error[2]**2) ** 0.5
                force_dist_norm = (summary.unmodeled_force_disturbance[0]**2 + summary.unmodeled_force_disturbance[1]**2 + summary.unmodeled_force_disturbance[2]**2) ** 0.5
                instant_force_norm = (summary.instantaneous_force_disturbance[0]**2 + summary.instantaneous_force_disturbance[1]**2 + summary.instantaneous_force_disturbance[2]**2) ** 0.5
                moment_dist_norm = (summary.unmodeled_moment_disturbance[0]**2 + summary.unmodeled_moment_disturbance[1]**2 + summary.unmodeled_moment_disturbance[2]**2) ** 0.5
                triggered = self.bridge.is_triggered()

                active_patch = self.bridge.get_active_patch()

                if active_patch is None:
                    # --------------------------------------------------------------
                    # CASE 1: Baseline System 1 running (No active LLM patch)
                    # --------------------------------------------------------------
                    needs_attention = (
                        triggered
                        or pos_err_norm > 0.30
                        or force_dist_norm > 2.5
                        or instant_force_norm > 2.5
                        or moment_dist_norm > 0.35
                        or summary.oscillation_detected
                    )

                    if needs_attention:
                        trigger_event = self.bridge.get_latest_trigger_event(max_age_s=3.0, current_time=summary.timestamp)
                        self.bridge.clear_trigger()
                        self._last_call_sim_time = summary.timestamp

                        # Query LLM asynchronously
                        llm_start = time.time()
                        patch = self.brain.analyze_and_patch(summary, trigger_event)
                        llm_wall_s = time.time() - llm_start
                        if patch:
                            self.bridge.apply_system2_patch(patch)
                            self._patch_applied_sim_time = summary.timestamp
                            self._patch_notified_stable = False
                            self._print_timing(trigger_event, summary.timestamp, llm_wall_s)
                            print(f"\n✨ [SYSTEM 2 - {self.drone_name.upper()}] Applied Control Patch to System 1:")
                            print(f"   Diagnosis: {patch.diagnosis}")
                            if patch.feedforward_force_offset and any(abs(x) > 0.01 for x in patch.feedforward_force_offset):
                                 print(f"   Feedforward Force: {patch.feedforward_force_offset} N")
                            if patch.motor_trim_scale:
                                 print(f"   Motor Trim: {patch.motor_trim_scale}")
                            if patch.gain_multipliers:
                                 print(f"   Gain Multipliers: {patch.gain_multipliers}\n")

                else:
                    # --------------------------------------------------------------
                    # CASE 2: Active Patch already applied to System 1
                    # --------------------------------------------------------------
                    self.bridge.clear_trigger()

                    # Check if disturbance has ended and flight is back to calm nominal hover
                    calm_hover = (
                        pos_err_norm < 0.15
                        and force_dist_norm < 1.0
                        and moment_dist_norm < 0.20
                        and not summary.oscillation_detected
                    )

                    if calm_hover:
                        if (summary.timestamp - self._last_call_sim_time) > 2.5:
                            self.bridge.clear_active_patch()
                            print(f"\n🟢 [SYSTEM 2 - {self.drone_name.upper()}] Disturbance ended & flight nominal. Resetting System 1 control law to baseline.\n")
                            self._last_call_sim_time = summary.timestamp
                            self._patch_notified_stable = False
                    else:
                        # Disturbance is still active. Is the active patch holding flight stable?
                        is_stable = (pos_err_norm < 0.30) and not summary.oscillation_detected
                        if is_stable and not self._patch_notified_stable:
                            print(f"🎯 [SYSTEM 2 - {self.drone_name.upper()}] Active control patch holding flight stable against disturbance. Monitoring...\n")
                            self._patch_notified_stable = True

                        # Re-check the patch at 2 s (not 6 s): if position error
                        # is still > 0.5 m or oscillation continues, re-consult.
                        patch_age = summary.timestamp - self._patch_applied_sim_time
                        severe_divergence = (pos_err_norm > 0.5) or (summary.oscillation_detected and patch_age > 2.0)

                        if severe_divergence and patch_age > 2.0:
                            print(f"⚠️ [SYSTEM 2 - {self.drone_name.upper()}] Active patch insufficient; re-consulting LLM to revise control law...")
                            trigger_event = self.bridge.get_latest_trigger_event(max_age_s=3.0, current_time=summary.timestamp)
                            self._last_call_sim_time = summary.timestamp
                            llm_start = time.time()
                            revised_patch = self.brain.analyze_and_patch(summary, trigger_event)
                            llm_wall_s = time.time() - llm_start
                            if revised_patch:
                                self.bridge.apply_system2_patch(revised_patch)
                                self._patch_applied_sim_time = summary.timestamp
                                self._print_timing(trigger_event, summary.timestamp, llm_wall_s)
                                print(f"\n🔄 [SYSTEM 2 - {self.drone_name.upper()}] Revised Control Patch:")
                                print(f"   Diagnosis: {revised_patch.diagnosis}\n")

            except Exception as e:
                import traceback
                print(f"⚠️ [SYSTEM 2 - {self.drone_name.upper()} WORKER ERROR] {e}")
                traceback.print_exc()

            time.sleep(0.05)


def run_test() -> None:
    """Self-test demonstrating live System 2 reasoning and patch generation."""
    print("Initializing System 2 Brain with Baseten LLM...")
    brain = System2Brain()

    sample_summary = TelemetrySummary(
        timestamp=4.2,
        window_duration_s=1.0,
        samples_count=500,
        roll_deg=1.2,
        pitch_deg=18.5,
        yaw_deg=0.5,
        gyro_rms=[0.2, 3.9, 0.1],
        oscillation_detected=True,
        oscillation_note="pitch oscillation detected at ~4.0Hz, wy_rms=3.9rad/s",
        pos_error=[-0.75, 0.02, -0.33],
        velocity=[-0.60, 0.0, -0.25],
        motor_commands_mean=[5.1, 5.1, 3.2, 3.2],
        motor_saturation_pct=[0.0, 0.0, 0.0, 0.0],
        unmodeled_force_disturbance=[-4.9, 0.1, -3.9],
        unmodeled_moment_disturbance=[0.0, -0.45, 0.0],
    )

    trigger_event = FastTriggerEvent(
        timestamp=4.2,
        reasons=["Position error (0.82m) exceeded threshold (0.30m)", "Physics residual spiked"],
        pos_error_norm=0.82,
        tilt_deg=18.5,
        gyro_norm=3.9,
        force_residual_norm=6.26,
    )

    print("Sending live telemetry summary to LLM...")
    patch = brain.analyze_and_patch(sample_summary, trigger_event)
    if patch:
        print("Generated Patch Output:")
        print(json.dumps(patch.to_dict(), indent=2))
    else:
        print("Failed to generate patch.")


if __name__ == "__main__":
    run_test()
