"""System 2 LLM Call Logger and Telemetry Auditor.

This module provides thread-safe, structured logging of all LLM interactions
between System 1 and System 2, recording:
  - Exact call timestamp & latency duration
  - Token counts (input / prompt tokens, output / completion tokens, reasoning tokens)
  - Full input prompt (system instructions and sliding-window telemetry text)
  - Full model output (raw text, reasoning thinking trace, and parsed JSON patch)
  - Call status and error diagnostics
"""

import os
import json
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# Default log storage directory: <workspace_root>/logs
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


@dataclass
class LLMCallRecord:
    """Represents a single LLM reasoning transaction."""
    call_id: str
    timestamp_iso: str
    timestamp_unix: float
    duration_s: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int]
    system_prompt: str
    user_prompt: str
    raw_output: Optional[str]
    reasoning_content: Optional[str]
    parsed_patch: Optional[Dict[str, Any]]
    status: str  # "success" or "error"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the record into a serializable dictionary."""
        return asdict(self)


class LLMLogger:
    """Thread-safe auditor and logger for System 2 LLM reasoning calls.

    Writes to:
      1. Machine-readable JSON Lines file (`llm_calls.jsonl`)
      2. Human-readable text log ledger (`llm_calls.log`)
      3. In-memory deque for programmatic inspection and telemetry dashboards
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        jsonl_filename: str = "llm_calls.jsonl",
        text_filename: str = "llm_calls.log",
        max_in_memory: int = 100,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / jsonl_filename
        self.text_path = self.log_dir / text_filename
        self.max_in_memory = max_in_memory

        self._lock = threading.Lock()
        self._history: List[LLMCallRecord] = []

    def log_call(
        self,
        duration_s: float,
        model: str,
        system_prompt: str,
        user_prompt: str,
        raw_output: Optional[str],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: Optional[int] = None,
        reasoning_content: Optional[str] = None,
        parsed_patch: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMCallRecord:
        """Records an LLM call transaction to disk and memory.

        Args:
            duration_s: Wall-clock latency of LLM completion in seconds.
            model: Model name string.
            system_prompt: Full system prompt sent to model.
            user_prompt: Full user prompt sent to model.
            raw_output: Raw content string returned by LLM.
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
            total_tokens: Total tokens consumed.
            reasoning_tokens: Reasoning tokens used by thinking models.
            reasoning_content: Reasoning trace string if available.
            parsed_patch: Parsed control patch dictionary.
            status: "success" or "error".
            error_message: Failure message if status is "error".
            metadata: Additional contextual metadata (e.g. triggers, sim timestamp).

        Returns:
            The created LLMCallRecord instance.
        """
        now = time.time()
        now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        call_id = f"call_{uuid.uuid4().hex[:8]}"

        record = LLMCallRecord(
            call_id=call_id,
            timestamp_iso=now_iso,
            timestamp_unix=now,
            duration_s=round(duration_s, 4),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            reasoning_content=reasoning_content,
            parsed_patch=parsed_patch,
            status=status,
            error_message=error_message,
            metadata=metadata or {},
        )

        with self._lock:
            # 1. Update In-memory history
            self._history.append(record)
            if len(self._history) > self.max_in_memory:
                self._history.pop(0)

            # 2. Append to JSON Lines file
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), default=str) + "\n")
                    f.flush()
            except Exception as e:
                print(f"⚠️ [LLM LOGGER] Failed to write to {self.jsonl_path}: {e}")

            # 3. Append to Human-readable text ledger
            try:
                with open(self.text_path, "a", encoding="utf-8") as f:
                    f.write(self._format_text_entry(record))
                    f.flush()
            except Exception as e:
                print(f"⚠️ [LLM LOGGER] Failed to write to {self.text_path}: {e}")

        return record

    def get_recent_calls(self, limit: int = 10) -> List[LLMCallRecord]:
        """Retrieves the most recent N logged LLM calls."""
        with self._lock:
            return list(self._history[-limit:])

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Computes aggregate statistics across all logged in-memory calls."""
        with self._lock:
            if not self._history:
                return {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "avg_duration_s": 0.0,
                }

            total_calls = len(self._history)
            success = sum(1 for r in self._history if r.status == "success")
            failed = total_calls - success
            p_tokens = sum(r.prompt_tokens for r in self._history)
            c_tokens = sum(r.completion_tokens for r in self._history)
            tot_tokens = sum(r.total_tokens for r in self._history)
            avg_duration = sum(r.duration_s for r in self._history) / total_calls

            return {
                "total_calls": total_calls,
                "successful_calls": success,
                "failed_calls": failed,
                "total_prompt_tokens": p_tokens,
                "total_completion_tokens": c_tokens,
                "total_tokens": tot_tokens,
                "avg_duration_s": round(avg_duration, 3),
            }

    @staticmethod
    def _format_text_entry(r: LLMCallRecord) -> str:
        """Formats a human-readable ledger block for text logging."""
        dur = float(r.duration_s) if isinstance(r.duration_s, (int, float)) else 0.0
        status_str = str(r.status).upper()
        lines = [
            "\n" + "=" * 74,
            f"🧠 LLM CALL RECORD [{r.call_id}] - {r.timestamp_iso}",
            f"Model: {r.model} | Latency: {dur:.2f}s | Status: {status_str}",
            f"Tokens: Prompt={r.prompt_tokens} | Completion={r.completion_tokens} | Total={r.total_tokens}"
            + (f" (Reasoning={r.reasoning_tokens})" if r.reasoning_tokens is not None else ""),
            "-" * 74,
            "📥 INPUT PROMPT:",
            str(r.user_prompt).strip(),
            "-" * 74,
            "📤 OUTPUT:",
        ]

        if str(r.status) == "error":
            lines.append(f"❌ ERROR: {r.error_message}")
            if r.raw_output:
                lines.append(f"Raw Output: {r.raw_output}")
        else:
            if r.parsed_patch:
                lines.append("Parsed Patch:")
                lines.append(json.dumps(r.parsed_patch, indent=2, default=str))
            else:
                lines.append(str(r.raw_output or "(Empty output)"))

        if r.reasoning_content:
            lines.append("-" * 74)
            lines.append(f"💭 Reasoning Trace ({len(str(r.reasoning_content))} chars):")
            trace_preview = str(r.reasoning_content).strip()
            if len(trace_preview) > 500:
                trace_preview = trace_preview[:250] + "\n... [truncated] ...\n" + trace_preview[-250:]
            lines.append(trace_preview)

        lines.append("=" * 74 + "\n")
        return "\n".join(str(l) for l in lines)


# Global default logger singleton
_DEFAULT_LOGGER: Optional[LLMLogger] = None
_LOGGER_INIT_LOCK = threading.Lock()


def get_llm_logger(log_dir: Optional[Path] = None) -> LLMLogger:
    """Returns the global default LLMLogger instance."""
    global _DEFAULT_LOGGER
    with _LOGGER_INIT_LOCK:
        if _DEFAULT_LOGGER is None:
            _DEFAULT_LOGGER = LLMLogger(log_dir=log_dir)
        return _DEFAULT_LOGGER
