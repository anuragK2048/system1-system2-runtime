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
from typing import Dict, List, Optional, Any, Tuple
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
        self.usage_json_path = self.log_dir / "provider_token_usage.json"
        self.usage_log_path = self.log_dir / "provider_token_usage.log"
        self.max_in_memory = max_in_memory

        self._lock = threading.Lock()
        self._history: List[LLMCallRecord] = []
        self._provider_usage: Dict[str, Dict[str, Any]] = {}
        self._init_provider_usage()

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

            # 4. Stateful token usage ledger update (separated by provider)
            try:
                self._update_provider_usage_locked(record)
            except Exception as e:
                print(f"⚠️ [LLM LOGGER] Failed to update provider usage ledger: {e}")

        return record

    def _extract_provider_and_model(self, model: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Resolves canonical provider key ('baseten', 'groq', 'fireworks', 'openai') and model name."""
        prov = metadata.get("provider") if metadata else None
        model_str = model or ""
        if ":" in model_str:
            p, m = model_str.split(":", 1)
            return p.lower().strip(), m.strip()
        if prov:
            return prov.lower().strip(), model_str.strip()

        m_lower = model_str.lower()
        if "muse" in m_lower or "baseten" in m_lower or "deepseek-v4" in m_lower:
            return "baseten", model_str
        if "fireworks" in m_lower or "accounts/fireworks" in m_lower:
            return "fireworks", model_str
        if "gpt-oss" in m_lower or "groq" in m_lower or "qwen" in m_lower:
            return "groq", model_str
        return "groq", model_str

    def _init_provider_usage(self) -> None:
        """Initializes or restores the stateful token usage ledger from disk."""
        # 1. Try reading existing JSON state
        if self.usage_json_path.exists():
            try:
                with open(self.usage_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._provider_usage = data.get("providers", {})
                    return
            except Exception as e:
                print(f"⚠️ [LLM LOGGER] Could not load {self.usage_json_path}: {e}")

        # 2. Rebuild state by aggregating historical records from llm_calls.jsonl if present
        self._provider_usage = {
            "baseten": self._make_empty_provider_dict(),
            "groq": self._make_empty_provider_dict(),
            "fireworks": self._make_empty_provider_dict(),
        }

        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                            prov, mod = self._extract_provider_and_model(
                                d.get("model", ""), d.get("metadata", {})
                            )
                            if prov not in self._provider_usage:
                                self._provider_usage[prov] = self._make_empty_provider_dict()

                            p_data = self._provider_usage[prov]
                            p_data["total_calls"] += 1
                            if d.get("status") == "success":
                                p_data["successful_calls"] += 1
                            else:
                                p_data["failed_calls"] += 1

                            pt = int(d.get("prompt_tokens") or 0)
                            ct = int(d.get("completion_tokens") or 0)
                            tt = int(d.get("total_tokens") or (pt + ct))

                            p_data["prompt_tokens"] += pt
                            p_data["completion_tokens"] += ct
                            p_data["total_tokens"] += tt
                            p_data["last_call_iso"] = d.get("timestamp_iso")

                            if mod not in p_data["models"]:
                                p_data["models"][mod] = {
                                    "total_calls": 0,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                }
                            m_data = p_data["models"][mod]
                            m_data["total_calls"] += 1
                            m_data["prompt_tokens"] += pt
                            m_data["completion_tokens"] += ct
                            m_data["total_tokens"] += tt
                        except Exception:
                            continue
            except Exception as e:
                print(f"⚠️ [LLM LOGGER] Error scanning historical logs: {e}")

        self._write_provider_usage_files()

    @staticmethod
    def _make_empty_provider_dict() -> Dict[str, Any]:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "last_call_iso": None,
            "models": {},
        }

    def _update_provider_usage_locked(self, record: LLMCallRecord) -> None:
        """Updates in-memory token ledger and flushes to disk (must be called with _lock held)."""
        prov, mod = self._extract_provider_and_model(record.model, record.metadata)
        if prov not in self._provider_usage:
            self._provider_usage[prov] = self._make_empty_provider_dict()

        p_data = self._provider_usage[prov]
        p_data["total_calls"] += 1
        if record.status == "success":
            p_data["successful_calls"] += 1
        else:
            p_data["failed_calls"] += 1

        p_data["prompt_tokens"] += record.prompt_tokens
        p_data["completion_tokens"] += record.completion_tokens
        p_data["total_tokens"] += record.total_tokens
        p_data["last_call_iso"] = record.timestamp_iso

        if mod not in p_data["models"]:
            p_data["models"][mod] = {
                "total_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        m_data = p_data["models"][mod]
        m_data["total_calls"] += 1
        m_data["prompt_tokens"] += record.prompt_tokens
        m_data["completion_tokens"] += record.completion_tokens
        m_data["total_tokens"] += record.total_tokens

        self._write_provider_usage_files()

    def get_provider_usage_summary(self) -> Dict[str, Any]:
        """Returns deep copy of stateful token/credit usage summary partitioned by provider."""
        with self._lock:
            grand_total = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            for p, d in self._provider_usage.items():
                grand_total["total_calls"] += d.get("total_calls", 0)
                grand_total["successful_calls"] += d.get("successful_calls", 0)
                grand_total["failed_calls"] += d.get("failed_calls", 0)
                grand_total["prompt_tokens"] += d.get("prompt_tokens", 0)
                grand_total["completion_tokens"] += d.get("completion_tokens", 0)
                grand_total["total_tokens"] += d.get("total_tokens", 0)

            now_iso = datetime.now(timezone.utc).isoformat()
            return {
                "last_updated_iso": now_iso,
                "grand_total": grand_total,
                "providers": {k: dict(v) for k, v in self._provider_usage.items()},
            }

    def _write_provider_usage_files(self) -> None:
        """Writes both machine-readable JSON and human-readable ledger tables."""
        summary = {
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
            "last_updated_unix": time.time(),
            "providers": self._provider_usage,
        }

        # 1. Write provider_token_usage.json
        try:
            with open(self.usage_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ [LLM LOGGER] Failed to write {self.usage_json_path}: {e}")

        # 2. Write provider_token_usage.log
        try:
            with open(self.usage_log_path, "w", encoding="utf-8") as f:
                f.write(self._format_provider_usage_ledger())
        except Exception as e:
            print(f"⚠️ [LLM LOGGER] Failed to write {self.usage_log_path}: {e}")

    def _format_provider_usage_ledger(self) -> str:
        """Formats a clean, stateful ledger showing token usage and credits across providers."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "=" * 98,
            "📊 SYSTEM 2 STATEFUL LLM TOKEN & CREDIT USAGE LEDGER (BY PROVIDER)",
            f"Last Updated: {now_iso}",
            "=" * 98,
            f"{'Provider':<14} {'Calls (OK/Err)':<18} {'Prompt Tokens':<16} {'Completion Tokens':<20} {'Total Tokens':<14} {'Last Active'}",
            "-" * 98,
        ]

        tot_calls = 0
        tot_ok = 0
        tot_err = 0
        tot_pt = 0
        tot_ct = 0
        tot_tt = 0

        # Sort standard providers first
        order = ["baseten", "groq", "fireworks"]
        all_keys = order + [k for k in sorted(self._provider_usage.keys()) if k not in order]

        for prov in all_keys:
            d = self._provider_usage.get(prov, self._make_empty_provider_dict())
            c = d.get("total_calls", 0)
            ok = d.get("successful_calls", 0)
            err = d.get("failed_calls", 0)
            pt = d.get("prompt_tokens", 0)
            ct = d.get("completion_tokens", 0)
            tt = d.get("total_tokens", 0)
            last_ts = d.get("last_call_iso") or "Never"
            if len(last_ts) > 19:
                last_ts = last_ts[:19].replace("T", " ")

            tot_calls += c
            tot_ok += ok
            tot_err += err
            tot_pt += pt
            tot_ct += ct
            tot_tt += tt

            calls_str = f"{c} ({ok}/{err})"
            lines.append(
                f"{prov.upper():<14} {calls_str:<18} {pt:<16,d} {ct:<20,d} {tt:<14,d} {last_ts}"
            )

        lines.extend([
            "-" * 98,
            f"{'TOTAL':<14} {f'{tot_calls} ({tot_ok}/{tot_err})':<18} {tot_pt:<16,d} {tot_ct:<20,d} {tot_tt:<14,d}",
            "=" * 98,
            "\n📋 DETAILED MODEL BREAKDOWN BY PROVIDER:",
        ])

        for prov in all_keys:
            d = self._provider_usage.get(prov, {})
            models = d.get("models", {})
            lines.append(f"\n[{prov.upper()}] - {d.get('total_calls', 0)} calls, {d.get('total_tokens', 0):,d} tokens")
            if not models:
                lines.append("  • No calls recorded yet.")
            else:
                for m_name, m_stats in sorted(models.items()):
                    lines.append(
                        f"  • {m_name}: {m_stats.get('total_calls', 0)} calls | "
                        f"In: {m_stats.get('prompt_tokens', 0):,d} | "
                        f"Out: {m_stats.get('completion_tokens', 0):,d} | "
                        f"Total: {m_stats.get('total_tokens', 0):,d} tokens"
                    )

        lines.append("=" * 98 + "\n")
        return "\n".join(lines)

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
