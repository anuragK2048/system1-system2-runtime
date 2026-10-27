"""Logging package for System 1 and System 2 Runtime.

Provides:
- LLMLogger: Structured auditor for all LLM calls, latencies, tokens, inputs, and outputs.
- LLMCallRecord: Data model for logged LLM transactions.
- get_llm_logger: Global singleton accessor for the LLM logger.
"""

from runtime.logging.llm_logger import (
    LLMLogger,
    LLMCallRecord,
    get_llm_logger,
)

__all__ = [
    "LLMLogger",
    "LLMCallRecord",
    "get_llm_logger",
]
