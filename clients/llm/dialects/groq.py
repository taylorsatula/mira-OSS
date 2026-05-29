"""Groq dialect for the provider-neutral LLM boundary.

Wire shape:
  - Thinking: top-level `reasoning_effort` (categorical only). When reasoning
    is active, also sets `reasoning_format: "parsed"` so Groq returns reasoning
    in `delta.reasoning` / `message.reasoning` rather than wrapping it inline
    with `<think>` tags.
  - Reasoning surfacing: `message.reasoning` (non-streaming) and
    `delta.reasoning` (streaming).
  - Cache fields: standard `prompt_tokens_details.cached_tokens` only.

When a caller passes only `budget_tokens`, the heuristic inherited from
OpenAIChatBase maps it to the nearest effort level and emits a
TranslationNote at WARNING.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any

from clients.llm.dialects.openai_chat_base import OpenAIChatBase
from clients.llm.thinking import TranslationNote
from clients.llm.types import ThinkingConfig


class GroqDialect(OpenAIChatBase):
    """Groq Chat Completions dialect (top-level reasoning_effort + reasoning_format)."""

    dialect_name = "groq"
    is_abstract = False
    native_thinking_fields = ("effort",)

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        effort = thinking.effort
        if effort is None and thinking.budget_tokens is not None:
            applied = self._budget_to_effort_heuristic(thinking.budget_tokens)
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=applied,
                reason=(
                    f"Groq dialect lacks native budget support; applied "
                    f"heuristic monotonic mapping"
                ),
            ))
            effort = applied
        elif effort is not None and thinking.budget_tokens is not None:
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=None,
                reason="Groq dialect uses effort natively; budget discarded",
            ))

        if effort is None:
            return

        payload["reasoning_effort"] = effort
        # Ask Groq to return reasoning as a separate field rather than wrapping
        # it inline in the assistant message with <think> tags.
        payload["reasoning_format"] = "parsed"

    def _extract_reasoning_message(self, message: MappingABC[str, Any]) -> str:
        candidate = message.get("reasoning")
        if isinstance(candidate, str):
            return candidate
        return ""

    def _extract_reasoning_delta(self, delta: MappingABC[str, Any]) -> str:
        reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        return ""
