"""Canonical OpenAI Chat Completions dialect.

Wire shape:
  - Thinking: top-level `reasoning_effort` field.
  - Reasoning surfacing: not exposed in the canonical OpenAI Chat Completions
    response (reasoning content is internal-only on most OpenAI models).
  - Cache fields: standard `prompt_tokens_details.cached_tokens` only.

When a caller passes `budget_tokens` without `effort`, this dialect applies
the inherited heuristic mapping and emits a TranslationNote at WARNING level -
neither OpenAI nor Groq publish per-effort token relationships, so the
translation is necessarily lossy.
"""

from __future__ import annotations

from typing import Any

from clients.llm.dialects.openai_chat_base import OpenAIChatBase
from clients.llm.thinking import TranslationNote
from clients.llm.types import EFFORT_LEVEL_ORDER, EffortLevel, ThinkingConfig


class OpenAIDialect(OpenAIChatBase):
    """OpenAI Chat Completions canonical dialect."""

    dialect_name = "openai"
    is_abstract = False
    native_thinking_fields = ("effort",)

    # Per-model effort ceilings. Models not listed support all effort levels.
    _MAX_EFFORT_PER_MODEL: dict[str, EffortLevel] = {
        # Add entries when a model ships with a documented effort cap.
    }

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        effort = thinking.effort
        if effort is None and thinking.budget_tokens is not None:
            applied = self._budget_to_effort_heuristic(thinking.budget_tokens)
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=applied,
                reason=(
                    f"OpenAI dialect lacks native budget support; applied "
                    f"heuristic monotonic mapping"
                ),
            ))
            effort = applied
        elif effort is not None and thinking.budget_tokens is not None:
            # Both set; effort wins. Note the discarded budget so operators see it.
            self._log_translation(TranslationNote(
                field="budget_tokens",
                requested=thinking.budget_tokens,
                applied=None,
                reason="OpenAI dialect uses effort natively; budget discarded",
            ))

        if effort is None:
            return

        clamped = self._clamp_effort_for_model(payload.get("model"), effort, original=effort)
        payload["reasoning_effort"] = clamped

    def _clamp_effort_for_model(
        self,
        model: object,
        requested: EffortLevel,
        *,
        original: EffortLevel,
    ) -> EffortLevel:
        if not isinstance(model, str):
            return requested
        ceiling = self._MAX_EFFORT_PER_MODEL.get(model)
        if ceiling is None:
            return requested
        ranking = EFFORT_LEVEL_ORDER
        if ranking.index(requested) <= ranking.index(ceiling):
            return requested
        self._log_translation(TranslationNote(
            field="effort",
            requested=original,
            applied=ceiling,
            reason=f"model {model!r} caps effort at {ceiling!r}",
        ))
        return ceiling
