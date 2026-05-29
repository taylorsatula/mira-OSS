"""Central model and dialect selection for LLMProvider."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import config
from clients.llm.types import DialectName, EffortLevel, coerce_dialect_name, coerce_effort


@dataclass(frozen=True)
class ModelSelection:
    """Immutable selection data for one LLM request."""

    dialect_name: DialectName
    model: str
    endpoint_url: str | None
    api_key_name: str | None
    max_tokens: int
    effort: EffortLevel | None
    internal_llm_name: str | None = None
    conversation_llm_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ModelSelection.model must be a non-empty string")
        if self.endpoint_url is not None and (
            not isinstance(self.endpoint_url, str) or not self.endpoint_url.strip()
        ):
            raise ValueError("ModelSelection.endpoint_url must be a non-empty string when provided")
        if self.api_key_name is not None and (
            not isinstance(self.api_key_name, str) or not self.api_key_name.strip()
        ):
            raise ValueError("ModelSelection.api_key_name must be a non-empty string when provided")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("ModelSelection.max_tokens must be a positive integer")
        if self.effort is not None:
            object.__setattr__(self, "effort", coerce_effort(self.effort))
        if self.internal_llm_name is not None and (
            not isinstance(self.internal_llm_name, str) or not self.internal_llm_name.strip()
        ):
            raise ValueError("ModelSelection.internal_llm_name must be a non-empty string when provided")
        if self.conversation_llm_name is not None and (
            not isinstance(self.conversation_llm_name, str) or not self.conversation_llm_name.strip()
        ):
            raise ValueError("ModelSelection.conversation_llm_name must be a non-empty string when provided")


class ModelResolver:
    """Resolves model and dialect selection only. It does not fetch credentials."""

    def resolve(
        self,
        *,
        internal_llm: str | None = None,
        conversation_llm: str | None = None,
        endpoint_url: str | None = None,
        model: str | None = None,
        dialect_name: DialectName | None = None,
        max_tokens: int | None = None,
        effort: EffortLevel | None = None,
    ) -> ModelSelection:
        if internal_llm is not None:
            if (
                conversation_llm is not None
                or dialect_name is not None
                or endpoint_url is not None
                or model is not None
            ):
                raise ValueError(
                    "internal_llm= is mutually exclusive with conversation_llm, dialect_name, endpoint_url, and model"
                )
            return self._resolve_internal_llm(internal_llm, max_tokens=max_tokens, effort=effort)

        if conversation_llm is not None:
            if dialect_name is not None or endpoint_url is not None or model is not None:
                raise ValueError(
                    "conversation_llm= is mutually exclusive with dialect_name, endpoint_url, and model"
                )
            return self._resolve_conversation_llm(conversation_llm, max_tokens=max_tokens, effort=effort)

        if dialect_name is None:
            raise ValueError("dialect_name is required when not using internal_llm or conversation_llm")
        resolved_model = (
            self._optional_non_empty_string(model, "model")
            if model is not None
            else self._required_non_empty_string(config.api.model, "config.api.model")
        )
        resolved_endpoint = self._optional_non_empty_string(endpoint_url, "endpoint_url")
        return ModelSelection(
            dialect_name=coerce_dialect_name(dialect_name),
            model=resolved_model,
            endpoint_url=resolved_endpoint,
            api_key_name=None,
            max_tokens=max_tokens if max_tokens is not None else config.api.max_tokens,
            effort=effort,
        )

    def fallback(self, conversation_llm: str = "claude-high") -> ModelSelection:
        return self._resolve_conversation_llm(conversation_llm, max_tokens=None, effort=None)

    def _resolve_internal_llm(
        self,
        name: str,
        *,
        max_tokens: int | None,
        effort: EffortLevel | None,
    ) -> ModelSelection:
        from utils.user_context import get_internal_llm

        name = self._required_non_empty_string(name, "internal_llm")
        llm_cfg = get_internal_llm(name)
        return ModelSelection(
            dialect_name=coerce_dialect_name(llm_cfg.dialect_name),
            model=self._required_non_empty_string(llm_cfg.model, f"internal_llm '{name}' model"),
            endpoint_url=self._optional_non_empty_string(
                llm_cfg.endpoint_url,
                f"internal_llm '{name}' endpoint_url",
            ),
            api_key_name=self._optional_non_empty_string(
                llm_cfg.api_key_name,
                f"internal_llm '{name}' api_key_name",
            ),
            max_tokens=max_tokens if max_tokens is not None else llm_cfg.max_tokens,
            effort=effort if effort is not None else (
                coerce_effort(llm_cfg.effort) if llm_cfg.effort is not None else None
            ),
            internal_llm_name=self._required_non_empty_string(llm_cfg.name, "internal_llm.name"),
        )

    def _resolve_conversation_llm(
        self,
        name: str,
        *,
        max_tokens: int | None,
        effort: EffortLevel | None,
    ) -> ModelSelection:
        from utils.user_context import resolve_conversation_llm

        name = self._required_non_empty_string(name, "conversation_llm")
        llm_cfg = resolve_conversation_llm(name)
        api_key_name = self._optional_non_empty_string(
            llm_cfg.api_key_name,
            f"conversation_llm '{name}' api_key_name",
        )
        return ModelSelection(
            dialect_name=coerce_dialect_name(llm_cfg.dialect_name),
            model=self._required_non_empty_string(llm_cfg.model, f"conversation_llm '{name}' model"),
            endpoint_url=self._optional_non_empty_string(
                llm_cfg.endpoint_url,
                f"conversation_llm '{name}' endpoint_url",
            ),
            api_key_name=api_key_name,
            max_tokens=max_tokens if max_tokens is not None else config.api.max_tokens,
            effort=effort,
            conversation_llm_name=self._required_non_empty_string(llm_cfg.name, "conversation_llm.name"),
        )

    def _required_non_empty_string(self, value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    def _optional_non_empty_string(self, value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        return self._required_non_empty_string(value, field_name)
