"""Public LLMProvider entry point backed by provider-neutral dialects."""

from __future__ import annotations

import logging
import signal
from collections.abc import Mapping as MappingABC
from typing import Any, Generator

from config import config
from clients.llm.events import (
    ErrorEvent,
    GenerationCancelled,
    StreamEvent,
)
from clients.llm.dialects.base import Dialect, ProviderContextOverflowError
from clients.llm.dialects.anthropic import AnthropicDialect, anthropic_thinking_params
from clients.llm.dialect_registry import get_registry
from clients.llm.accounting import UsageAccountingPolicy, UsageAccountingService
from clients.llm.capabilities import Requirements
from clients.llm.lifecycle import LLMLifecycle
from clients.llm.resolver import ModelResolver, ModelSelection
from clients.llm.types import (
    DialectName,
    EffortLevel,
    Request,
    RequestMetadata,
    Result,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    coerce_effort,
)

from utils.llm_tap import toggle as _toggle_traffic_tap

signal.signal(signal.SIGUSR1, _toggle_traffic_tap)


class ContextOverflowError(Exception):
    """Raised when request exceeds the selected model context window."""

    def __init__(self, estimated_tokens: int | None, context_window: int | None, provider: str):
        self.estimated_tokens = estimated_tokens
        self.context_window = context_window
        self.provider = provider
        estimated_label = f"~{estimated_tokens}" if estimated_tokens is not None else "unknown"
        window_label = str(context_window) if context_window is not None else "unknown"
        super().__init__(
            f"Context overflow: {estimated_label} tokens vs {window_label} limit ({provider})"
        )


def build_batch_params(
    purpose: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build Anthropic Batch API params from internal_llm config.

    Anthropic-only: callers must select an internal_llm purpose whose
    dialect_name is 'anthropic'. Routing a non-Anthropic purpose through the
    batch path is rejected loudly here rather than left to surface as a
    confusing wire error.
    """
    from utils.user_context import get_internal_llm

    llm_cfg = get_internal_llm(purpose)
    if llm_cfg.dialect_name != "anthropic":
        raise AssertionError(
            f"build_batch_params is Anthropic-only; resolved purpose "
            f"'{purpose}' uses dialect '{llm_cfg.dialect_name}'"
        )
    params: dict[str, Any] = {
        "model": llm_cfg.model,
        "max_tokens": llm_cfg.max_tokens,
        "system": [{"type": "text", "text": system_prompt}],
        "messages": messages,
    }
    if llm_cfg.effort:
        thinking_params, _ = anthropic_thinking_params(
            model=llm_cfg.model,
            thinking=ThinkingConfig(effort=coerce_effort(llm_cfg.effort)),
        )
        params.update(thinking_params)
    return params


class LLMProvider:
    """Public LLM API for application code."""

    _logger = logging.getLogger("llm_provider")

    def __init__(
        self,
        temperature: float = config.api.temperature,
        timeout: int = config.api.timeout,
        api_key: str | None = None,
    ):
        self.logger = logging.getLogger("llm_provider")
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key
        self.resolver = ModelResolver()
        self.accounting_service = UsageAccountingService()

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        *,
        internal_llm: str | None = None,
        conversation_llm: str | None = None,
        dialect_name: DialectName | None = None,
        model: str | None = None,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | list[dict[str, Any]] | None = None,
        thinking: ThinkingConfig | None = None,
        effort: EffortLevel | None = None,
        thinking_tokens: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        container_id: str | None = None,
        allow_negative: bool = False,
        allow_provider_stall_fallback: bool = True,
    ) -> Result:
        """Generate a blocking response from the selected provider.

        Returns a final ``Result``. Callers that need incremental events should
        call ``stream_events()`` instead.
        """
        self._validate_messages(messages)
        resolved_thinking = self._coalesce_thinking(
            effort=effort,
            thinking_tokens=thinking_tokens,
            thinking=thinking,
        )
        selection = self._resolve_selection(
            internal_llm=internal_llm,
            conversation_llm=conversation_llm,
            dialect_name=dialect_name,
            endpoint_url=endpoint_url,
            model=model,
            max_tokens=max_tokens,
            effort=resolved_thinking.effort or effort,
        )
        request = self._build_request(
            messages,
            tools,
            selection=selection,
            system_prompt=system_prompt,
            thinking=resolved_thinking,
            temperature=temperature,
            container_id=container_id,
        )
        dialect = self._dialect_for_selection(selection, api_key=api_key)
        dialect.capabilities.validate(self._requirements_for_request(request), dialect.dialect_name)
        lifecycle = LLMLifecycle(
            accounting_policy=UsageAccountingPolicy.for_current_build(
                allow_negative=allow_negative,
            ),
            accounting_service=self.accounting_service,
        )
        try:
            fallback_factory = (
                lambda: self._fallback_dialect_and_request(
                    messages,
                    tools,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    container_id=container_id,
                )
            ) if allow_provider_stall_fallback else None
            return lifecycle.complete(
                request,
                dialect,
                fallback_factory=fallback_factory,
            )
        except ProviderContextOverflowError as error:
            raise ContextOverflowError(
                error.estimated_tokens,
                error.context_window,
                error.endpoint,
            ) from error

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        *,
        internal_llm: str | None = None,
        conversation_llm: str | None = None,
        dialect_name: DialectName | None = None,
        effort: EffortLevel | None = None,
        thinking_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        container_id: str | None = None,
        endpoint_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt: str | list[dict[str, Any]] | None = None,
        allow_negative: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        """Stream normalized LLM events."""
        try:
            self._validate_messages(messages)
            resolved_thinking = self._coalesce_thinking(
                effort=effort,
                thinking_tokens=thinking_tokens,
                thinking=thinking,
            )
            selection = self._resolve_selection(
                internal_llm=internal_llm,
                conversation_llm=conversation_llm,
                dialect_name=dialect_name,
                endpoint_url=endpoint_url,
                model=model,
                max_tokens=max_tokens,
                effort=resolved_thinking.effort or effort,
            )
            request = self._build_request(
                messages,
                tools,
                selection=selection,
                system_prompt=system_prompt,
                thinking=resolved_thinking,
                temperature=temperature,
                container_id=container_id,
            )
            dialect = self._dialect_for_selection(selection, api_key=api_key)
            dialect.capabilities.validate(self._requirements_for_request(request), dialect.dialect_name)
            lifecycle = LLMLifecycle(
                accounting_policy=UsageAccountingPolicy.for_current_build(
                    allow_negative=allow_negative,
                ),
                accounting_service=self.accounting_service,
            )
            for event in lifecycle.stream(
                request,
                dialect,
                fallback_factory=lambda: self._fallback_dialect_and_request(
                    messages,
                    tools,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    container_id=container_id,
                ),
            ):
                yield event
        except GenerationCancelled:
            raise
        except ProviderContextOverflowError as error:
            raise ContextOverflowError(
                error.estimated_tokens,
                error.context_window,
                error.endpoint,
            ) from error
        except Exception as error:
            self.logger.error("LLM API request failed: %s", error, exc_info=True)
            yield ErrorEvent(error=str(error))
            raise

    def _coalesce_thinking(
        self,
        *,
        effort: EffortLevel | None,
        thinking_tokens: int | None,
        thinking: ThinkingConfig | None,
    ) -> ThinkingConfig:
        """Collapse convenience kwargs and explicit ThinkingConfig at the boundary.

        Callers may pass either `thinking=ThinkingConfig(...)` directly OR the
        legacy convenience kwargs `effort=` / `thinking_tokens=`, but not both.
        Mixing raises ValueError so callers don't end up with one half of their
        intent silently discarded.
        """
        if thinking is not None:
            if effort is not None or thinking_tokens is not None:
                raise ValueError(
                    "Pass thinking= OR effort/thinking_tokens, not both"
                )
            return thinking
        if effort is None and thinking_tokens is None:
            return ThinkingConfig()
        return ThinkingConfig(effort=effort, budget_tokens=thinking_tokens)

    def _resolve_selection(
        self,
        *,
        internal_llm: str | None,
        conversation_llm: str | None,
        dialect_name: DialectName | None,
        endpoint_url: str | None,
        model: str | None,
        max_tokens: int | None,
        effort: EffortLevel | None,
    ) -> ModelSelection:
        return self.resolver.resolve(
            internal_llm=internal_llm,
            conversation_llm=conversation_llm,
            dialect_name=dialect_name,
            endpoint_url=endpoint_url,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
        )

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition | dict[str, Any]] | None,
        *,
        selection: ModelSelection,
        system_prompt: str | list[dict[str, Any]] | None,
        thinking: ThinkingConfig,
        temperature: float | None,
        container_id: str | None,
    ) -> Request:
        extracted_system, prepared_messages = self._prepare_messages(messages)
        request_system = system_prompt if system_prompt is not None else extracted_system

        # Merge the resolver's per-internal-LLM effort default into the caller's
        # thinking config when the caller didn't already specify effort. The
        # selection.effort field carries values configured in the internal_llm
        # DB table (e.g., 'extraction' purpose defaults to 'high').
        merged_thinking = thinking
        if thinking.effort is None and selection.effort is not None:
            merged_thinking = ThinkingConfig(
                effort=selection.effort,
                budget_tokens=thinking.budget_tokens,
            )

        return Request(
            messages=prepared_messages,
            system=request_system,
            tools=tuple(self._normalize_tools([] if tools is None else tools)),
            model=selection.model,
            max_tokens=selection.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            thinking=merged_thinking,
            container_id=container_id,
            metadata=RequestMetadata(
                endpoint_url=selection.endpoint_url,
                internal_llm_name=selection.internal_llm_name,
                conversation_llm_name=selection.conversation_llm_name,
            ),
        )

    def _normalize_tools(
        self,
        tools: list[ToolDefinition | dict[str, Any]],
    ) -> list[ToolDefinition]:
        definitions = []
        for tool in tools:
            definitions.append(tool if isinstance(tool, ToolDefinition) else ToolDefinition.from_mapping(tool))
        return definitions

    def _fallback_dialect_and_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition | dict[str, Any]] | None,
        *,
        system_prompt: str | list[dict[str, Any]] | None,
        temperature: float | None,
        container_id: str | None,
    ) -> tuple[Dialect, Request]:
        fallback_selection = self.resolver.fallback()
        fallback_request = self._build_request(
            messages,
            tools,
            selection=fallback_selection,
            system_prompt=system_prompt,
            thinking=ThinkingConfig(),
            temperature=temperature,
            container_id=container_id,
        )
        fallback_dialect = self._dialect_for_selection(fallback_selection, api_key=None)
        fallback_dialect.capabilities.validate(
            self._requirements_for_request(fallback_request),
            fallback_dialect.dialect_name,
        )
        return fallback_dialect, fallback_request

    def _requirements_for_request(self, request: Request) -> Requirements:
        # container_upload blocks are not a hard files-capability requirement:
        # the Anthropic dialect consumes them natively, and the OpenAI-family
        # dialects degrade them to a text warning during message conversion.
        return Requirements(
            container_reuse=bool(request.container_id),
        )

    def _dialect_for_selection(
        self,
        selection: ModelSelection,
        *,
        api_key: str | None,
    ) -> Dialect:
        resolved_api_key = self._select_api_key(selection, api_key=api_key)
        cls = get_registry().get(selection.dialect_name)
        return cls.from_selection(
            selection,
            api_key=resolved_api_key,
            timeout=self.timeout,
            artifact_sink=self._create_file_artifact_sink(),
        )

    def _select_api_key(
        self,
        selection: ModelSelection,
        *,
        api_key: str | None,
    ) -> str | None:
        if api_key is not None:
            return self._validate_api_key(api_key, "api_key (call kwarg)")
        if self.api_key is not None:
            return self._validate_api_key(self.api_key, "api_key (LLMProvider default)")
        return None  # dialect's from_selection resolves api_key_name via Vault

    def _validate_api_key(self, value: str, source: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PermissionError(f"{source} resolved to an empty API key")
        return value

    def create_files_manager(self):
        """Create the Anthropic Files API manager for upload/cleanup workflows."""
        from clients.files_manager import FilesManager
        from clients.vault_client import get_api_key

        api_key = (
            self._validate_api_key(self.api_key, "api_key")
            if self.api_key is not None
            else self._validate_api_key(get_api_key(config.api.api_key_name), config.api.api_key_name)
        )
        dialect = AnthropicDialect(
            api_key=api_key,
            timeout=self.timeout,
            artifact_sink=self._create_file_artifact_sink(),
        )
        return FilesManager(dialect.client)

    def _create_file_artifact_sink(self):
        from utils.artifact_store import UserArtifactStore

        return UserArtifactStore()

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system_content = None
        prepared_messages = []
        for message in messages:
            if message["role"] == "system":
                if system_content is not None:
                    raise ValueError("Multiple system messages found — only one is supported")
                system_content = message["content"]
            else:
                prepared_messages.append(message)
        return system_content, prepared_messages

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Cannot send empty messages list to LLM API")
        for index, message in enumerate(messages):
            if not isinstance(message, MappingABC):
                raise ValueError(f"LLM message {index} must be a mapping")
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"LLM message {index} has invalid role: {role!r}")

            if role == "assistant" and message.get("tool_calls"):
                self._validate_assistant_tool_calls(message["tool_calls"], index)
                content = message.get("content")
                if content is not None and content != "":
                    self._validate_message_content(content, index=index, role=role)
                continue

            if "content" not in message:
                raise ValueError(f"LLM message {index} ({role}) missing content")
            self._validate_message_content(message["content"], index=index, role=role)
            if role == "tool" and not message.get("tool_call_id"):
                raise ValueError(f"Tool message {index} missing tool_call_id")

    def _validate_assistant_tool_calls(self, tool_calls: Any, message_index: int) -> None:
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(f"Assistant message {message_index} tool_calls must be a non-empty list")
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, MappingABC):
                raise ValueError(f"Assistant message {message_index} tool_call {call_index} must be a mapping")
            if not isinstance(tool_call.get("id"), str) or not tool_call["id"].strip():
                raise ValueError(f"Assistant message {message_index} tool_call {call_index} missing id")
            function = tool_call.get("function")
            if not isinstance(function, MappingABC):
                raise ValueError(f"Assistant message {message_index} tool_call {call_index} missing function")
            if not isinstance(function.get("name"), str) or not function["name"].strip():
                raise ValueError(f"Assistant message {message_index} tool_call {call_index} missing function name")
            arguments = function.get("arguments") if "arguments" in function else None
            if arguments is not None and not isinstance(arguments, (str, MappingABC)):
                raise ValueError(
                    f"Assistant message {message_index} tool_call {call_index} arguments must be a JSON string or mapping"
                )

    def _validate_message_content(self, content: Any, *, index: int, role: str) -> None:
        if isinstance(content, str):
            if not content.strip():
                raise ValueError(f"Cannot send empty {role} message to LLM API")
            return
        if not isinstance(content, list) or not content:
            raise ValueError(f"Cannot send empty or invalid {role} message to LLM API")
        for block_index, block in enumerate(content):
            if not isinstance(block, MappingABC):
                raise ValueError(f"Content block {block_index} in message {index} must be a mapping")
            block_type = block.get("type")
            if not isinstance(block_type, str) or not block_type:
                raise ValueError(f"Content block {block_index} in message {index} missing type")
            if block_type == "text":
                if not isinstance(block.get("text"), str) or not block["text"].strip():
                    raise ValueError(f"text block in message {index}, block {block_index} missing text")
            elif block_type == "reasoning":
                if not isinstance(block.get("text"), str) or not block["text"].strip():
                    raise ValueError(f"reasoning block in message {index}, block {block_index} missing text")
            elif block_type == "tool_call":
                if not isinstance(block.get("id"), str) or not block["id"].strip():
                    raise ValueError(f"tool_call block in message {index}, block {block_index} missing id")
                if not isinstance(block.get("name"), str) or not block["name"].strip():
                    raise ValueError(f"tool_call block in message {index}, block {block_index} missing name")
                if not isinstance(block.get("input"), MappingABC):
                    raise ValueError(f"tool_call block in message {index}, block {block_index} input must be a mapping")
            elif block_type == "file_ref":
                if not isinstance(block.get("file_id"), str) or not block["file_id"].strip():
                    raise ValueError(
                        f"file_ref block in message {index}, block {block_index} "
                        "is missing required file_id field"
                    )
            elif block_type in {"image", "document"}:
                if not isinstance(block.get("media_type"), str) or not block["media_type"]:
                    raise ValueError(f"{block_type} block in message {index}, block {block_index} missing media_type")
                if not isinstance(block.get("data"), str) or not block["data"]:
                    raise ValueError(f"{block_type} block in message {index}, block {block_index} missing data")
            else:
                raise ValueError(f"Unsupported content block type in message {index}, block {block_index}: {block_type}")

    def extract_text_content(self, response: Result) -> str:
        return response.text

    def extract_thinking_content(self, response: Result) -> str:
        if not response.reasoning:
            return ""
        return response.reasoning.text

    def extract_tool_calls(self, response: Result) -> list[ToolCall]:
        return list(response.tool_calls)
