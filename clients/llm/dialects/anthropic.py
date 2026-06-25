"""Anthropic dialect for the provider-neutral LLM boundary."""

from __future__ import annotations

import copy
import logging
import random
import time
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

import anthropic
import httpx

from clients.llm.artifacts import FileArtifactSink
from clients.llm.events import (
    CompleteEvent,
    FileArtifactEvent,
    StreamEvent,
    TextEvent,
    ThinkingEvent,
    ToolCompletedEvent,
    ToolDetectedEvent,
    ToolExecutingEvent,
)
from clients.llm.dialects.base import (
    Dialect,
    ProviderAuthError,
    ProviderContextOverflowError,
    ProviderProtocolError,
    ProviderRetryableError,
)
from clients.llm.capabilities import Capabilities
from clients.llm.thinking import TranslationNote, uses_adaptive_thinking
from clients.llm.types import (
    EffortLevel,
    EFFORT_LEVEL_ORDER,
    ProviderMetadata,
    ReasoningArtifact,
    ReasoningEntry,
    Request,
    Result,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    Usage,
    coerce_stop_reason,
)

if TYPE_CHECKING:
    from clients.llm.resolver import ModelSelection

logger = logging.getLogger(__name__)

OVERLOAD_MAX_RETRIES = 3
OVERLOAD_BASE_DELAY = 1.0
OVERLOAD_MAX_DELAY = 8.0
CODE_EXECUTION_BETA_FLAG = "code-execution-2025-08-25"
FILES_API_BETA_FLAG = "files-api-2025-04-14"
CODE_EXECUTION = {"type": "code_execution_20250825", "name": "code_execution"}
ANTHROPIC_BETA_FLAGS = [CODE_EXECUTION_BETA_FLAG, FILES_API_BETA_FLAG]


def _apply_cache_control(anthro_block: dict[str, Any], neutral_block: dict[str, Any]) -> None:
    """Translate cache TTL hint from neutral block to Anthropic cache_control."""
    ttl = neutral_block.get("cache")
    if ttl:
        anthro_block["cache_control"] = {"type": "ephemeral", "ttl": ttl}


def is_overloaded_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return "overloaded" in error_text or "overloaded_error" in error_text


class AnthropicDialect(Dialect):
    """Anthropic Messages API dialect."""

    dialect_name = "anthropic"
    is_abstract = False
    capabilities = Capabilities(
        batch=True,
        files=True,
        container_reuse=True,
        server_code_execution=True,
    )
    # Anthropic accepts both effort (4.6+ adaptive) and budget_tokens (legacy)
    native_thinking_fields = ("effort", "budget")
    _accepted_round_trip_fields = frozenset({"thinking_signatures"})

    # Documented budget_tokens per effort level for legacy thinking. Adaptive
    # thinking on 4.6+ uses effort natively and does not consult this map for
    # the forward direction; the inverse (budget->effort) is used when a caller
    # passes only budget against an adaptive-thinking model.
    _LEGACY_BUDGET_PER_EFFORT: dict[EffortLevel, int] = {
        "low": 1024,
        "medium": 2048,
        "high": 8192,
        "xhigh": 16000,
        "max": 31999,
    }

    # Per-model effort ceilings. Models not listed here support all effort
    # levels. No entries today - Anthropic effort caps live inline in each
    # model's public docs. Add entries here when a model ships with a ceiling.
    _MAX_EFFORT_PER_MODEL: dict[str, EffortLevel] = {}

    def __init__(
        self,
        *,
        api_key: str,
        timeout: int = 60,
        client: anthropic.Anthropic | None = None,
        artifact_sink: FileArtifactSink | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise PermissionError("Anthropic api_key must be a non-empty string")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("Anthropic timeout must be a positive integer")
        self.api_key = api_key
        self.timeout = timeout
        self.endpoint_url = "anthropic"
        self.artifact_sink = artifact_sink
        self._active_stream: Any = None
        self.client = client or anthropic.Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        )
        from utils.logging_config import instrument_anthropic_client

        instrument_anthropic_client(self.client)

    @classmethod
    def from_selection(
        cls,
        selection: "ModelSelection",
        *,
        api_key: str | None,
        timeout: int,
        artifact_sink: FileArtifactSink | None,
    ) -> "AnthropicDialect":
        if api_key is None:
            if selection.api_key_name is None:
                raise PermissionError(
                    f"Anthropic API key is required for model '{selection.model}' "
                    f"(no api_key_name configured on the selection)"
                )
            from clients.vault_client import get_api_key

            api_key = get_api_key(selection.api_key_name)
        if not isinstance(api_key, str) or not api_key.strip():
            raise PermissionError(
                f"Anthropic API key is required for model '{selection.model}' "
                f"(key name: {selection.api_key_name})"
            )
        return cls(api_key=api_key, timeout=timeout, artifact_sink=artifact_sink)

    def complete(self, request: Request) -> Result:
        params = self._build_params(request)
        message = self._call_with_overload_retry(
            lambda: self.client.beta.messages.create(**params, betas=ANTHROPIC_BETA_FLAGS),
            mode="non-streaming",
        )
        result = self._normalize_message(message, request)
        self._log_response(result, request.model)
        return result

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        params = self._build_params(request)
        tool_uses_seen: set[str] = set()
        final_message = None

        def run_stream():
            return self.client.beta.messages.stream(**params, betas=ANTHROPIC_BETA_FLAGS)

        for attempt in range(OVERLOAD_MAX_RETRIES):
            try:
                with run_stream() as stream:
                    self._active_stream = stream
                    try:
                        for event in stream:
                            if event.type == "text":
                                yield TextEvent(content=event.text)
                            elif event.type == "content_block_delta":
                                delta = getattr(event, "delta", None)
                                if getattr(delta, "type", None) == "thinking_delta":
                                    yield ThinkingEvent(content=delta.thinking)
                            elif event.type == "content_block_start":
                                block = event.content_block
                                if block.type == "tool_use" and block.id not in tool_uses_seen:
                                    tool_uses_seen.add(block.id)
                                    yield ToolDetectedEvent(tool_name=block.name, tool_id=block.id)
                                elif block.type == "server_tool_use":
                                    yield ToolDetectedEvent(tool_name=block.name, tool_id=block.id)
                                    yield ToolExecutingEvent(
                                        tool_name=block.name,
                                        tool_id=block.id,
                                        arguments=getattr(block, "input", {}) or {},
                                    )
                        final_message = stream.get_final_message()
                    finally:
                        self._active_stream = None
                break
            except anthropic.APIStatusError as error:
                if is_overloaded_error(error) and attempt < OVERLOAD_MAX_RETRIES - 1:
                    delay = min(OVERLOAD_BASE_DELAY * (2 ** attempt), OVERLOAD_MAX_DELAY)
                    time.sleep(delay * (0.5 + random.random()))
                    continue
                self._raise_anthropic_error(error, mode="streaming")
            except anthropic.APIError as error:
                raise ProviderRetryableError("anthropic", 500, "streaming", str(error)) from error

        if final_message is None:
            raise ProviderProtocolError("anthropic", "streaming", "Stream ended without final message")

        result = self._normalize_message(final_message, request)
        self._log_response(result, request.model)

        yield from self._server_tool_completed_events(final_message.content)
        yield from self._file_artifact_events(final_message.content)
        yield CompleteEvent(response=result)

    def current_partial_usage(self) -> Usage | None:
        stream = self._active_stream
        if stream is None:
            return None
        snapshot = getattr(stream, "current_message_snapshot", None)
        if snapshot is None:
            return None
        raw = getattr(snapshot, "usage", None)
        if raw is None:
            return None
        return Usage(
            input_tokens=getattr(raw, "input_tokens", 0) or 0,
            output_tokens=getattr(raw, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        )

    def _build_params(self, request: Request) -> dict[str, Any]:
        sanitized = [self.sanitize_outbound_message(m) for m in request.messages]
        messages = self._serialize_messages(sanitized, thinking_active=request.thinking.active)
        max_tokens = request.max_tokens
        thinking_params, thinking_adjustment = self._translate_thinking(
            model=request.model,
            thinking=request.thinking,
        )
        max_tokens += thinking_adjustment
        if "haiku" in request.model.lower() and max_tokens > 8192:
            max_tokens = 8192

        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if request.thinking.active:
            params.update(thinking_params)
        elif request.temperature is not None:
            params["temperature"] = request.temperature

        system = self._system_param(request.system)
        if system is not None:
            params["system"] = system

        tools = self._anthropic_tools(request.tools)
        if tools:
            params["tools"] = tools

        has_code_execution = any(
            tool.get("type") == CODE_EXECUTION["type"]
            for tool in tools
        )
        if request.container_id and has_code_execution:
            params["container"] = request.container_id
        elif request.container_id:
            logger.debug("Skipping container reuse because code_execution is not in tools")
        self._log_request(params, request.model)
        return params

    def _translate_thinking(
        self,
        *,
        model: str,
        thinking: ThinkingConfig,
    ) -> tuple[dict[str, Any], int]:
        """Translate ThinkingConfig into Anthropic API params.

        Adaptive thinking (Claude 4.6+) uses effort natively; legacy thinking
        uses budget_tokens natively. Both mappings between effort and budget
        are documented by Anthropic — cross-knob translation is canonical, not
        heuristic, so no TranslationNote is emitted for the conversion path.
        Notes ARE emitted when (a) both effort and budget_tokens are set and
        one is discarded, or (b) effort is clamped to a model's ceiling.
        """
        if not thinking.active:
            return {}, 0

        adaptive = uses_adaptive_thinking(model)
        effort = thinking.effort
        budget = thinking.budget_tokens

        if adaptive:
            # Native field is effort; legacy budget gets inverted to effort if needed.
            if effort is not None and budget is not None:
                # Both set; effort wins on adaptive models. Note the discarded budget.
                self._log_translation(TranslationNote(
                    field="budget_tokens",
                    requested=budget,
                    applied=None,
                    reason="adaptive model uses effort natively; budget discarded",
                ))
            if effort is None and budget is not None:
                effort = self._budget_to_effort_canonical(budget)
            resolved_effort = self._clamp_effort(model, effort or "high", original=effort)
            return (
                {
                    "thinking": {"type": "adaptive", "display": "summarized"},
                    "output_config": {"effort": resolved_effort},
                },
                0,
            )

        # Legacy thinking - native field is budget.
        if budget is None:
            # Caller provided only effort; translate canonically.
            clamped = self._clamp_effort(model, effort or "high", original=effort)
            budget = self._LEGACY_BUDGET_PER_EFFORT[clamped]
        elif effort is not None:
            # Both set; budget wins on legacy models. Note the discarded effort.
            self._log_translation(TranslationNote(
                field="effort",
                requested=effort,
                applied=None,
                reason="legacy model uses budget_tokens natively; effort discarded",
            ))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, budget

    def _budget_to_effort_canonical(self, budget_tokens: int) -> EffortLevel:
        """Invert the documented effort->budget map (closest documented level)."""
        # Pick the smallest documented level whose budget >= the requested budget.
        # EFFORT_LEVEL_ORDER is the canonical low-to-max ranking.
        for level in EFFORT_LEVEL_ORDER:
            if budget_tokens <= self._LEGACY_BUDGET_PER_EFFORT[level]:
                return level
        return "max"

    def _clamp_effort(
        self,
        model: str,
        requested: EffortLevel,
        *,
        original: EffortLevel | None,
    ) -> EffortLevel:
        """Clamp effort to per-model ceiling, logging a note when clamping occurs."""
        ceiling = self._MAX_EFFORT_PER_MODEL.get(model)
        if ceiling is None:
            return requested
        ranking = EFFORT_LEVEL_ORDER
        if ranking.index(requested) <= ranking.index(ceiling):
            return requested
        # Only emit a note when the caller asked for the clamped level (not
        # when we derived it from a budget); deriving stays silent because
        # the mapping is canonical.
        if original is not None:
            self._log_translation(TranslationNote(
                field="effort",
                requested=requested,
                applied=ceiling,
                reason=f"model {model!r} caps effort at {ceiling!r}",
            ))
        return ceiling

    def _system_param(self, system: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]] | None:
        if isinstance(system, list):
            # Translate cache TTL hints on system blocks to cache_control
            serialized: list[dict[str, Any]] = []
            for block in system:
                if block.get("type") == "text" and block.get("cache"):
                    anthro_block = {"type": "text", "text": block.get("text", "")}
                    _apply_cache_control(anthro_block, block)
                    serialized.append(anthro_block)
                else:
                    serialized.append(block)
            return serialized
        if isinstance(system, str) and system:
            return system
        return None

    def _anthropic_tools(self, tools: tuple[ToolDefinition, ...]) -> list[dict[str, Any]]:
        if not tools:
            return []
        serialized: list[dict[str, Any]] = []
        for tool in tools:
            tool_dict = tool.to_dict()
            if tool.cache is not None:
                tool_dict["cache_control"] = {"type": "ephemeral", "ttl": tool.cache}
                del tool_dict["cache"]
            serialized.append(tool_dict)
        serialized.insert(0, copy.deepcopy(CODE_EXECUTION))
        return serialized

    def _serialize_messages(
        self,
        messages: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        thinking_active: bool,
    ) -> list[dict[str, Any]]:
        """Convert neutral message format to Anthropic wire format."""
        serialized: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def _flush_tool_results() -> None:
            if not pending_tool_results:
                return
            serialized.append({
                "role": "user",
                "content": list(pending_tool_results),
            })
            pending_tool_results.clear()

        for message in messages:
            role = message.get("role")

            if role == "tool":
                # Buffer tool results for Anthropic's required user-message packing
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": self._serialize_tool_content(message.get("content")),
                    **({"is_error": True} if message.get("is_error") else {}),
                })
                continue

            _flush_tool_results()

            content = message.get("content")

            if role == "assistant":
                serialized.append({
                    "role": "assistant",
                    "content": self._serialize_assistant_content(content, message, thinking_active),
                })
            elif role == "user":
                serialized.append({
                    "role": "user",
                    "content": self._serialize_user_content(content),
                })
            else:
                serialized.append({"role": role, "content": content})

        _flush_tool_results()

        return serialized

    def _serialize_assistant_content(
        self,
        content: Any,
        message: dict[str, Any],
        thinking_active: bool,
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        if not isinstance(content, list):
            return content

        # Reconstruct thinking blocks from ReasoningBlocks + thinking_signatures metadata
        thinking_signatures = message.get("thinking_signatures", [])
        sig_index = 0

        result: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "reasoning":
                if not thinking_active:
                    continue
                text = block.get("text", "")
                anthro_block: dict[str, Any] = {"type": "thinking", "thinking": text}
                matched = False
                while sig_index < len(thinking_signatures):
                    sig = thinking_signatures[sig_index]
                    sig_index += 1
                    if sig.get("type") == "thinking":
                        anthro_block["signature"] = sig["signature"]
                        matched = True
                        break
                    elif sig.get("type") == "redacted_thinking":
                        result.append({"type": "redacted_thinking", "data": sig["data"]})
                if not matched:
                    continue
                result.append(anthro_block)
            elif block_type == "tool_call":
                result.append({
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
            elif block_type == "text":
                anthro_block = {"type": "text", "text": block.get("text", "")}
                _apply_cache_control(anthro_block, block)
                result.append(anthro_block)
            elif block_type == "image":
                result.append(self._serialize_inner_block(block))
            elif block_type == "document":
                result.append(self._serialize_inner_block(block))

        while sig_index < len(thinking_signatures):
            sig = thinking_signatures[sig_index]
            sig_index += 1
            if sig.get("type") == "redacted_thinking":
                result.append({"type": "redacted_thinking", "data": sig["data"]})

        return result

    def _serialize_user_content(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content

        result: list[object] = []
        for block in content:
            if not isinstance(block, dict):
                result.append(block)
                continue
            block_type = block.get("type", "")

            if block_type in {"image", "document", "file_ref", "text"}:
                result.append(self._serialize_inner_block(block, apply_cache_control=True))
            else:
                result.append(block)

        return result

    def _serialize_tool_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        return [self._serialize_inner_block(block) for block in content]

    def _serialize_inner_block(
        self,
        block: object,
        *,
        apply_cache_control: bool = False,
    ) -> object:
        if not isinstance(block, dict):
            return block
        block_type = block.get("type", "")

        if block_type == "image":
            anthro_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.get("media_type", ""),
                    "data": block.get("data", ""),
                },
            }
            if apply_cache_control:
                _apply_cache_control(anthro_block, block)
            return anthro_block
        elif block_type == "document":
            anthro_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.get("media_type", ""),
                    "data": block.get("data", ""),
                },
            }
            if apply_cache_control:
                _apply_cache_control(anthro_block, block)
            return anthro_block
        elif block_type == "file_ref":
            anthro_block = {
                "type": "container_upload",
                "file_id": block.get("file_id", ""),
            }
            if apply_cache_control:
                _apply_cache_control(anthro_block, block)
            return anthro_block
        elif block_type == "text":
            anthro_block = {"type": "text", "text": block.get("text", "")}
            if apply_cache_control:
                _apply_cache_control(anthro_block, block)
            return anthro_block
        return block

    def _call_with_overload_retry(self, operation, *, mode: str):
        for attempt in range(OVERLOAD_MAX_RETRIES):
            try:
                return operation()
            except anthropic.APIStatusError as error:
                if is_overloaded_error(error) and attempt < OVERLOAD_MAX_RETRIES - 1:
                    delay = min(OVERLOAD_BASE_DELAY * (2 ** attempt), OVERLOAD_MAX_DELAY)
                    time.sleep(delay * (0.5 + random.random()))
                    continue
                self._raise_anthropic_error(error, mode=mode)
            except anthropic.APITimeoutError as error:
                raise ProviderRetryableError("anthropic", 504, mode, "Request timed out") from error
            except anthropic.APIError as error:
                raise ProviderRetryableError("anthropic", 500, mode, str(error)) from error

    def _normalize_message(self, message: Any, request: Request | None) -> Result:
        text_parts = []
        entries: list[ReasoningEntry] = []
        redacted_data = []
        tool_calls = []
        recognized_types = {"text", "thinking", "redacted_thinking", "tool_use"}

        for block in getattr(message, "content", []) or []:
            block_type = block.type
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "thinking":
                entries.append(ReasoningEntry(
                    text=block.thinking,
                    signature=getattr(block, "signature", None),
                ))
            elif block_type == "redacted_thinking":
                redacted_data.append(block.data)
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    tool_name=block.name,
                    input=block.input,
                ))
            else:
                # server_tool_use and code_execution_tool_result are server-side
                # execution blocks — handled by _server_tool_completed_events() and
                # _file_artifact_events(). Log them so operators see what was skipped.
                logger.warning(
                    "Anthropic _normalize_message skipped response block type '%s' — "
                    "this block type does not contribute to Result text, reasoning, or tool_calls",
                    block_type,
                )

        usage = None
        if getattr(message, "usage", None):
            raw_usage = message.usage
            usage = Usage(
                input_tokens=raw_usage.input_tokens,
                output_tokens=raw_usage.output_tokens,
                cache_creation_input_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            )

        container_id = None
        if getattr(message, "container", None):
            container_id = message.container.id
        elif request and request.container_id:
            container_id = request.container_id

        reasoning = None
        if entries or redacted_data:
            reasoning = ReasoningArtifact(
                entries=tuple(entries),
                redacted_data=tuple(redacted_data),
            )

        return Result(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            reasoning=reasoning,
            usage=usage,
            stop_reason=coerce_stop_reason(
                getattr(message, "stop_reason", "end_turn") or "end_turn"
            ),
            provider_metadata=ProviderMetadata(
                dialect_name=self.dialect_name,
            ),
            container_id=container_id,
        )

    def _server_tool_completed_events(self, content_blocks: list) -> Iterator[ToolCompletedEvent]:
        from anthropic.types.beta import BetaCodeExecutionResultBlock

        server_tool_map = {}
        for block in content_blocks:
            if block.type == "server_tool_use":
                server_tool_map[block.id] = block.name

        for block in content_blocks:
            if block.type != "code_execution_tool_result":
                continue
            tool_id = getattr(block, "tool_use_id", "")
            tool_name = server_tool_map.get(tool_id, "code_execution")
            if isinstance(block.content, BetaCodeExecutionResultBlock):
                inner = block.content
                file_count = sum(1 for output in inner.content if getattr(output, "file_id", None))
                result_summary = (
                    f"[code_execution: rc={inner.return_code}, "
                    f"stdout={len(inner.stdout or '')}B, stderr={len(inner.stderr or '')}B, "
                    f"files={file_count}]"
                )
            else:
                error_code = getattr(block.content, "error_code", "unknown")
                result_summary = f"[code_execution error: {error_code}]"
            yield ToolCompletedEvent(tool_name=tool_name, tool_id=tool_id, result=result_summary)

    def _file_artifact_events(self, content_blocks: list) -> Iterator[FileArtifactEvent]:
        from anthropic.types.beta import BetaCodeExecutionResultBlock

        for block in content_blocks:
            if block.type != "code_execution_tool_result":
                continue
            if not isinstance(block.content, BetaCodeExecutionResultBlock):
                continue
            for output in block.content.content:
                file_id = getattr(output, "file_id", None)
                if not file_id:
                    continue
                try:
                    metadata = self.client.beta.files.retrieve_metadata(
                        file_id,
                        betas=[FILES_API_BETA_FLAG],
                    )
                    if self.artifact_sink is None:
                        raise RuntimeError("File artifact sink is required for code execution artifacts")
                    response = self.client.beta.files.download(file_id, betas=[FILES_API_BETA_FLAG])
                    self.artifact_sink.save_file_artifact(
                        file_id=file_id,
                        filename=metadata.filename,
                        mime_type=metadata.mime_type,
                        size_bytes=metadata.size_bytes,
                        content=response.read(),
                    )
                    yield FileArtifactEvent(
                        file_id=file_id,
                        filename=metadata.filename,
                        mime_type=metadata.mime_type,
                        size_bytes=metadata.size_bytes,
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to process file artifact %s: %s",
                        file_id,
                        error,
                    )
                    continue

    def _raise_anthropic_error(self, error: anthropic.APIStatusError, *, mode: str):
        status_code = error.status_code
        message = str(error)
        if status_code == 400:
            lowered = message.lower()
            if "prompt is too long" in lowered or "context" in lowered or "too many tokens" in lowered:
                raise ProviderContextOverflowError("anthropic", mode, message)
        if status_code in (401, 403):
            raise ProviderAuthError("anthropic", mode, message)
        if status_code == 429 or status_code >= 500:
            raise ProviderRetryableError("anthropic", status_code, mode, message)
        raise ProviderProtocolError("anthropic", mode, f"Anthropic API error ({status_code}): {message}")

    def _log_request(self, params: dict[str, Any], model: str) -> None:
        from utils.llm_tap import is_active as _tap_active, log_request as _tap_request

        if _tap_active():
            _tap_request(provider=self.dialect_name, endpoint="anthropic", model=model, body=params)

    def _log_response(self, result: Result, model: str) -> None:
        from utils.llm_tap import is_active as _tap_active, log_response as _tap_response

        if _tap_active():
            _tap_response(provider=self.dialect_name, model=model, response_data=result)


def anthropic_thinking_params(
    *,
    model: str,
    thinking: ThinkingConfig,
) -> tuple[dict[str, Any], int]:
    """Translate a ThinkingConfig into Anthropic API params.

    Module-level entry point for code paths that don't construct a full
    AnthropicDialect (build_batch_params and the Anthropic batch transport
    in agents/batch.py). Translation logic is identical to the dialect's
    _translate_thinking method.
    """
    dialect = object.__new__(AnthropicDialect)
    return AnthropicDialect._translate_thinking(dialect, model=model, thinking=thinking)


def normalize_anthropic_message(message: Any, request: Request | None = None) -> Result:
    """Normalize an Anthropic SDK Message without constructing a transport client."""
    dialect = object.__new__(AnthropicDialect)
    return AnthropicDialect._normalize_message(dialect, message, request)
