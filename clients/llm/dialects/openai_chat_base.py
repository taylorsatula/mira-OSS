"""Shared OpenAI Chat Completions transport for OpenAI-family dialects.

This base class owns the genuinely shared mechanics: message conversion,
tool-schema serialization, streaming SSE framework, usage parsing, and HTTP
error normalization. Subclasses (OpenAIDialect, OpenRouterDialect, GroqDialect)
override hook methods for the parts that diverge:

  - _serialize_thinking: how to encode the caller's ThinkingConfig into the
    request payload (top-level reasoning_effort, nested reasoning block, etc).
  - _extract_reasoning_message / _extract_reasoning_delta: which fields the
    provider uses to surface reasoning content.
  - _extract_cache_usage: which prompt_tokens_details fields carry cache
    write/read counts.
  - _accepted_round_trip_fields: which provider-specific round-trip fields
    (e.g., OpenRouter's reasoning_details) to copy onto the wire. Declared
    on the base Dialect class; each dialect whitelists only the fields its
    provider natively accepts.

The base class is not directly instantiable — it raises NotImplementedError
from from_selection. Discovery in clients.llm.dialect_registry intentionally
skips this module by name.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from collections.abc import Mapping as MappingABC
from typing import Any, NoReturn, TYPE_CHECKING

import httpx

from clients.llm.artifacts import FileArtifactSink
from clients.llm.events import CompleteEvent, StreamEvent, TextEvent, ThinkingEvent, ToolDetectedEvent
from clients.llm.dialects.base import (
    Dialect,
    ProviderAuthError,
    ProviderContextOverflowError,
    ProviderProtocolError,
    ProviderRetryableError,
)
from clients.llm.types import (
    EffortLevel,
    ProviderMetadata,
    ReasoningArtifact,
    ReasoningEntry,
    Request,
    Result,
    StopReason,
    ThinkingConfig,
    ToolCall,
    ToolDefinition,
    Usage,
)

if TYPE_CHECKING:
    from clients.llm.resolver import ModelSelection

logger = logging.getLogger(__name__)


class ToolNotLoadedError(ProviderProtocolError):
    """Provider reported a tool call for a tool not included in the request."""

    def __init__(self, endpoint: str, mode: str, tool_name: str, original_message: str):
        self.tool_name = tool_name
        self.original_message = original_message
        super().__init__(
            endpoint,
            mode,
            f"Tool '{tool_name}' not loaded by provider: {original_message}",
        )


class OpenAIChatBase(Dialect):
    """Abstract base for OpenAI-compatible Chat Completions dialects.

    Not a concrete dialect — it has no dialect_name override and the registry
    skips this module during discovery.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        if not isinstance(endpoint_url, str) or not endpoint_url.strip():
            raise ValueError(f"{type(self).__name__} endpoint_url must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise PermissionError(f"{type(self).__name__} api_key must be a non-empty string when provided")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError(f"{type(self).__name__} timeout must be a positive integer")
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_selection(
        cls,
        selection: "ModelSelection",
        *,
        api_key: str | None,
        timeout: int,
        artifact_sink: FileArtifactSink | None,
    ) -> "OpenAIChatBase":
        # Shared construction for OpenAI-family dialects: pull endpoint from
        # the selection, source API key from the argument or Vault.
        if not selection.endpoint_url:
            raise ValueError(
                f"{cls.__name__} requires endpoint_url on the ModelSelection"
            )
        if api_key is None and selection.api_key_name:
            from clients.vault_client import get_api_key

            api_key = get_api_key(selection.api_key_name)
        return cls(endpoint_url=selection.endpoint_url, api_key=api_key, timeout=timeout)

    def _log_request(self, request: Request, payload: dict[str, Any]) -> None:
        from utils.llm_tap import is_active as _tap_active, log_request as _tap_request

        if _tap_active():
            _tap_request(
                provider=self.dialect_name,
                endpoint=self.endpoint_url,
                model=request.model,
                body=payload,
            )

    def _log_response(self, request: Request, result: Result) -> None:
        from utils.llm_tap import is_active as _tap_active, log_response as _tap_response

        if _tap_active():
            _tap_response(
                provider=self.dialect_name,
                model=request.model,
                response_data=result,
                endpoint=self.endpoint_url,
            )

    # ------------------------------------------------------------------
    # Hooks: subclasses override these for dialect-specific behavior.
    # ------------------------------------------------------------------

    def _serialize_thinking(self, payload: dict[str, Any], thinking: ThinkingConfig) -> None:
        """Encode the caller's ThinkingConfig into the request payload.

        Default no-op — base does not assume any thinking knob. Subclasses
        override to set reasoning_effort, nested reasoning blocks, or both.
        """

    def _extract_reasoning_message(self, message: MappingABC[str, Any]) -> str:
        """Extract reasoning text from a non-streaming response message."""
        return ""

    def _extract_reasoning_delta(self, delta: MappingABC[str, Any]) -> str:
        """Extract reasoning text from a streaming delta. Default: empty."""
        return ""

    def _extract_cache_usage(self, prompt_details: MappingABC[str, Any]) -> tuple[int, int]:
        """Return (cache_write_tokens, cache_read_tokens) from prompt_tokens_details."""
        return (0, self._optional_usage_token(prompt_details, "cached_tokens"))

    @staticmethod
    def _budget_to_effort_heuristic(budget_tokens: int) -> EffortLevel:
        """Best-effort monotonic mapping of budget hint to effort category.

        Heuristic — neither OpenAI nor Groq publish per-effort token budgets,
        so this is a documented guess. Dialects whose provider DOES publish
        thresholds should override this method with the documented values.
        """
        if budget_tokens <= 2048:
            return "low"
        if budget_tokens <= 8192:
            return "medium"
        if budget_tokens <= 16000:
            return "high"
        if budget_tokens <= 32000:
            return "xhigh"
        return "max"

    # ------------------------------------------------------------------
    # Transport: non-streaming completion.
    # ------------------------------------------------------------------

    def complete(self, request: Request) -> Result:
        payload = self._build_payload(request, stream=False)
        headers = self._headers()
        self._log_request(request, payload)

        response = httpx.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, mode="non-streaming")

        result = self._parse_response(
            self._decode_response_json(response, "non-streaming"),
            request,
        )
        self._log_response(request, result)
        return result

    # ------------------------------------------------------------------
    # Transport: streaming SSE framework.
    # ------------------------------------------------------------------

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        from utils import http_client

        payload = self._build_payload(request, stream=True)
        headers = self._headers()
        self._log_request(request, payload)

        accumulated_text = ""
        accumulated_reasoning = ""
        accumulated_reasoning_details: list[Any] = []
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Usage | None = None
        detected_tool_ids: set[str] = set()
        saw_reasoning_delta = False

        with http_client.stream(
            "POST",
            self.endpoint_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code >= 400:
                error_text = response.read().decode("utf-8", errors="replace")
                self._raise_provider_http_error(
                    status=response.status_code,
                    error_body=_json_or_none(error_text),
                    fallback_text=error_text,
                    mode="streaming",
                )

            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    raise ProviderProtocolError(
                        self.endpoint_url,
                        "streaming",
                        f"Malformed SSE JSON chunk: {line[:200]}",
                    )

                if chunk.get("usage"):
                    chunk_usage = self._parse_usage(chunk["usage"])
                    if usage is None:
                        usage = chunk_usage
                    else:
                        usage = Usage(
                            input_tokens=max(usage.input_tokens, chunk_usage.input_tokens),
                            output_tokens=max(usage.output_tokens, chunk_usage.output_tokens),
                            cache_creation_input_tokens=max(
                                usage.cache_creation_input_tokens,
                                chunk_usage.cache_creation_input_tokens,
                            ),
                            cache_read_input_tokens=max(
                                usage.cache_read_input_tokens,
                                chunk_usage.cache_read_input_tokens,
                            ),
                        )

                choices = chunk.get("choices")
                if not choices:
                    continue

                if not isinstance(choices, list):
                    raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE choices must be a list")
                choice = choices[0]
                if not isinstance(choice, MappingABC):
                    raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE choice must be an object")
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if not isinstance(delta, MappingABC):
                    raise ProviderProtocolError(self.endpoint_url, "streaming", "SSE delta must be an object")

                if delta.get("content"):
                    text = delta["content"]
                    if not isinstance(text, str):
                        raise ProviderProtocolError(
                            self.endpoint_url,
                            "streaming",
                            "SSE content delta must be a string",
                        )
                    accumulated_text += text
                    yield TextEvent(content=text)

                reasoning_text = self._extract_reasoning_delta(delta)
                if reasoning_text:
                    saw_reasoning_delta = True
                    reasoning_delta = self._new_reasoning_text(accumulated_reasoning, reasoning_text)
                    if reasoning_delta:
                        accumulated_reasoning += reasoning_delta
                        yield ThinkingEvent(content=reasoning_delta)

                if delta.get("reasoning_details"):
                    details = delta["reasoning_details"]
                    if not isinstance(details, list):
                        raise ProviderProtocolError(
                            self.endpoint_url,
                            "streaming",
                            "SSE reasoning_details delta must be a list",
                        )
                    accumulated_reasoning_details.extend(details)
                    if not saw_reasoning_delta:
                        details_text = self._reasoning_details_text(details)
                        reasoning_delta = self._new_reasoning_text(accumulated_reasoning, details_text)
                        if reasoning_delta:
                            accumulated_reasoning += reasoning_delta
                            yield ThinkingEvent(content=reasoning_delta)

                if delta.get("tool_calls"):
                    if not isinstance(delta["tool_calls"], list):
                        raise ProviderProtocolError(
                            self.endpoint_url,
                            "streaming",
                            "SSE tool_calls delta must be a list",
                        )
                    for tool_call_delta in delta["tool_calls"]:
                        if not isinstance(tool_call_delta, MappingABC):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE tool_call delta must be an object",
                            )
                        index = tool_call_delta.get("index")
                        if type(index) is not int:
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE tool_call delta index must be an integer",
                            )
                        state = accumulated_tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if tool_call_delta.get("id"):
                            if not isinstance(tool_call_delta["id"], str):
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call id delta must be a string",
                                )
                            state["id"] = tool_call_delta["id"]
                        function_delta = tool_call_delta.get("function") or {}
                        if not isinstance(function_delta, MappingABC):
                            raise ProviderProtocolError(
                                self.endpoint_url,
                                "streaming",
                                "SSE tool_call function delta must be an object",
                            )
                        if function_delta.get("name"):
                            if not isinstance(function_delta["name"], str):
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call name delta must be a string",
                                )
                            state["name"] = function_delta["name"]
                        if function_delta.get("arguments"):
                            if not isinstance(function_delta["arguments"], str):
                                raise ProviderProtocolError(
                                    self.endpoint_url,
                                    "streaming",
                                    "SSE tool_call arguments delta must be a string",
                                )
                            state["arguments"] += function_delta["arguments"]

                        if state["id"] and state["name"] and state["id"] not in detected_tool_ids:
                            detected_tool_ids.add(state["id"])
                            yield ToolDetectedEvent(
                                tool_name=state["name"],
                                tool_id=state["id"],
                            )

        if usage is None:
            logger.warning(
                "%s stream from %s ended without usage despite include_usage; "
                "billing will be skipped for model=%s",
                self.dialect_name,
                self.endpoint_url,
                request.model,
            )

        result = Result(
            text=accumulated_text,
            tool_calls=self._parse_stream_tool_calls(accumulated_tool_calls, request),
            reasoning=self._build_reasoning(
                reasoning_text=accumulated_reasoning,
                reasoning_details=accumulated_reasoning_details or None,
            ),
            usage=usage,
            stop_reason=self._normalize_finish_reason(finish_reason),
            provider_metadata=ProviderMetadata(
                dialect_name=self.dialect_name,
                endpoint_url=self.endpoint_url,
            ),
        )

        self._log_response(request, result)

        yield CompleteEvent(response=result)

    # ------------------------------------------------------------------
    # Payload construction.
    # ------------------------------------------------------------------

    def _build_payload(self, request: Request, *, stream: bool) -> dict[str, Any]:
        max_tokens = request.max_tokens
        if request.thinking.budget_tokens is not None:
            max_tokens += request.thinking.budget_tokens

        messages = []
        if request.system:
            messages.append(self._convert_system_prompt(request.system))
        sanitized = [self.sanitize_outbound_message(m) for m in request.messages]
        messages.extend(self._convert_messages(sanitized, request))

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
        if request.thinking.active:
            self._serialize_thinking(payload, request.thinking)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _convert_system_prompt(self, system: str | list[dict[str, Any]]) -> dict[str, str]:
        if isinstance(system, list):
            text = "".join(
                block.get("text", "")
                for block in system
                if block.get("type") == "text"
            )
            return {"role": "system", "content": text}
        return {"role": "system", "content": system}

    # ------------------------------------------------------------------
    # Message conversion.
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        request: Request,
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                tool_content = message.get("content")
                extracted_images: list[dict[str, Any]] = []
                tool_text = self._extract_text_from_content(tool_content, extracted_images)
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", ""),
                    "content": tool_text,
                })
                if extracted_images:
                    openai_messages.append({
                        "role": "user",
                        "content": extracted_images,
                    })
                continue

            content = message.get("content")
            if role == "user":
                openai_messages.extend(self._convert_user_message(content))
            elif role == "assistant":
                openai_messages.append(self._convert_assistant_message(message, content, request))
        return openai_messages

    def _convert_user_message(self, content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return [{"role": "user", "content": content}]

        content_parts: list[dict[str, Any]] = []
        dropped_block_types: list[str] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                content_parts.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                content_parts.append(self._convert_image_block(block))
            elif block_type == "file_ref":
                file_id = block.get("file_id", "unknown")
                content_parts.append({"type": "text", "text": f"[File upload not supported by this provider: {file_id}]"})
            elif block_type == "reasoning":
                dropped_block_types.append("reasoning")
            else:
                content_parts.append(block)

        if dropped_block_types:
            logger.warning(
                "OpenAI-family dialect dropped %d block(s) of type(s) %s from user message — "
                "these types have no equivalent in OpenAI Chat Completions",
                len(dropped_block_types),
                list(dict.fromkeys(dropped_block_types)),
            )

        if not content_parts:
            # Entire user message would vanish (e.g., message with only reasoning blocks).
            # Log loudly so operators see the data loss instead of discovering it as
            # a broken conversation later.
            logger.error(
                "OpenAI-family dialect produced empty user message from content with %d block(s) — "
                "user turn is being dropped entirely. Dropped types: %s",
                len(dropped_block_types),
                dropped_block_types,
            )
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                f"Cannot convert user message: all blocks stripped (types: {dropped_block_types}). "
                "OpenAI-family providers have no equivalent for these block types.",
            )

        has_image = any(part.get("type") == "image_url" for part in content_parts)
        if has_image:
            return [{"role": "user", "content": content_parts}]
        else:
            text_content = "".join(part.get("text", "") for part in content_parts if part.get("type") == "text")
            if text_content:
                return [{"role": "user", "content": text_content}]
            return []

    def _extract_text_from_content(
        self,
        content: Any,
        extracted_images: list[dict[str, Any]],
    ) -> str:
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, MappingABC):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "image":
                    extracted_images.append(self._convert_image_block(block))
            return "".join(text_parts)
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content if content is not None else "")

    def _convert_image_block(self, block: dict[str, Any]) -> dict[str, Any]:
        media_type = block.get("media_type")
        if not isinstance(media_type, str) or not media_type:
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                "Image block media_type must be a non-empty string",
            )
        data = block.get("data")
        if not isinstance(data, str) or not data:
            raise ProviderProtocolError(
                self.endpoint_url,
                "message-conversion",
                "Image block data must be a non-empty string",
            )
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
                "detail": "auto",
            },
        }

    def _convert_assistant_message(
        self,
        message: dict[str, Any],
        content: Any,
        request: Request,
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_call":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                self._coerce_tool_input(
                                    block.get("input") if "input" in block else None,
                                    tool_name=block["name"],
                                    tool_id=block["id"],
                                    request=request,
                                    mode="message-conversion",
                                )
                            ),
                        },
                    })
        elif isinstance(content, str):
            text_parts.append(content)

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function = tool_call["function"]
                arguments = function.get("arguments") if "arguments" in function else None
                if isinstance(arguments, str):
                    arguments = json.dumps(
                        self._parse_tool_arguments(
                            arguments,
                            tool_name=function["name"],
                            tool_id=tool_call["id"],
                            request=request,
                            mode="message-conversion",
                        )
                    )
                else:
                    arguments = json.dumps(
                        self._coerce_tool_input(
                            arguments,
                            tool_name=function["name"],
                            tool_id=tool_call["id"],
                            request=request,
                            mode="message-conversion",
                        )
                    )
                tool_calls.append({
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                })

        converted: dict[str, Any] = {"role": "assistant"}
        converted["content"] = "".join(text_parts) if text_parts else None
        if tool_calls:
            converted["tool_calls"] = tool_calls
        # Copy round-trip fields onto the wire. sanitize_outbound_message()
        # already stripped foreign metadata upstream, so only fields this
        # dialect accepts can be present.
        for field in self._accepted_round_trip_fields:
            if message.get(field):
                converted[field] = list(message[field])
        return converted

    def _convert_tools(self, tools: tuple[ToolDefinition, ...]) -> list[dict[str, Any]]:
        result = []
        for tool in tools:
            tool_dict: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            if tool.provider_options:
                tool_dict["function"].update(dict(tool.provider_options))
            if tool.cache is not None:
                tool_dict["function"]["cache"] = tool.cache
            result.append(tool_dict)
        return result

    # ------------------------------------------------------------------
    # Response parsing.
    # ------------------------------------------------------------------

    def _parse_response(self, response: dict[str, Any], request: Request) -> Result:
        if (
            "choices" not in response
            or not isinstance(response["choices"], list)
            or not response["choices"]
        ):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response missing choices")
        if "usage" not in response:
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response missing usage")

        choice = response["choices"][0]
        if not isinstance(choice, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response choice must be an object")
        message = choice.get("message") or {}
        if not isinstance(message, MappingABC) or not message:
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response choice has empty message")

        reasoning_details = response.get("reasoning_details") or message.get("reasoning_details")
        reasoning_text = self._extract_reasoning_message(message)
        if not reasoning_text and isinstance(reasoning_details, list):
            reasoning_text = self._reasoning_details_text(reasoning_details)

        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, MappingABC) and block.get("type") == "text"
            )
        else:
            raise ProviderProtocolError(
                self.endpoint_url,
                "non-streaming",
                "Response message content must be a string, list, or null",
            )

        tool_calls = message.get("tool_calls") or ()
        if not isinstance(tool_calls, (list, tuple)):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Response tool_calls must be a list")

        return Result(
            text=text,
            tool_calls=tuple(
                self._parse_tool_call(tool_call, request)
                for tool_call in tool_calls
            ),
            reasoning=self._build_reasoning(
                reasoning_text=reasoning_text,
                reasoning_details=reasoning_details,
            ),
            usage=self._parse_usage(response["usage"]),
            stop_reason=self._normalize_finish_reason(choice.get("finish_reason")),
            provider_metadata=ProviderMetadata(
                dialect_name=self.dialect_name,
                endpoint_url=self.endpoint_url,
            ),
        )

    def _reasoning_details_text(self, reasoning_details: list[Any]) -> str:
        parts = []
        for item in reasoning_details:
            if isinstance(item, MappingABC) and item.get("type") == "reasoning.text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    def _new_reasoning_text(self, accumulated_reasoning: str, candidate: str) -> str:
        if not candidate:
            return ""
        if accumulated_reasoning and candidate.startswith(accumulated_reasoning):
            return candidate[len(accumulated_reasoning):]
        if accumulated_reasoning.endswith(candidate):
            return ""
        return candidate

    def _build_reasoning(self, *, reasoning_text: str, reasoning_details: Any) -> ReasoningArtifact | None:
        if not reasoning_text and not reasoning_details:
            return None
        details = (
            tuple(item for item in reasoning_details if isinstance(item, dict))
            if isinstance(reasoning_details, list)
            else ()
        )
        entries = (ReasoningEntry(text=reasoning_text),) if reasoning_text else ()
        return ReasoningArtifact(entries=entries, provider_details=details)

    def _decode_response_json(self, response: httpx.Response, mode: str) -> dict[str, Any]:
        try:
            parsed = response.json()
        except ValueError as error:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Provider returned non-JSON response: {response.text[:200]}",
            ) from error
        if not isinstance(parsed, dict):
            raise ProviderProtocolError(self.endpoint_url, mode, "Provider response JSON must be an object")
        return parsed

    # ------------------------------------------------------------------
    # Tool-call parsing.
    # ------------------------------------------------------------------

    def _parse_tool_arguments(
        self,
        raw_arguments: Any,
        *,
        tool_name: str,
        tool_id: str,
        request: Request,
        mode: str,
    ) -> dict[str, Any]:
        required = self._required_tool_fields(tool_name, request, mode=mode)
        if raw_arguments is None or raw_arguments == "":
            if required:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    mode,
                    (
                        f"Tool call '{tool_id}' for '{tool_name}' omitted JSON arguments "
                        f"required by schema: {required}"
                    ),
                )
            return {}
        if not isinstance(raw_arguments, str):
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' arguments must be a JSON string",
            )
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' contains malformed JSON arguments",
            ) from error
        return self._coerce_tool_input(
            parsed,
            tool_name=tool_name,
            tool_id=tool_id,
            request=request,
            mode=mode,
        )

    def _coerce_tool_input(
        self,
        value: Any,
        *,
        tool_name: str,
        tool_id: str,
        request: Request,
        mode: str,
    ) -> dict[str, Any]:
        required = self._required_tool_fields(tool_name, request, mode=mode)
        if value is None:
            if required:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    mode,
                    (
                        f"Tool call '{tool_id}' for '{tool_name}' omitted input required by schema: "
                        f"{required}"
                    ),
                )
            return {}
        if not isinstance(value, MappingABC):
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' input must be a JSON object",
            )
        missing = [field for field in required if field not in value]
        if missing:
            raise ProviderProtocolError(
                self.endpoint_url,
                mode,
                f"Tool call '{tool_id}' for '{tool_name}' missing required fields: {missing}",
            )
        return dict(value)

    def _required_tool_fields(self, tool_name: str, request: Request, *, mode: str) -> tuple[str, ...]:
        tool = self._loaded_tool_definition(tool_name, request, mode=mode)
        required = tool.input_schema.get("required", [])
        return tuple(required) if isinstance(required, (list, tuple)) else ()

    def _loaded_tool_definition(self, tool_name: str, request: Request, *, mode: str) -> ToolDefinition:
        for tool in request.tools:
            if tool.name == tool_name:
                return tool
        raise ToolNotLoadedError(
            self.endpoint_url,
            mode,
            tool_name,
            f"Provider attempted to call tool '{tool_name}' that was not included in the request",
        )

    def _parse_tool_call(self, tool_call: dict[str, Any], request: Request) -> ToolCall:
        if not isinstance(tool_call, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call must be an object")
        function = tool_call.get("function") or {}
        if not isinstance(function, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call function must be an object")
        tool_id = tool_call.get("id")
        tool_name = function.get("name")
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call missing id")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ProviderProtocolError(self.endpoint_url, "non-streaming", "Tool call function missing name")
        return ToolCall(
            id=tool_id,
            tool_name=tool_name,
            input=self._parse_tool_arguments(
                function.get("arguments") if "arguments" in function else None,
                tool_name=tool_name,
                tool_id=tool_id,
                request=request,
                mode="non-streaming",
            ),
        )

    def _parse_stream_tool_calls(
        self,
        accumulated_tool_calls: dict[int, dict[str, Any]],
        request: Request,
    ) -> tuple[ToolCall, ...]:
        calls = []
        for index in sorted(accumulated_tool_calls):
            state = accumulated_tool_calls[index]
            if not state["id"] or not state["name"]:
                raise ProviderProtocolError(
                    self.endpoint_url,
                    "streaming",
                    f"Streamed tool call at index {index} ended without id and name",
                )
            calls.append(
                ToolCall(
                    id=state["id"],
                    tool_name=state["name"],
                    input=self._parse_tool_arguments(
                        state["arguments"],
                        tool_name=state["name"],
                        tool_id=state["id"],
                        request=request,
                        mode="streaming",
                    ),
                )
            )
        return tuple(calls)

    # ------------------------------------------------------------------
    # Usage parsing.
    # ------------------------------------------------------------------

    def _parse_usage(self, usage: dict[str, Any]) -> Usage:
        if not isinstance(usage, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "usage", "Usage payload must be an object")
        prompt_details = usage.get("prompt_tokens_details") or {}
        if not isinstance(prompt_details, MappingABC):
            raise ProviderProtocolError(self.endpoint_url, "usage", "prompt_tokens_details must be an object")
        prompt_tokens = self._required_usage_token(usage, "prompt_tokens")
        completion_tokens = self._required_usage_token(usage, "completion_tokens")
        cache_write, cache_read = self._extract_cache_usage(prompt_details)
        return Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )

    def _required_usage_token(self, usage: MappingABC[str, Any], field_name: str) -> int:
        if field_name not in usage:
            raise ProviderProtocolError(self.endpoint_url, "usage", f"Usage payload missing {field_name}")
        value = usage[field_name]
        if type(value) is not int or value < 0:
            raise ProviderProtocolError(
                self.endpoint_url,
                "usage",
                f"Usage field {field_name} must be a non-negative integer",
            )
        return value

    def _optional_usage_token(self, usage: MappingABC[str, Any], field_name: str) -> int:
        value = usage.get(field_name, 0)
        if value is None:
            return 0
        if type(value) is not int or value < 0:
            raise ProviderProtocolError(
                self.endpoint_url,
                "usage",
                f"Usage field {field_name} must be a non-negative integer when provided",
            )
        return value

    def _normalize_finish_reason(self, finish_reason: str | None) -> StopReason:
        mapping: dict[str, StopReason] = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "max_tokens": "max_tokens",
            "recitation": "end_turn",
            "safety": "end_turn",
            "error": "end_turn",
        }
        normalized = finish_reason.lower() if isinstance(finish_reason, str) else None
        if normalized not in mapping and normalized is not None:
            logger.warning("Unmapped finish_reason='%s', defaulting to end_turn", finish_reason)
        return mapping.get(normalized, "end_turn")

    # ------------------------------------------------------------------
    # Error normalization.
    # ------------------------------------------------------------------

    def _handle_http_error(self, error: httpx.HTTPStatusError, *, mode: str) -> NoReturn:
        response = error.response
        error_body = None
        fallback_text = str(error)
        if response is not None:
            fallback_text = response.text
            try:
                error_body = response.json()
            except ValueError:
                pass
            self._raise_provider_http_error(
                status=response.status_code,
                error_body=error_body,
                fallback_text=fallback_text,
                mode=mode,
            )
        raise ProviderProtocolError(
            self.endpoint_url,
            mode,
            f"{self.dialect_name} provider HTTP error without response: {fallback_text}",
        )

    def _raise_provider_http_error(
        self,
        *,
        status: int,
        error_body: dict[str, Any] | None,
        fallback_text: str,
        mode: str,
    ) -> NoReturn:
        if error_body and status >= 400:
            logger.error("%s API %s error %d — raw body: %s", self.dialect_name, mode, status, repr(error_body))
        error_message = _extract_provider_message(error_body, fallback_text)
        if status == 400 and error_body:
            error_info = error_body.get("error", {})
            error_code = str(error_info.get("code", ""))
            if "context_length" in error_code or "reduce the length" in error_message.lower():
                raise ProviderContextOverflowError(self.endpoint_url, mode, error_message)
            if error_code == "tool_use_failed":
                match = re.search(r"attempted to call tool '(\w+)'", error_message)
                if match:
                    raise ToolNotLoadedError(self.endpoint_url, mode, match.group(1), error_message)

        if status in (401, 403):
            raise ProviderAuthError(self.endpoint_url, mode, error_message)
        if status == 429 or status >= 500:
            raise ProviderRetryableError(self.endpoint_url, status, mode, error_message)
        raise ProviderProtocolError(
            self.endpoint_url,
            mode,
            f"{self.dialect_name} provider API error {status}: {error_message}",
        )


def _json_or_none(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_provider_message(error_body: dict[str, Any] | None, fallback_text: str) -> str:
    if not error_body:
        return fallback_text
    error_info = error_body.get("error", {})
    message = error_info.get("message")
    if not message:
        return fallback_text
    # Include additional context fields (e.g. OpenRouter provider/description) that
    # carry the real error reason beyond the top-level message string.
    extra = {}
    for key in ("provider", "description", "code"):
        if key in error_info and key != "message":
            extra[key] = error_info[key]
    if extra:
        tags = ", ".join(f"{k}={v!s}" for k, v in extra.items())
        return f"{message} [{tags}]"
    return message
