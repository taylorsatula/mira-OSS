"""Asynchronously compact committed tool results in the hot Valkey cache."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from clients.llm_provider import LLMProvider
from cns.core.message import Message
from cns.infrastructure.valkey_message_cache import ToolResultCachePatch
from cns.services.async_work_barrier import get_async_work_barrier
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.core.events import ToolResultHistoryCommittedEvent
    from cns.infrastructure.valkey_message_cache import ValkeyMessageCache
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)

TOOL_RESULT_SUMMARIZE_THRESHOLD = 2000
SUMMARY_MAX_TOKENS = 1024
MIN_PROSE_FIELD_CHARS = 500
ROOT_ARRAY = "$"

_ARRAY_FIELDS = ("results", "emails", "items", "messages")
_TEXT_FIELDS = ("content", "body", "text")
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


@dataclass(frozen=True)
class JsonCompactionTarget:
    """One supported edit to apply to a JSON tool result."""

    mode: Literal["array", "text"]
    field: str


class JsonCompactionDecision(BaseModel):
    """Model output for one preselected JSON compaction target."""

    model_config = ConfigDict(extra="forbid", strict=True)

    keep_indices: list[int] = Field(
        default_factory=list,
        description="Original array indices to retain for array mode.",
    )
    summary: str = Field(
        default="",
        description="Replacement text for text mode.",
    )


class ToolResultSummarizer:
    """Compact oversized committed tool results without mutating durable history."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        message_cache: ValkeyMessageCache,
    ) -> None:
        self._llm_provider = llm_provider
        self._message_cache = message_cache
        self._barrier = get_async_work_barrier()
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="tool_summarizer",
        )

    def subscribe(self, event_bus: EventBus) -> None:
        """Subscribe to post-commit tool result events."""
        event_bus.subscribe(
            "ToolResultHistoryCommittedEvent",
            self._handle_tool_results_committed,
        )
        logger.info(
            "ToolResultSummarizer subscribed to ToolResultHistoryCommittedEvent"
        )

    def _handle_tool_results_committed(
        self,
        event: ToolResultHistoryCommittedEvent,
    ) -> None:
        """Queue oversized string tool results from the committed turn."""
        user_id = get_current_user_id()
        if user_id != event.user_id:
            raise RuntimeError(
                "Tool result commit event user does not match the active user context"
            )

        items = tuple(
            message
            for message in event.tool_messages
            if (
                message.role == "tool"
                and isinstance(message.content, str)
                and len(message.content) > TOOL_RESULT_SUMMARIZE_THRESHOLD
                and not message.is_error
                and not message.metadata.get("tool_result_retrieved")
                and message.metadata.get("tool_result_id")
                and message.metadata.get("tool_result_session_id")
            )
        )
        if not items:
            return

        done = self._barrier.register_work(user_id)
        try:
            self._executor.submit(
                copy_context().run,
                self._summarize_and_update,
                items,
                done,
            )
        except Exception:
            done()
            logger.exception(
                "Failed to queue tool result summarization for user %s",
                user_id,
            )

    def _summarize_and_update(
        self,
        items: tuple[Message, ...],
        done_callback: Callable[[], None],
    ) -> None:
        """Build summaries and conditionally patch their current cache entries."""
        try:
            patches: list[ToolResultCachePatch] = []
            for message in items:
                tool_name = (
                    message.metadata.get("tool_name")
                    or message.tool_call_id
                    or "unknown"
                )
                try:
                    compacted = self._summarize_one(message, tool_name)
                except Exception:
                    logger.exception(
                        "Tool result compaction failed for tool '%s'; keeping original",
                        tool_name,
                    )
                    continue
                if compacted is not None:
                    patches.append(ToolResultCachePatch(
                        message_id=message.id,
                        expected_content=cast(str, message.content),
                        compacted_content=compacted,
                    ))

            applied_count = self._message_cache.apply_tool_result_patches(patches)
            if applied_count:
                logger.info(
                    "Compacted %d tool result(s) in Valkey for user %s",
                    applied_count,
                    get_current_user_id(),
                )
        except Exception:
            logger.exception("Tool result summarization pipeline failed")
        finally:
            done_callback()

    def _summarize_one(self, message: Message, tool_name: str) -> str | None:
        """Return a smaller cache representation, or None when unsupported."""
        if not isinstance(message.content, str):
            raise TypeError("Tool result summarization requires string content")
        content = message.content

        parsed_json: dict[str, object] | list[object] | None = None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, (dict, list)):
                parsed_json = parsed
        except json.JSONDecodeError:
            pass

        target = (
            self._select_json_target(parsed_json)
            if parsed_json is not None
            else None
        )
        if parsed_json is not None and target is None:
            return None

        from config.prompts.loader import load_prompt

        response = self._llm_provider.generate_response(
            system_prompt=load_prompt("tool_result_summarization_system.txt"),
            messages=[{
                "role": "user",
                "content": load_prompt("tool_result_summarization_user.txt").format(
                    tool_name=tool_name,
                    input_format="json" if parsed_json is not None else "text",
                    target_mode=target.mode if target is not None else "plain_text",
                    target_field=target.field if target is not None else "",
                    tool_output=content,
                ),
            }],
            internal_llm="analysis",
            max_tokens=SUMMARY_MAX_TOKENS,
            allow_negative=True,
        )
        response_text = self._llm_provider.extract_text_content(response).strip()
        if not response_text:
            return None

        compacted = (
            self._apply_json_decision(parsed_json, target, response_text)
            if parsed_json is not None and target is not None
            else response_text
        )
        if len(compacted) >= len(content):
            logger.warning(
                "Tool result compaction was not smaller for tool '%s'; keeping original",
                tool_name,
            )
            return None
        return compacted

    @staticmethod
    def _select_json_target(
        value: dict[str, object] | list[object],
    ) -> JsonCompactionTarget | None:
        if isinstance(value, list):
            if len(value) > 1 and all(isinstance(item, (dict, list)) for item in value):
                return JsonCompactionTarget(mode="array", field=ROOT_ARRAY)
            return None

        array_candidates = []
        for field in _ARRAY_FIELDS:
            candidate = value.get(field)
            if (
                isinstance(candidate, list)
                and len(candidate) > 1
                and all(isinstance(item, (dict, list)) for item in candidate)
            ):
                array_candidates.append(field)
        if array_candidates:
            largest = max(
                array_candidates,
                key=lambda field: len(json.dumps(value.get(field))),
            )
            return JsonCompactionTarget(mode="array", field=largest)

        text_candidates = []
        for field in _TEXT_FIELDS:
            candidate = value.get(field)
            if (
                isinstance(candidate, str)
                and len(candidate) >= MIN_PROSE_FIELD_CHARS
            ):
                text_candidates.append(field)
        if text_candidates:
            largest = max(text_candidates, key=lambda field: len(str(value.get(field))))
            return JsonCompactionTarget(mode="text", field=largest)
        return None

    def _apply_json_decision(
        self,
        original: dict[str, object] | list[object],
        target: JsonCompactionTarget,
        response_text: str,
    ) -> str:
        """Apply one validated decision to a copy of the original JSON."""
        from json_repair import repair_json

        decision = JsonCompactionDecision.model_validate(
            repair_json(response_text, return_objects=True)
        )
        compacted = deepcopy(original)

        if target.mode == "array":
            original_array = self._get_array(original, target.field)
            compacted_array = self._get_array(compacted, target.field)
            indices = decision.keep_indices
            if (
                not indices
                or indices != sorted(set(indices))
                or indices[0] < 0
                or indices[-1] >= len(original_array)
                or len(indices) >= len(original_array)
            ):
                raise ValueError("Array decision is invalid or does not reduce data")
            compacted_array[:] = [compacted_array[index] for index in indices]
        else:
            if not isinstance(original, dict) or not isinstance(compacted, dict):
                raise ValueError("Text decisions require a JSON object")
            original_text = original.get(target.field)
            if not isinstance(original_text, str):
                raise ValueError("Text decision target is not a string")
            summary = decision.summary.strip()
            if not summary or len(summary) >= len(original_text):
                raise ValueError("Text decision is empty or not smaller")
            if not set(_URL_PATTERN.findall(summary)) <= set(
                _URL_PATTERN.findall(original_text)
            ):
                raise ValueError("Text decision invented or changed a URL")
            compacted[target.field] = summary

        return json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _get_array(
        value: dict[str, object] | list[object],
        field: str,
    ) -> list[object]:
        array = value if field == ROOT_ARRAY else (
            value.get(field) if isinstance(value, dict) else None
        )
        if not isinstance(array, list):
            raise ValueError(f"JSON target is not an array: {field}")
        return array


_summarizer: ToolResultSummarizer | None = None


def initialize_tool_result_summarizer(
    llm_provider: LLMProvider,
    event_bus: EventBus,
    message_cache: ValkeyMessageCache,
) -> ToolResultSummarizer:
    """Initialize the global tool result summarizer."""
    global _summarizer
    _summarizer = ToolResultSummarizer(llm_provider, message_cache)
    _summarizer.subscribe(event_bus)
    logger.info("Tool result summarizer initialized")
    return _summarizer


def get_tool_result_summarizer() -> ToolResultSummarizer:
    """Get the global tool result summarizer instance."""
    if _summarizer is None:
        raise RuntimeError(
            "Tool result summarizer not initialized. "
            "Call initialize_tool_result_summarizer() during startup."
        )
    return _summarizer
