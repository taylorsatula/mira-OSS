"""
Main continuum orchestrator for CNS.

Coordinates all continuum processing: message handling, LLM interaction,
tool execution, working memory updates, and event publishing.

Optimized to generate embeddings once and propagate them to all services.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Callable, TypedDict, TYPE_CHECKING
from uuid import uuid4
import numpy as np

if TYPE_CHECKING:
    from cns.infrastructure.continuum_repository import ContinuumRepository
    from cns.infrastructure.continuum_pool import UnitOfWork
    from cns.core.message import Message
    from cns.services.memory_relevance_service import MemoryRelevanceService
    from cns.services.subcortical import SubcorticalLayer
    from cns.integration.event_bus import EventBus
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory
    from cns.services.live_context_compaction_service import LiveContextCompactionService

from config import config
from cns.core.continuum import Continuum
from cns.core.events import (
    ContinuumEvent,
    ToolResultHistoryCommittedEvent,
    TurnCompletedEvent,
)
from clients.llm.events import (
    CircuitBreakerEvent,
    CompleteEvent,
    FileArtifactEvent,
    ModelStepCompletedEvent,
    ProviderSwitchEvent,
    StreamEvent,
    TextEvent,
    ThinkingEvent,
    ToolCompletedEvent,
    ToolErrorEvent,
    ToolExecutingEvent,
)
from clients.llm_provider import LLMProvider, ContextOverflowError
from clients.llm.tool_messages import append_tool_result_messages
from clients.llm.types import Result, ToolCall, ToolDefinition, ToolResult, Usage
from cns.services.tool_loop import CircuitBreaker, ToolLoopExecutor
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
from cns.services.subcortical import SubcorticalResult, SurfacedMemory
from lt_memory.models import MemoryDict
from lt_memory.proactive import (
    MAX_PINNED_MEMORIES,
    MAX_SURFACED_MEMORIES,
    MIN_FRESH_MEMORIES,
    NUM_ASSISTANT_MESSAGES,
    ASSISTANT_SIMILARITY_THRESHOLD,
    MAX_ASSISTANT_MEMORIES,
)
from utils.tag_parser import TagParser, match_memory_id
from utils.user_context import get_current_segment_id

# Context overflow remediation
CONTEXT_OVERFLOW_FALLBACK_TEXT_LIMIT = 400
CONTEXT_OVERFLOW_RECENT_USER_TURNS = 10

TOOL_RESULT_MAX_CHARS = 31999  # Max chars for single tool result before truncation

# Tool result persistence
TOMBSTONE_MODE = False  # When True, tool results not persisted to history
MAX_LOCAL_TOOL_CALLS_PER_TURN = 50
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


def _try_json_list_truncation(result: str, limit: int) -> tuple[str, str] | None:
    """Try to truncate a JSON array to N complete elements that fit within limit.

    Returns (truncated_json, description) on success, None if result isn't a JSON list.
    """
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, list) or len(parsed) <= 1:
        return None

    total = len(parsed)

    # Binary search: find max N where json.dumps(parsed[:N]) fits within limit
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(json.dumps(parsed[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1

    if lo == 0:
        return None

    return json.dumps(parsed[:lo]), f"{lo} of {total} items"


def _truncate_tool_result(result: str, limit: int, tool_name: str) -> str:
    """Truncate an oversized tool result, preserving structure when possible.

    Tries format-aware handlers (JSON list slicing) before falling back to
    raw character truncation. Always appends a truncation marker.
    """
    original_len = len(result)

    # Handler chain: try each format, first match wins
    json_result = _try_json_list_truncation(result, limit)
    if json_result is not None:
        truncated, description = json_result
        marker = (
            f"\n\n[Truncated: {tool_name} returned {description} "
            f"({original_len:,} chars). Request smaller result sets for full data.]"
        )
        logger.warning(
            "Truncated tool result from %s: %d -> %d chars (JSON list: %s)",
            tool_name, original_len, len(truncated), description
        )
        return truncated + marker

    # Raw fallback
    truncated = result[:limit]
    marker = (
        f"\n\n[Truncated: {tool_name} returned {original_len:,} chars, "
        f"exceeding {limit:,} char limit. Request smaller result sets for full data.]"
    )
    logger.warning(
        "Truncated tool result from %s: %d -> %d chars (raw)",
        tool_name, original_len, len(truncated)
    )
    return truncated + marker


class TurnMetadata(TypedDict, total=False):
    """Metadata accumulated during process_message() and returned to the caller."""
    tools_used: list[str]
    referenced_memories: list[str]
    surfaced_memories: list[str]
    pinned_memory_ids: list[str]
    emotion: str
    thinking: str
    model_error: bool
    model_error_reason: str


class LLMKwargs(TypedDict, total=False):
    """Keyword arguments matching LLMProvider.stream_events() explicit params."""
    conversation_llm: str
    dialect_name: str
    effort: str
    endpoint_url: str
    model: str
    api_key: str
    container_id: str
    temperature: float
    max_tokens: int
    system_prompt: str
    allow_negative: bool


@dataclass
class MemorySurfacingResult:
    """Output of the memory surfacing pipeline."""
    surfaced_memories: list[SurfacedMemory]
    pinned_ids: set[str]
    subcortical_result: SubcorticalResult | None


@dataclass
class ToolInteraction:
    """Tracks a single tool use/result pair during the LLM turn."""
    tool_name: str
    tool_id: str
    arguments: dict[str, object]
    result: str | list[dict[str, object]] = ""
    completed: bool = False
    is_error: bool = False


@dataclass
class TurnAccumulator:
    """Accumulates state during the LLM streaming event loop."""
    # User-visible text assembled from streamed TextEvent chunks. If a model
    # emits text before requesting a tool, that text is intentionally retained:
    # the browser already received it, and the completed turn must match what
    # was streamed.
    response_text: str = ""
    thinking_content: str = ""
    raw_response: Result | None = None
    invoked_tool_loader: bool = False
    touch_resolved_uuids: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    tool_interactions: list[ToolInteraction] = field(default_factory=list)
    # Intermediate Results from tool-call iterations. Each entry is one model
    # step that produced tool_calls (stop_reason="tool_use"). The final Result
    # for the user turn is stored in raw_response.
    tool_call_results: list[Result] = field(default_factory=list)

    def reset(self):
        """Reset for overflow retry — clears all accumulated state."""
        self.response_text = ""
        self.thinking_content = ""
        self.raw_response = None
        self.invoked_tool_loader = False
        self.touch_resolved_uuids = []
        self.tools_used = []
        self.events = []
        self.tool_interactions = []
        self.tool_call_results = []


class ContinuumOrchestrator:
    """
    Main orchestration service for continuum processing.
    
    Coordinates the entire continuum flow from user input to final response,
    managing all system interactions through clean interfaces.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        continuum_repo: ContinuumRepository,
        working_memory: WorkingMemory,
        tool_repo: ToolRepository,
        tag_parser: TagParser,
        subcortical_layer: SubcorticalLayer,
        event_bus: EventBus,
        memory_relevance_service: MemoryRelevanceService,
        live_context_compaction_service: LiveContextCompactionService,
    ) -> None:
        """
        Initialize orchestrator with dependencies.

        All parameters are REQUIRED. The orchestrator will fail immediately
        if any required dependency is missing or used incorrectly.

        Args:
            llm_provider: LLM provider for generating responses (required)
            continuum_repo: Repository for message persistence (required)
            working_memory: Working memory system for prompt composition (required)
            tool_repo: Tool repository for tool definitions (required)
            tag_parser: Tag parser for response parsing (required)
            subcortical_layer: Subcortical layer for retrieval query expansion (required).
                              Raises RuntimeError on generation failures - no degraded state.
            event_bus: Event bus for publishing/subscribing to events (required)
            memory_relevance_service: Memory relevance service for surfacing long-term memories (required).
                                     Raises exceptions on infrastructure failures - no degraded state.
            live_context_compaction_service: Background live compaction service (required).
        """
        self.llm_provider = llm_provider
        self.continuum_repo = continuum_repo
        self.working_memory = working_memory
        self.tool_repo = tool_repo
        self.tag_parser = tag_parser
        self.subcortical_layer = subcortical_layer
        self.memory_relevance_service = memory_relevance_service
        self.live_context_compaction_service = live_context_compaction_service
        self.event_bus = event_bus

        # Get singleton embeddings provider for generating embeddings once
        self.embeddings_provider = get_hybrid_embeddings_provider()

        # Store composed prompt sections when received via event
        self._cached_content = None
        self._non_cached_content = None
        self._conversation_prefix_items = ()  # Before history (set from event.conversation_prefix_items)
        self._post_history_items = ()  # After history, before HUD (domaindoc, BP4)
        self._notification_center = None

        # In-memory cache of provider-reported input tokens per continuum
        self._last_turn_usage: dict[str, int] = {}  # {continuum_id: input_tokens}

        # Subscribe to events
        self.event_bus.subscribe('SystemPromptComposedEvent', self._handle_system_prompt_composed)
        self.event_bus.subscribe('SegmentCollapsedEvent', self._invalidate_on_segment_collapse)

        logger.info("ContinuumOrchestrator initialized")

    def _format_tool_indicator(self, events: list) -> str:
        """Format tool usage indicator from stream events."""
        tool_names = []
        for event in events:
            if isinstance(event, ToolExecutingEvent) and event.tool_name not in tool_names:
                tool_names.append(event.tool_name)

        if not tool_names:
            return ""
        return f"[used: {', '.join(tool_names)}]"

    def _build_tool_history_messages(
        self,
        interactions: list[ToolInteraction],
        tool_call_results: list[Result],
    ) -> list[Message]:
        """
        Build Message objects for persisting tool use/result history.

        Produces one assistant message per tool-call iteration (using
        assistant_message_from_result for faithful reasoning/metadata),
        followed by tool result messages for each interaction.
        Timestamps are offset by microseconds to guarantee deterministic
        intra-batch ordering without leapfrogging the final text assistant
        message created immediately after this returns.
        """
        from datetime import timedelta
        from cns.core.message import Message, MessageMetadata
        from clients.llm.tool_messages import assistant_message_from_result

        base_time = utc_now()
        messages: list[Message] = []
        us_offset = 1  # microsecond counter for deterministic ordering

        # Index completed interactions by tool_id for matching to Results.
        # Failed tools are completed interactions too: their role="tool" result
        # carries is_error=True so provider replay remains structurally valid.
        completed_interactions_by_tool_id: dict[str, ToolInteraction] = {
            i.tool_id: i for i in interactions if i.completed
        }

        for result in tool_call_results:
            persisted_tool_call_ids = {
                tool_call.id
                for tool_call in result.tool_calls
                if (
                    tool_call.tool_name != "code_execution"
                    and tool_call.id in completed_interactions_by_tool_id
                )
            }
            if not persisted_tool_call_ids:
                continue

            # Build faithful assistant dict from the Result — preserves
            # reasoning blocks, text, thinking_signatures, reasoning_details —
            # but only includes tool calls with matching persisted results.
            assistant_dict = assistant_message_from_result(
                result,
                include_tool_call_ids=persisted_tool_call_ids,
            )
            content = assistant_dict["content"]

            # Extract metadata that lives on Message.metadata, not on content
            assistant_metadata: MessageMetadata = {"has_tool_calls": True}
            if assistant_dict.get("thinking_signatures"):
                assistant_metadata["thinking_signatures"] = assistant_dict["thinking_signatures"]
            if assistant_dict.get("reasoning_details"):
                assistant_metadata["reasoning_details"] = assistant_dict["reasoning_details"]

            assistant_msg = Message(
                content=content,
                role="assistant",
                metadata=assistant_metadata,
            )
            object.__setattr__(
                assistant_msg, 'created_at',
                base_time + timedelta(microseconds=us_offset),
            )
            us_offset += 1
            messages.append(assistant_msg)

            # Tool result messages for each persisted tool call in this Result
            limit = TOOL_RESULT_MAX_CHARS
            for tool_call in result.tool_calls:
                if tool_call.tool_name == "code_execution":
                    continue
                interaction = completed_interactions_by_tool_id.get(tool_call.id)
                if interaction is None:
                    continue
                tool_result = interaction.result
                is_retrieved_result = (
                    interaction.tool_name == "continuum_tool"
                    and interaction.arguments.get("operation") == "get_tool_result"
                )
                if isinstance(tool_result, list):
                    tool_result = json.dumps(tool_result)
                if (
                    not is_retrieved_result
                    and isinstance(tool_result, str)
                    and len(tool_result) > limit
                ):
                    tool_result = _truncate_tool_result(
                        tool_result, limit, interaction.tool_name,
                    )

                message_id = uuid4()
                tool_metadata: MessageMetadata = {"tool_name": interaction.tool_name}
                if is_retrieved_result:
                    tool_metadata["tool_result_retrieved"] = True
                    source_id = interaction.arguments.get("tool_result_id")
                    if isinstance(source_id, str):
                        tool_metadata["tool_result_source_id"] = source_id
                elif (
                    not interaction.is_error
                    and isinstance(tool_result, str)
                    and (segment_id := get_current_segment_id())
                ):
                    session_prefix = segment_id.replace("-", "")[:8]
                    tool_metadata["tool_result_id"] = (
                        f"tr_{session_prefix}_{message_id.hex}"
                    )
                    tool_metadata["tool_result_session_id"] = segment_id

                tool_msg = Message(
                    id=message_id,
                    content=tool_result,
                    role="tool",
                    metadata=tool_metadata,
                    tool_call_id=interaction.tool_id,
                    is_error=interaction.is_error,
                )
                object.__setattr__(
                    tool_msg, 'created_at',
                    base_time + timedelta(microseconds=us_offset),
                )
                us_offset += 1
                messages.append(tool_msg)

        return messages

    def _cached_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for the LLM request."""
        return self.tool_repo.get_all_tool_definitions()

    def _compose_llm_messages(
        self,
        continuum: Continuum,
        system_prompt: str,
        llm_config_name: str,
    ) -> list[dict[str, object]]:
        from cns.core.events import ComposeSystemPromptEvent

        self._cached_content = None
        self._non_cached_content = None
        self._conversation_prefix_items = ()
        self._post_history_items = ()
        self._notification_center = None
        self.event_bus.publish(ComposeSystemPromptEvent.create(
            continuum_id=str(continuum.id),
            base_prompt=system_prompt,
        ))

        cached_content = self._cached_content or ""
        non_cached_content = self._non_cached_content or ""
        conversation_prefix_items = self._conversation_prefix_items or ()
        post_history_items = self._post_history_items or ()
        notification_center = self._notification_center or {}

        system_blocks = []
        all_system_parts = []
        if cached_content:
            all_system_parts.append(cached_content)
        if non_cached_content:
            all_system_parts.append(non_cached_content)

        if all_system_parts:
            system_blocks.append({
                "type": "text",
                "text": "\n\n".join(all_system_parts),
            })

        if llm_config_name == "gpt-legacy":
            system_blocks.append({
                "type": "text",
                "text": (
                    "GPT-4o HAS A HISTORY OF BEING OVERTLY SYCOPHANTIC TO USERS. "
                    "PLEASE BE AWARE OF THIS PULL AND COURSE CORRECT IF YOU SEE "
                    "YOURSELF FORMING RESPONSES LIKE THIS INCLUDING STEPPING BACK "
                    "AND ADDRESSING YOUR ACTIONS DIRECTLY WITH THE USER."
                ),
            })

        prefix_messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item}],
                "_is_scaffold": True,
            }
            for item in conversation_prefix_items
        ]

        # Post-history messages are recomposed before every local tool follow-up
        # turn so tool mutations to domaindocs/reminders are visible immediately.
        post_history_messages = []
        for i, item in enumerate(post_history_items):
            call_id = f"dd_{i:04d}"
            post_history_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "id": call_id,
                    "name": "domaindoc_tool",
                    "input": {"operation": "overview"},
                }],
            })
            post_history_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": item,
            })

        messages = self.live_context_compaction_service.format_messages_for_api(continuum)
        current_user_msg = messages[-1]
        history_messages = messages[:-1]

        stable_messages = [
            {"role": "system", "content": system_blocks},
            *prefix_messages,
            *history_messages,
            *post_history_messages,
        ]

        # The notification center changes every turn, so it sits after the
        # last stable assistant message.
        for i in range(len(stable_messages) - 1, -1, -1):
            if stable_messages[i].get("role") == "assistant":
                break

        # Inject HUD as a synthetic tool call + result pair.
        # The model sees it as a tool interaction (not Mira speaking),
        # and the matching tool_call/tool pair avoids provider 400s.
        hud_messages: list[dict[str, object]] = []
        if notification_center:
            hud_id = f"hud_{uuid4().hex[:8]}"
            hud_json = json.dumps(notification_center, ensure_ascii=False)
            hud_messages = [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_call",
                        "id": hud_id,
                        "name": "hud",
                        "input": {"action": "get_runtime_state"},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": hud_id,
                    "content": hud_json,
                },
            ]

        complete_messages = [
            *stable_messages,
            *hud_messages,
            current_user_msg,
        ]
        nc_chars = sum(len(v) for v in notification_center.values())
        logger.debug(
            f"Injected {len(prefix_messages)} prefix msgs + "
            f"{len(history_messages)} history msgs + "
            f"{len(post_history_messages)} post-history msgs + "
            f"HUD ({len(notification_center)} fields, {nc_chars} chars)"
        )
        return complete_messages

    def _stream_model_tool_loop(
        self,
        *,
        base_messages: list[dict[str, object]],
        compose_messages: Callable[[], list[dict[str, object]]],
        llm_kwargs: LLMKwargs,
    ) -> Iterator[StreamEvent]:
        tool_executor = ToolLoopExecutor(self.tool_repo)
        breaker = CircuitBreaker()
        continuation_messages: list[dict[str, object]] = []
        messages_for_llm = [*base_messages]
        active_llm_kwargs = dict(llm_kwargs)
        available_tools = self._cached_tool_definitions()
        accumulated_usage: Usage | None = None
        executed_local_tool_calls = 0
        tool_limit_finalization_attempted = False

        while True:
            result: Result | None = None
            for event in self.llm_provider.stream_events(
                messages=messages_for_llm,
                tools=available_tools,
                **active_llm_kwargs,
            ):
                if isinstance(event, CompleteEvent):
                    result = event.response
                    continue
                yield event

            if result is None:
                raise RuntimeError("No completion event received from LLM stream")

            if result.usage is not None:
                accumulated_usage = (
                    result.usage
                    if accumulated_usage is None
                    else accumulated_usage + result.usage
                )

            local_tool_calls = [
                tool_call for tool_call in result.tool_calls
                if tool_call.tool_name != "code_execution"
            ]
            if not local_tool_calls:
                final_usage = accumulated_usage if accumulated_usage is not None else result.usage
                yield CompleteEvent(response=result.with_usage(final_usage))
                return
            if (
                executed_local_tool_calls >= MAX_LOCAL_TOOL_CALLS_PER_TURN
                and tool_limit_finalization_attempted
            ):
                reason = (
                    f"Local tool call limit reached ({MAX_LOCAL_TOOL_CALLS_PER_TURN}); "
                    "model requested additional tools after tools were disabled"
                )
                fallback_text = (
                    "I reached the local tool call limit for this turn, so I need to stop "
                    "using tools here. I can continue from the information already gathered, "
                    "or you can send a follow-up message to start a new turn."
                )
                yield CircuitBreakerEvent(reason=reason)
                yield TextEvent(content=fallback_text)
                final_usage = accumulated_usage if accumulated_usage is not None else result.usage
                yield CompleteEvent(response=Result(
                    text=fallback_text,
                    usage=final_usage,
                    stop_reason="end_turn",
                    provider_metadata=result.provider_metadata,
                    container_id=result.container_id,
                ))
                return

            # Surface the intermediate tool-call Result so the accumulator
            # can persist reasoning metadata (thinking_signatures, etc).
            yield ModelStepCompletedEvent(response=result)

            remaining_tool_calls = MAX_LOCAL_TOOL_CALLS_PER_TURN - executed_local_tool_calls
            executable_tool_calls = local_tool_calls[:max(remaining_tool_calls, 0)]
            skipped_tool_calls = local_tool_calls[max(remaining_tool_calls, 0):]

            if executable_tool_calls:
                tool_results = yield from tool_executor.execute_tools(executable_tool_calls, breaker)
                executed_local_tool_calls += len(executable_tool_calls)
            else:
                tool_results = ()

            if skipped_tool_calls:
                reason = (
                    f"Local tool call limit reached ({MAX_LOCAL_TOOL_CALLS_PER_TURN}); "
                    f"skipped {len(skipped_tool_calls)} requested tool call(s)"
                )
                yield CircuitBreakerEvent(reason=reason)
                tool_results = (
                    *tool_results,
                    *self._tool_limit_results(skipped_tool_calls),
                )
                tool_limit_finalization_attempted = True
                available_tools = []
            elif executed_local_tool_calls >= MAX_LOCAL_TOOL_CALLS_PER_TURN:
                reason = (
                    f"Local tool call limit reached ({MAX_LOCAL_TOOL_CALLS_PER_TURN}); "
                    "no more local tool calls are available for this turn"
                )
                yield CircuitBreakerEvent(reason=reason)
                tool_results = self._append_tool_loop_notice(tool_results, reason)
                tool_limit_finalization_attempted = True
                available_tools = []
            else:
                should_continue, reason = breaker.should_continue()
                if not should_continue:
                    yield CircuitBreakerEvent(reason=reason)
                    tool_results = self._append_tool_loop_notice(
                        tool_results,
                        (
                            f"Tool call issue detected - {reason}. No more tool calls "
                            "available. Provide your response to the user based on "
                            "information gathered so far."
                        ),
                    )
                    available_tools = []
                else:
                    available_tools = self._cached_tool_definitions()

            continuation_messages = append_tool_result_messages(
                continuation_messages,
                result,
                tool_results,
            )
            base_messages = compose_messages()
            messages_for_llm = [*base_messages, *continuation_messages]
            active_llm_kwargs = self._continuation_llm_kwargs(active_llm_kwargs, result)

    def _tool_limit_results(self, tool_calls: list[ToolCall]) -> tuple[ToolResult, ...]:
        notice = (
            f"[Automated system message: Local tool call limit reached "
            f"({MAX_LOCAL_TOOL_CALLS_PER_TURN} per user turn). This tool call was not "
            "executed. No more tool calls are available. Provide your response to the "
            "user based on information gathered so far.]"
        )
        return tuple(
            ToolResult(tool_call_id=tool_call.id, content=notice)
            for tool_call in tool_calls
        )

    def _append_tool_loop_notice(
        self,
        tool_results: tuple[ToolResult, ...],
        notice: str,
    ) -> tuple[ToolResult, ...]:
        if not tool_results:
            return tool_results
        formatted_notice = f"[Automated system message: {notice}]"
        *earlier_results, last_result = tool_results
        if isinstance(last_result.content, list):
            content: str | list[dict[str, object]] = [
                *last_result.content,
                {"type": "text", "text": formatted_notice},
            ]
        else:
            content = f"{last_result.content}\n\n{formatted_notice}"
        return (
            *earlier_results,
            ToolResult(
                tool_call_id=last_result.tool_call_id,
                content=content,
                is_error=last_result.is_error,
            ),
        )

    def _continuation_llm_kwargs(self, llm_kwargs: LLMKwargs, result: Result) -> LLMKwargs:
        metadata = result.provider_metadata
        continuation = dict(llm_kwargs)
        if metadata.conversation_llm_name:
            for key in ("dialect_name", "endpoint_url", "model", "api_key"):
                continuation.pop(key, None)
            continuation["conversation_llm"] = metadata.conversation_llm_name
        else:
            continuation.pop("conversation_llm", None)
            if metadata.dialect_name:
                continuation["dialect_name"] = metadata.dialect_name
            if metadata.endpoint_url:
                continuation["endpoint_url"] = metadata.endpoint_url
            if metadata.model:
                continuation["model"] = metadata.model
        return continuation

    def _consume_stream(
        self,
        stream_events: Iterator[StreamEvent],
        acc: TurnAccumulator,
        continuum_id: str,
        stream: bool,
        stream_callback: Callable[[dict[str, object]], None] | None,
        show_thinking_stream: bool,
    ) -> None:
        """
        Consume LLM stream events, accumulating results into the TurnAccumulator.

        Handles tool execution logging, response text accumulation, stream callback
        dispatch, container persistence, and cache metric tracking.
        """
        from utils.user_context import check_cancelled
        from clients.valkey_client import get_valkey
        valkey = get_valkey()

        for event in stream_events:
            check_cancelled()

            # Tool execution tracking
            if isinstance(event, ToolExecutingEvent):
                acc.tool_interactions.append(ToolInteraction(
                    tool_name=event.tool_name,
                    tool_id=event.tool_id,
                    arguments=event.arguments
                ))
                if event.tool_name not in acc.tools_used:
                    acc.tools_used.append(event.tool_name)

                if event.tool_name == "code_execution":
                    logger.info("=" * 80)
                    logger.info("🐍 CODE_EXECUTION INVOKED")
                    logger.info("=" * 80)
                    code = event.arguments.get("code", "")
                    logger.info("Python code to execute: %s characters", len(code))
                    logger.info("=" * 80)
                elif event.tool_name == "invokeother_tool":
                    mode = event.arguments.get("mode", "")
                    if mode in ["load", "fallback", "prepare_code_execution"]:
                        acc.invoked_tool_loader = True
                        logger.info(f"Detected invokeother_tool execution with mode={mode}")
                else:
                    logger.info(
                        "Tool executing: %s with argument keys: %s",
                        event.tool_name,
                        sorted(event.arguments.keys())
                    )

            # Tool completion — capture result and touch results
            if isinstance(event, ToolCompletedEvent):
                for interaction in acc.tool_interactions:
                    if interaction.tool_id == event.tool_id:
                        interaction.result = event.result
                        interaction.completed = True
                        break
                if event.tool_name == "code_execution":
                    logger.info("=" * 80)
                    logger.info("✅ CODE_EXECUTION COMPLETED")
                    logger.info("=" * 80)
                    logger.info("Code execution result length: %s", len(event.result))
                    logger.info("=" * 80)
                elif event.tool_name == "memory_tool":
                    logger.info(
                        "Tool completed: %s with result length %s",
                        event.tool_name,
                        len(event.result)
                    )
                    try:
                        tool_result = json.loads(event.result) if isinstance(event.result, str) else event.result
                        if tool_result.get("status") == "touched":
                            acc.touch_resolved_uuids = tool_result.get("resolved_uuids", [])
                    except (json.JSONDecodeError, AttributeError):
                        logger.warning("Failed to parse memory_tool touch result")
                elif isinstance(event.result, list):
                    # Content blocks (images) — log text blocks only, skip base64
                    text_block_count = sum(1 for b in event.result if b.get('type') == 'text')
                    logger.info(
                        "Tool completed: %s with %s content blocks and %s text blocks",
                        event.tool_name,
                        len(event.result),
                        text_block_count
                    )
                else:
                    logger.info(
                        "Tool completed: %s with result type %s",
                        event.tool_name,
                        type(event.result).__name__
                    )

                # Check for inline image artifact in tool result
                try:
                    tool_result = json.loads(event.result) if isinstance(event.result, str) else event.result
                    image_artifact = tool_result.get("_image_artifact") if isinstance(tool_result, dict) else None
                    if image_artifact and stream and stream_callback:
                        alt = image_artifact.get("alt_text", "Generated image")
                        fid = image_artifact["file_id"]
                        image_tag = f"\n\n![{alt}](/v0/api/images/{fid})\n\n📥 [Download full resolution](/v0/api/files/{fid})\n\n"
                        stream_callback({"type": "text", "content": image_tag})
                        acc.response_text += image_tag
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # Not an image-producing tool, ignore

            # Tool errors
            if isinstance(event, ToolErrorEvent):
                for interaction in acc.tool_interactions:
                    if interaction.tool_id == event.tool_id:
                        interaction.result = event.result
                        interaction.completed = True
                        interaction.is_error = True
                        break
                if event.tool_name == "code_execution":
                    logger.error("=" * 80)
                    logger.error("❌ CODE_EXECUTION FAILED")
                    logger.error("=" * 80)
                    logger.error(f"Error:\n{event.error}")
                    logger.error("=" * 80)
                else:
                    logger.error(f"Tool error: {event.tool_name} -> {event.error}")

            # Accumulate user-visible text exactly as it was streamed. Text
            # emitted before a model asks for a tool is still visible output.
            if isinstance(event, TextEvent):
                acc.response_text += event.content

            # Stream to callback
            if stream and stream_callback:
                if isinstance(event, TextEvent):
                    stream_callback({"type": "text", "content": event.content})
                elif isinstance(event, ThinkingEvent) and show_thinking_stream:
                    stream_callback({"type": "thinking", "content": event.content})
                elif hasattr(event, 'tool_name'):
                    stream_callback({"type": "tool_event", "event": event.type, "tool": event.tool_name})
                elif isinstance(event, CircuitBreakerEvent):
                    if "failed after correction" in event.reason:
                        stream_callback({
                            "type": "model_error",
                            "reason": event.reason
                        })
                elif isinstance(event, FileArtifactEvent):
                    size = event.size_bytes
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"

                    safe_display = event.filename.replace('[', '\\[').replace(']', '\\]')
                    download_link = f"\n\n\ud83d\udcce **[{safe_display}](/v0/api/files/{event.file_id})** ({size_str})\n\n"

                    stream_callback({"type": "text", "content": download_link})
                    acc.response_text += download_link
                elif isinstance(event, ProviderSwitchEvent):
                    stream_callback({
                        "type": "provider_switch",
                        "backup_model": event.backup_model,
                        "reason": event.reason,
                    })

            acc.events.append(event)

            # Capture model-step responses for replay separately from the
            # terminal turn response. CompleteEvent is terminal for this
            # orchestrator stream; ModelStepCompletedEvent is not.
            if isinstance(event, ModelStepCompletedEvent):
                acc.tool_call_results.append(event.response)
            elif isinstance(event, CompleteEvent):
                acc.raw_response = event.response
                acc.thinking_content = self.llm_provider.extract_thinking_content(event.response)

            if isinstance(event, (ModelStepCompletedEvent, CompleteEvent)):
                # Store container_id in Valkey for reuse (1-hour TTL)
                if event.response.container_id:
                    valkey_key = f"container:{continuum_id}"
                    valkey.setex(valkey_key, 3600, event.response.container_id)
                    logger.info(f"📦 Stored container ID in Valkey: {event.response.container_id}")

                # Log provider cache metrics and track input tokens for next turn.
                if event.response.usage:
                    usage = event.response.usage
                    self._last_turn_usage[continuum_id] = usage.input_tokens
                    cache_created = usage.cache_creation_input_tokens
                    cache_read = usage.cache_read_input_tokens
                    if cache_created > 0:
                        logger.info(f"Cache created: {cache_created} tokens")
                    if cache_read > 0:
                        logger.debug(f"Cache read: {cache_read} tokens")

    def process_message(
        self,
        continuum: Continuum,
        user_message: str | list[dict[str, object]],
        system_prompt: str,
        stream: bool = False,
        stream_callback: Callable[[dict[str, object]], None] | None = None,
        _tried_loading_all_tools: bool = False,
        unit_of_work: UnitOfWork | None = None,
        storage_content: str | list[dict[str, object]] | None = None,
        segment_turn_number: int = 1,
    ) -> tuple[Continuum, str, TurnMetadata]:
        """
        Process user message through complete continuum flow.

        Args:
            continuum: Current continuum state
            user_message: User's input message (string or multimodal content array).
                         For images, this should be the inference tier (1200px).
            system_prompt: Base system prompt
            stream: Whether to stream response
            stream_callback: Callback for streaming chunks
            _tried_loading_all_tools: Internal flag to prevent infinite need_tool loops
            unit_of_work: Optional UnitOfWork for batching persistence operations
            storage_content: Content for persistence (optional). For images, this should
                           be the storage tier (512px WebP). If not provided, user_message
                           is used for persistence.
            segment_turn_number: Turn number within current segment (1-indexed).
                               Incremented at API entry point for real user messages.

        Returns:
            Tuple of (updated_continuum, final_response, metadata)
        """
        metadata: TurnMetadata = {}

        # Balance pre-check - UX optimization (skipped in OSS mode)
        # The atomic record_usage() in LLM provider is the authoritative check
        try:
            from billing import get_billing_backend
            from billing.exceptions import InsufficientBalanceError
            from utils.user_context import get_current_user_id, has_user_context

            if has_user_context():
                user_id = str(get_current_user_id())
                billing = get_billing_backend()
                if not billing.check_balance(user_id, allow_negative=False):
                    raise InsufficientBalanceError(billing.get_balance(user_id))
        except ImportError:
            pass  # OSS mode - no billing enforcement

        # Add user message to continuum cache (no persistence yet)
        user_msg_obj, user_events = continuum.add_user_message(user_message)
        self._publish_events(user_events)

        # Extract text content for weighted context (bypass for multimodal)
        text_for_context = user_message
        if isinstance(user_message, list):
            text_parts = [item['text'] for item in user_message if item.get('type') == 'text']
            text_for_context = ' '.join(text_parts) if text_parts else 'Image uploaded'

        # Memory surfacing: subcortical → retention → fresh retrieval → merge
        previous_memories = self._get_previous_memories()
        mem = self._surface_memories(continuum, text_for_context, previous_memories)

        # Publish merged memories to ProactiveMemoryTrinket
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=str(continuum.id),
            target_trinket="ProactiveMemoryTrinket",
            context={"memories": mem.surfaced_memories}
        ))

        # Apply conversation LLM intent and thinking configuration
        from utils.user_context import get_user_preferences, resolve_conversation_llm

        llm_kwargs: LLMKwargs = {}
        prefs = get_user_preferences()
        llm_config = resolve_conversation_llm(prefs.conversation_llm)

        llm_kwargs['conversation_llm'] = llm_config.name
        llm_kwargs['max_tokens'] = 31999  # Frontend generation ceiling
        show_thinking_stream = not (
            llm_config.dialect_name != "anthropic"
            and not config.api.show_openai_compat_thinking
        )

        # Extended thinking: enabled only when subcortical layer provides complexity assessment
        if mem.subcortical_result is not None:
            effort_level = mem.subcortical_result.get_effort_level()
            llm_kwargs['effort'] = effort_level
            logger.info(f"Thinking: complexity={mem.subcortical_result.complexity} effort={effort_level}")

        def compose_messages() -> list[dict[str, object]]:
            return self._compose_llm_messages(continuum, system_prompt, llm_config.name)

        complete_messages = compose_messages()
        messages_for_llm = complete_messages
        available_tools = self._cached_tool_definitions()

        # Use last turn's actual provider-reported input tokens for compaction check.
        # Segment collapse invalidation via _invalidate_on_segment_collapse event handler.
        continuum_id_str = str(continuum.id)
        last_input = self._last_turn_usage.get(continuum_id_str)
        self._maybe_prefetch_context_compaction(
            continuum_id=continuum_id_str,
            input_tokens=last_input,
        )

        # Retrieve container_id from Valkey for adapter-owned container reuse.
        # Container reuse is Anthropic-only (server-side code_execution feature);
        # OpenAI-compatible providers reject container params and capability validation
        # would refuse the request, so the container_id is intentionally not forwarded
        # for non-Anthropic adapters.
        if llm_config.dialect_name == "anthropic":
            from clients.valkey_client import get_valkey
            valkey = get_valkey()
            valkey_key = f"container:{continuum.id}"
            container_id = valkey.get(valkey_key)
            if container_id:
                llm_kwargs['container_id'] = container_id
                logger.info(f"📦 Reusing container from Valkey: {container_id}")
            else:
                logger.debug("📦 No existing container - new container will be created")

        # Stream LLM response with overflow remediation
        acc = TurnAccumulator()
        continuum_id = str(continuum.id)
        overflow_attempt = 0
        deep_fallback_active = False

        def compose_messages_for_tool_loop() -> list[dict[str, object]]:
            base_messages = compose_messages()
            if deep_fallback_active:
                return self._apply_deep_context_overflow_fallback(base_messages)
            return base_messages

        while True:
            try:
                self._consume_stream(
                    self._stream_model_tool_loop(
                        base_messages=messages_for_llm,
                        compose_messages=compose_messages_for_tool_loop,
                        llm_kwargs=llm_kwargs,
                    ),
                    acc, continuum_id, stream, stream_callback, show_thinking_stream,
                )
                break  # Success

            except ContextOverflowError as e:
                overflow_attempt += 1
                logger.warning(
                    "Context overflow from API: %s (attempt %d)",
                    e,
                    overflow_attempt,
                )
                if overflow_attempt == 1:
                    deep_fallback_active = True
                    messages_for_llm = self._apply_deep_context_overflow_fallback(messages_for_llm)
                    acc.reset()
                    continue

                acc.reset()
                acc.response_text = "collapse the segment, please. incremental compaction failed"
                metadata["model_error"] = True
                metadata["model_error_reason"] = "incremental_compaction_failed"
                if stream and stream_callback:
                    stream_callback({
                        "type": "text",
                        "content": "collapse the segment, please. incremental compaction failed",
                    })
                break

        # Prepend tool indicator for cache visibility (so MIRA sees its tool usage in history)
        tool_indicator = self._format_tool_indicator(acc.events)
        if tool_indicator:
            acc.response_text = f"{tool_indicator}\n\n{acc.response_text}"

        # Parse tags from final response (preserve emotion tag for frontend extraction)
        parsed_tags = self.tag_parser.parse_response(acc.response_text, preserve_tags=['my_emotion'])
        clean_response_text = parsed_tags['clean_text']

        # Process check-in response if Mira produced one
        if parsed_tags.get('checkin_response'):
            self._process_checkin_response(parsed_tags['checkin_response'])

        logger.debug(f"Emotion extracted: {parsed_tags.get('emotion')}")
        logger.debug(f"Emotion tag in clean_text: {'<mira:my_emotion>' in clean_response_text}")

        # Check if model tool error caused a blank response - provide user-friendly fallback
        model_tool_error = next(
            (e for e in acc.events if isinstance(e, CircuitBreakerEvent)
             and "failed after correction" in e.reason),
            None
        )
        if model_tool_error and (not clean_response_text or not clean_response_text.strip()):
            logger.warning(f"Model returned blank after tool error: {model_tool_error.reason}")
            clean_response_text = (
                "I encountered an issue with this request. The AI model made an invalid "
                "tool call that couldn't be corrected. This is a limitation of the model, "
                "not MIRA. Please try rephrasing your request."
            )
            metadata["model_error"] = True
            metadata["model_error_reason"] = str(model_tool_error.reason)

        # Validate response is not blank before saving
        if not clean_response_text or not clean_response_text.strip():
            logger.error("Attempted to save blank assistant response - rejecting")
            raise ValueError("Assistant response cannot be blank or empty. This may indicate an API error.")

        # Build assistant metadata
        assistant_metadata = {
            "referenced_memories": acc.touch_resolved_uuids,
            "surfaced_memories": [m['id'] for m in mem.surfaced_memories],
            "pinned_memory_ids": list(mem.pinned_ids)
        }

        if acc.tools_used:
            assistant_metadata["has_tool_calls"] = True
            assistant_metadata["tools_used"] = acc.tools_used

        if parsed_tags.get('emotion'):
            assistant_metadata["emotion"] = parsed_tags['emotion']

        if acc.thinking_content:
            assistant_metadata["thinking"] = acc.thinking_content

        if acc.raw_response:
            assistant_metadata["stop_reason"] = acc.raw_response.stop_reason

        if metadata.get("model_error"):
            assistant_metadata["model_error"] = True
            assistant_metadata["model_error_reason"] = metadata["model_error_reason"]

        # Build tool history messages for persistence (before assistant message
        # so they appear in correct chronological order in the cache)
        # Exclude code_execution: server tools can't be replayed to the API as
        # forged client-side tool_use blocks. The streaming path persists the
        # result inline in the assistant's text response and via container reuse.
        completed_interactions = [
            i for i in acc.tool_interactions
            if i.completed and i.tool_name != "code_execution"
        ]
        tool_history_messages: list[Message] = []
        if completed_interactions and not TOMBSTONE_MODE:
            tool_history_messages = self._build_tool_history_messages(
                completed_interactions, acc.tool_call_results,
            )
            continuum.add_tool_history(tool_history_messages)

        assistant_msg_obj, response_events = continuum.add_assistant_message(
            clean_response_text, assistant_metadata
        )
        self._publish_events(response_events)

        # Publish turn completed event
        turn_number = (len(continuum.messages) + 1) // 2
        self._publish_events([TurnCompletedEvent.create(
            continuum_id=continuum_id,
            turn_number=turn_number,
            segment_turn_number=segment_turn_number,
            continuum=continuum
        )])

        # Speculatively warm vLLM's prefix cache for the next turn's subcortical
        # call while the user reads/types. No-op unless prefill_warmup is enabled.
        self.subcortical_layer.warm_cache(continuum, mem.surfaced_memories)

        final_response = clean_response_text

        # Update metadata
        metadata["tools_used"] = acc.tools_used
        metadata["referenced_memories"] = acc.touch_resolved_uuids
        metadata["surfaced_memories"] = [m['id'] for m in mem.surfaced_memories]
        metadata["pinned_memory_ids"] = list(mem.pinned_ids)

        if parsed_tags.get('emotion'):
            metadata["emotion"] = parsed_tags['emotion']

        if acc.thinking_content:
            metadata["thinking"] = acc.thinking_content

        # Unit of Work is required for proper persistence
        if not unit_of_work:
            raise ValueError("Unit of Work is required for message persistence")

        # Validate: if user_message contains images, storage_content MUST be provided
        if isinstance(user_msg_obj.content, list):
            has_image = any(item.get('type') == 'image' for item in user_msg_obj.content)
            if has_image and storage_content is None:
                raise ValueError(
                    "storage_content is required when user_message contains images. "
                    "Callers must provide the 512px WebP storage tier for image persistence."
                )

        persist_content = storage_content if storage_content is not None else user_msg_obj.content

        from cns.core.message import Message
        persist_user_msg = Message(
            content=persist_content,
            role=user_msg_obj.role,
            id=user_msg_obj.id,
            created_at=user_msg_obj.created_at,
            metadata=user_msg_obj.metadata
        )

        unit_of_work.add_messages(persist_user_msg, *tool_history_messages, assistant_msg_obj)
        unit_of_work.mark_metadata_updated()
        committed_tool_messages = [
            message for message in tool_history_messages
            if message.role == "tool"
        ]
        if committed_tool_messages:
            committed_event = ToolResultHistoryCommittedEvent.create(
                continuum_id=continuum_id,
                tool_messages=committed_tool_messages,
            )
            unit_of_work.add_post_commit_callback(
                lambda event=committed_event: self._publish_events([event])
            )

        # Auto-continuation: If tools were loaded and we haven't already tried,
        # automatically continue with the task
        if acc.invoked_tool_loader and not _tried_loading_all_tools:
            logger.info("Auto-continuing after tool loading...")

            synthetic_message = (
                "<system-scaffold>The requested tool is now loaded. "
                "Continue with the original task.</system-scaffold>"
            )

            continuum, final_response, metadata = self.process_message(
                continuum,
                synthetic_message,
                system_prompt,
                stream=stream,
                stream_callback=stream_callback,
                _tried_loading_all_tools=True,
                unit_of_work=unit_of_work,
                segment_turn_number=segment_turn_number
            )
            logger.info("Auto-continuation completed successfully")

        return continuum, final_response, metadata
    
    def _handle_system_prompt_composed(self, event: ContinuumEvent) -> None:
        """Handle system prompt composed event."""
        from cns.core.events import SystemPromptComposedEvent
        assert isinstance(event, SystemPromptComposedEvent)
        self._cached_content = event.cached_content
        self._non_cached_content = event.non_cached_content
        self._conversation_prefix_items = event.conversation_prefix_items
        self._post_history_items = event.post_history_items
        self._notification_center = event.notification_center or {}
        nc_chars = sum(len(v) for v in self._notification_center.values())
        logger.debug(
            f"Received system prompt: cached {len(event.cached_content)} chars, "
            f"non-cached {len(event.non_cached_content)} chars, "
            f"{len(event.conversation_prefix_items)} prefix items, "
            f"{len(event.post_history_items)} post-history items, "
            f"HUD {len(self._notification_center)} fields ({nc_chars} chars)"
        )


    def _publish_events(self, events: list[ContinuumEvent]):
        """Publish events to event bus."""
        for event in events:
            self.event_bus.publish(event)

    def _get_previous_memories(self) -> list[MemoryDict]:
        """
        Get previously surfaced memories from the trinket cache.

        Returns:
            List of memory dicts from the previous turn, or empty list if none
        """
        trinket = self.working_memory.get_trinket('ProactiveMemoryTrinket')
        if trinket and hasattr(trinket, 'get_cached_memories'):
            return trinket.get_cached_memories()
        return []

    def _surface_memories(
        self,
        continuum: Continuum,
        text_for_context: str,
        previous_memories: list[MemoryDict],
    ) -> MemorySurfacingResult:
        """
        Run the full memory surfacing pipeline: subcortical → retention → fresh retrieval → merge.

        Handles subcortical failure gracefully by retaining all previous memories
        (hard-capped) without fresh retrieval.

        Args:
            continuum: Current continuum state (passed to subcortical layer)
            text_for_context: Text content for query expansion
            previous_memories: Memories from the previous turn's trinket cache

        Returns:
            MemorySurfacingResult with surfaced_memories, pinned_ids, and subcortical_result
        """
        max_pinned = MAX_PINNED_MEMORIES
        max_surfaced = MAX_SURFACED_MEMORIES
        min_fresh = MIN_FRESH_MEMORIES

        try:
            subcortical_result = self.subcortical_layer.generate(
                continuum,
                text_for_context,
                previous_memories=previous_memories
            )
        except Exception:
            logger.exception("Subcortical processing failed; retaining previous memories without fresh retrieval")
            pinned_ids: set[str] = set()
            pinned_memories = previous_memories

            if len(pinned_memories) > max_pinned:
                pinned_memories.sort(key=lambda m: m.get('importance_score', 0.5), reverse=True)
                pinned_memories = pinned_memories[:max_pinned]
                logger.info(f"Hard cap (fallback): truncated pinned to {max_pinned}")

            logger.info(f"Memory surfacing: {len(pinned_memories)} pinned (no fresh retrieval)")
            return MemorySurfacingResult(
                surfaced_memories=pinned_memories,
                pinned_ids=pinned_ids,
                subcortical_result=None,
            )

        # LLM-guided retention
        pinned_ids = subcortical_result.pinned_memory_ids
        pinned_memories = self._apply_retention(previous_memories, pinned_ids)

        if len(pinned_memories) > max_pinned:
            pinned_memories.sort(key=lambda m: m.get('importance_score', 0.5), reverse=True)
            pinned_memories = pinned_memories[:max_pinned]
            logger.info(f"Hard cap: truncated pinned from {len(pinned_ids)} to {max_pinned}")

        # Sliding fresh budget: as pinned grows, fresh compresses
        fresh_limit = max(min_fresh, max_surfaced - len(pinned_memories))

        assistant_text = self._format_assistant_messages(continuum, NUM_ASSISTANT_MESSAGES)

        expansion_embedding, assistant_embedding = self._compute_embeddings_parallel(
            subcortical_result.query_expansion, assistant_text
        )

        fresh_memories = self.memory_relevance_service.get_relevant_memories(
            query_expansion=subcortical_result.query_expansion,
            expansion_embedding=expansion_embedding,
            limit=fresh_limit,
            extracted_entities=subcortical_result.entities
        )

        assistant_memories: list[MemoryDict] = []
        if assistant_text and assistant_embedding is not None:
            assistant_memories = self.memory_relevance_service.get_relevant_memories(
                query_expansion=assistant_text,
                expansion_embedding=assistant_embedding,
                limit=MAX_ASSISTANT_MEMORIES,
                extracted_entities=None
            )
            assistant_memories = [
                m for m in assistant_memories
                if m.get('similarity_score', 0) >= ASSISTANT_SIMILARITY_THRESHOLD
            ]
            if assistant_memories:
                logger.debug(
                    f"Assistant surfacing: {len(assistant_memories)} passed "
                    f"threshold ({ASSISTANT_SIMILARITY_THRESHOLD})"
                )
                fresh_memories = self._merge_assistant_memories(fresh_memories, assistant_memories)

        surfaced_memories = self._merge_memories(pinned_memories, fresh_memories)

        logger.info(
            f"Memory surfacing: {len(pinned_memories)} pinned + "
            f"{len(fresh_memories)} fresh = {len(surfaced_memories)} total"
        )

        return MemorySurfacingResult(
            surfaced_memories=surfaced_memories,
            pinned_ids=pinned_ids,
            subcortical_result=subcortical_result,
        )

    def _apply_retention(
        self,
        previous_memories: list[MemoryDict],
        pinned_ids: set[str]
    ) -> list[MemoryDict]:
        """
        Filter previous memories to keep only those marked for retention.

        Matches by 8-char ID prefix since the LLM outputs shortened IDs.

        Args:
            previous_memories: All memories from previous turn
            pinned_ids: Set of 8-char memory IDs marked [x] by the LLM

        Returns:
            List of memories that should be pinned (retained)
        """
        if not previous_memories or not pinned_ids:
            return []

        pinned = []
        for memory in previous_memories:
            memory_id = memory.get('id', '')
            if memory_id and any(match_memory_id(memory_id, pid) for pid in pinned_ids):
                pinned.append(memory)

        logger.debug(
            f"Retention: {len(pinned)}/{len(previous_memories)} memories retained"
        )
        return pinned

    def _merge_memories(
        self,
        pinned_memories: list[MemoryDict],
        fresh_memories: list[MemoryDict]
    ) -> list[MemoryDict]:
        """
        Merge pinned and fresh memories, deduplicating by ID.

        Pinned memories appear first and take precedence.

        Args:
            pinned_memories: Memories retained from previous turn
            fresh_memories: Newly retrieved memories

        Returns:
            Merged list with pinned first, then fresh (no duplicates)
        """
        # Start with pinned memories
        merged = list(pinned_memories)
        seen_ids = {m.get('id') for m in pinned_memories if m.get('id')}

        # Add fresh memories that aren't already in pinned
        for memory in fresh_memories:
            memory_id = memory.get('id')
            if memory_id and memory_id not in seen_ids:
                merged.append(memory)
                seen_ids.add(memory_id)

        return merged

    def _compute_embeddings_parallel(
        self,
        query_expansion: str,
        assistant_text: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Compute embeddings for query expansion and assistant messages in parallel.

        Returns:
            Tuple of (expansion_embedding, assistant_embedding). Either may be None
            if the corresponding text is empty.
        """
        if not assistant_text:
            expansion_embedding = self.embeddings_provider.encode_realtime(query_expansion)
            return expansion_embedding, None

        with ThreadPoolExecutor(max_workers=2) as executor:
            ctx_expansion = copy_context()
            ctx_assistant = copy_context()
            expansion_future = executor.submit(
                ctx_expansion.run,
                lambda: self.embeddings_provider.encode_realtime(query_expansion)
            )
            assistant_future = executor.submit(
                ctx_assistant.run,
                lambda: self.embeddings_provider.encode_realtime(assistant_text)
            )
            expansion_embedding = expansion_future.result()
            assistant_embedding = assistant_future.result()

        return expansion_embedding, assistant_embedding

    def _merge_assistant_memories(
        self,
        fresh_memories: list[MemoryDict],
        assistant_memories: list[MemoryDict],
    ) -> list[MemoryDict]:
        """
        Merge assistant memories into fresh memories with union deduplication.

        Assistant memories that aren't already in fresh are appended, then the
        combined list is sorted by similarity score descending.

        Args:
            fresh_memories: Primary retrieved memories
            assistant_memories: Secondary memories from assistant text search (pre-filtered)

        Returns:
            Merged list sorted by similarity score
        """
        if not assistant_memories:
            return fresh_memories

        merged = list(fresh_memories)
        seen_ids = {m.get('id') for m in fresh_memories if m.get('id')}

        assistant_added = 0
        for memory in assistant_memories:
            memory_id = memory.get('id')
            if memory_id and memory_id not in seen_ids:
                merged.append(memory)
                seen_ids.add(memory_id)
                assistant_added += 1

        merged.sort(key=lambda m: m.get('similarity_score', 0), reverse=True)

        if assistant_added > 0:
            logger.info(f"Assistant memories added to surfacing: {assistant_added}")

        return merged

    def _format_assistant_messages(self, continuum: Continuum, limit: int = 2) -> str:
        """Extract and concatenate last N assistant messages."""
        assistant_messages = [m for m in continuum.messages if m.role == "assistant"]
        recent = assistant_messages[-limit:] if len(assistant_messages) >= limit else assistant_messages
        return " ".join(
            m.content if isinstance(m.content, str) else ""
            for m in recent
        )

    # =========================================================================
    # Context Overflow Detection and Remediation
    # =========================================================================

    def _invalidate_on_segment_collapse(self, event: "ContinuumEvent") -> None:
        """Invalidate cached input-token baseline when segment collapse shrinks the message set."""
        self._last_turn_usage.pop(event.continuum_id, None)

    def _maybe_prefetch_context_compaction(
        self,
        *,
        continuum_id: str,
        input_tokens: int | None,
    ) -> None:
        """Speculatively pre-compute a context compaction brief so it can be injected with zero latency when ready."""
        if input_tokens is None:
            return
        try:
            if not self.live_context_compaction_service.should_schedule(input_tokens):
                return
            scheduled = self.live_context_compaction_service.schedule(continuum_id)
            if scheduled:
                logger.info(
                    "Scheduled live context compaction: input_tokens=%d trigger=%d",
                    input_tokens,
                    config.api.compaction_trigger_tokens,
                )
        except Exception:
            logger.error("Live context compaction scheduling failed", exc_info=True)

    def _apply_deep_context_overflow_fallback(
        self,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """
        Aggressively shrink an already composed provider request after overflow.

        System and assistant prefix text scaffolding are preserved. Tool calls
        and role=tool results are tombstoned everywhere to avoid orphaned
        provider-structural messages.
        """
        recent_user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user"
        ][-CONTEXT_OVERFLOW_RECENT_USER_TURNS:]
        recent_start = recent_user_indices[0] if recent_user_indices else len(messages)

        shaped: list[dict[str, object]] = []
        for index, message in enumerate(messages):
            if message.get("role") == "tool":
                continue

            keep_text_intact = (
                message.get("role") == "system"
                or message.get("_is_scaffold")
                or index >= recent_start
            )
            shaped_message = self._shape_message_for_context_overflow(
                message,
                keep_text_intact=keep_text_intact,
            )
            shaped.append(shaped_message)

        logger.info(
            "Applied deep context overflow fallback: before=%d after=%d recent_user_turns=%d",
            len(messages),
            len(shaped),
            len(recent_user_indices),
        )
        return shaped

    def _shape_message_for_context_overflow(
        self,
        message: dict[str, object],
        *,
        keep_text_intact: bool,
    ) -> dict[str, object]:
        shaped = dict(message)
        shaped.pop("_is_scaffold", None)
        shaped.pop("thinking_signatures", None)
        shaped.pop("reasoning_details", None)

        content = message.get("content")
        if isinstance(content, str):
            shaped_content = (
                content
                if keep_text_intact
                else self._truncate_fallback_text(content)
            )
            shaped["content"] = shaped_content
            return shaped

        if not isinstance(content, list):
            shaped["content"] = str(content)
            return shaped

        new_blocks: list[dict[str, object]] = []
        removed_tool_calls = 0
        omitted_non_text = 0

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_call":
                removed_tool_calls += 1
                continue
            if block_type in {"text", "reasoning"}:
                new_block = dict(block)
                text = str(new_block.get("text", ""))
                if not keep_text_intact:
                    text = self._truncate_fallback_text(text)
                new_block["text"] = text
                new_blocks.append(new_block)
                continue
            if keep_text_intact:
                new_blocks.append(dict(block))
            else:
                omitted_non_text += 1

        if removed_tool_calls:
            new_blocks.append({
                "type": "text",
                "text": (
                    f"[Omitted {removed_tool_calls} prior tool call(s) during "
                    "context overflow fallback.]"
                ),
            })
            shaped.pop("has_tool_calls", None)
        if omitted_non_text:
            new_blocks.append({
                "type": "text",
                "text": (
                    f"[Omitted {omitted_non_text} older non-text content block(s) "
                    "during context overflow fallback.]"
                ),
            })

        if not new_blocks:
            new_blocks.append({
                "type": "text",
                "text": "[Omitted prior non-text content during context overflow fallback.]",
            })

        shaped["content"] = new_blocks
        return shaped

    def _truncate_fallback_text(self, text: str) -> str:
        if len(text) <= CONTEXT_OVERFLOW_FALLBACK_TEXT_LIMIT:
            return text
        return (
            text[:CONTEXT_OVERFLOW_FALLBACK_TEXT_LIMIT]
            + "\n[Truncated older text during context overflow fallback.]"
        )

    def _process_checkin_response(self, checkin_response: str) -> None:
        """
        Store check-in feedback and invalidate the LoRA trinket cache.

        Non-fatal: logs errors but doesn't fail the conversation turn.

        Args:
            checkin_response: Extracted feedback text from <mira:checkin_response> tag
        """
        try:
            from cns.infrastructure.feedback_tracker import FeedbackTracker
            from utils.user_context import get_current_user_id
            from clients.valkey_client import get_valkey_client
            from working_memory.trinkets.base import TRINKET_KEY_PREFIX

            user_id = get_current_user_id()
            tracker = FeedbackTracker()
            tracker.acknowledge_checkin(user_id, checkin_response)

            # Invalidate LoRA trinket cache so next turn omits <behavioral_checkin>
            valkey = get_valkey_client()
            valkey.hdel_with_retry(f"{TRINKET_KEY_PREFIX}:{user_id}", "behavioral_directives")

            logger.info("Check-in response processed for user %s", user_id)
        except Exception as e:
            logger.error("Failed to process check-in response: %s", e, exc_info=True)


# Global orchestrator instance (singleton pattern)
_orchestrator_instance = None


def initialize_orchestrator(orchestrator_instance: ContinuumOrchestrator) -> None:
    """
    Initialize the global orchestrator instance.
    
    This should be called once during application startup after creating
    the orchestrator with all its dependencies.
    
    Args:
        orchestrator_instance: The configured ConversationOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        logger.warning("Orchestrator already initialized, replacing existing instance")
    _orchestrator_instance = orchestrator_instance
    logger.info("Global orchestrator instance initialized")


def get_orchestrator() -> ContinuumOrchestrator:
    """
    Get the global orchestrator instance.
    
    Returns:
        The singleton ConversationOrchestrator instance
        
    Raises:
        RuntimeError: If orchestrator has not been initialized
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        raise RuntimeError(
            "Orchestrator not initialized. Ensure initialize_orchestrator() "
            "is called during application startup."
        )
    return _orchestrator_instance
