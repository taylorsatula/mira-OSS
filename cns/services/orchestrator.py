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
from typing import Callable, Literal, TypedDict, TYPE_CHECKING
from uuid import UUID
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

import anthropic.types
from config import config
from cns.core.continuum import Continuum
from cns.core.events import (
    ContinuumEvent,
    TurnCompletedEvent
)
from cns.core.stream_events import StreamEvent
from clients.llm_provider import LLMProvider, ContextOverflowError
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

# Context overflow remediation
TOPIC_DRIFT_WINDOW_SIZE = 3
TOPIC_DRIFT_THRESHOLD = 0.6
OVERFLOW_FALLBACK_PRUNE_COUNT = 5
TOOL_RESULT_MAX_CHARS = 100_000  # Max chars for single tool result before truncation

# Tool result persistence
TOMBSTONE_MODE = False  # When True, tool results not persisted to history
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


def _safe_slice(messages: list[dict[str, object]], cut_idx: int) -> list[dict[str, object]]:
    """Slice messages at cut_idx, preserving tool_use/tool_result pair integrity.

    Returns messages[:1] + messages[adjusted_cut:], where the cut is shifted
    to avoid orphaning tool_use or tool_result blocks at the boundary.
    """

    def _is_tool_block(msg: dict[str, object], role: str, block_type: str) -> bool:
        if msg.get("role") != role:
            return False
        content = msg.get("content")
        return (isinstance(content, list)
                and any(isinstance(b, dict) and b.get("type") == block_type for b in content))

    # Adjust forward past orphaned tool_result at the start of the kept window
    original_cut = cut_idx
    while cut_idx < len(messages) and _is_tool_block(messages[cut_idx], "user", "tool_result"):
        cut_idx += 1

    result = messages[:1] + messages[cut_idx:]

    # Trim trailing orphaned tool_use at the end
    while len(result) > 1 and _is_tool_block(result[-1], "assistant", "tool_use"):
        result.pop()

    if cut_idx != original_cut:
        logger.warning(
            "Tool pair safety: adjusted cut index %d -> %d to preserve "
            "tool_use/tool_result integrity (skipped %d orphaned messages)",
            original_cut, cut_idx, cut_idx - original_cut
        )

    return result


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
    model_preference: str
    effort: str
    endpoint_url: str
    model_override: str
    api_key_override: str
    container_id: str
    temperature: float
    max_tokens: int
    system_override: str
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


@dataclass
class TurnAccumulator:
    """Accumulates state during the LLM streaming event loop."""
    response_text: str = ""
    thinking_content: str = ""
    raw_response: anthropic.types.Message | None = None
    invoked_tool_loader: bool = False
    touch_resolved_uuids: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    tool_interactions: list[ToolInteraction] = field(default_factory=list)

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
        """
        self.llm_provider = llm_provider
        self.continuum_repo = continuum_repo
        self.working_memory = working_memory
        self.tool_repo = tool_repo
        self.tag_parser = tag_parser
        self.subcortical_layer = subcortical_layer
        self.memory_relevance_service = memory_relevance_service
        self.event_bus = event_bus

        # Get singleton embeddings provider for generating embeddings once
        self.embeddings_provider = get_hybrid_embeddings_provider()

        # Store composed prompt sections when received via event
        self._cached_content = None
        self._non_cached_content = None
        self._conversation_prefix_items = ()  # Before history (currently unused, reserved)
        self._post_history_items = ()  # After history, before HUD (domaindoc, BP4)
        self._notification_center = None

        # In-memory token tracking for context overflow detection
        # Tracks actual input tokens from previous turn for accurate estimation
        self._last_turn_usage: dict[str, int] = {}  # {continuum_id: input_tokens}
        # One-shot context trim from async LLM judgment (future extension)
        self._pending_context_trim: dict[str, int] = {}  # {continuum_id: trim_index}

        # Subscribe to system prompt composed event
        self.event_bus.subscribe('SystemPromptComposedEvent', self._handle_system_prompt_composed)

        logger.info("ContinuumOrchestrator initialized")

    def _format_tool_indicator(self, events: list) -> str:
        """Format tool usage indicator from stream events."""
        from cns.core.stream_events import ToolExecutingEvent

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
    ) -> list[Message]:
        """
        Build Message objects for persisting tool use/result history.

        Produces one assistant message containing tool_use content blocks,
        followed by one tool message per interaction with the raw result.
        Timestamps are offset by microseconds (not milliseconds) to guarantee
        deterministic intra-batch ordering without leapfrogging the final text
        assistant message that is created immediately after this returns —
        millisecond offsets pushed tool history past the natural utc_now() of
        the subsequent message, inverting chronological order in the DB.
        """
        from datetime import timedelta
        from cns.core.message import Message

        base_time = utc_now()
        messages: list[Message] = []

        # Assistant message with tool_use blocks
        tool_use_blocks = [
            {
                "type": "tool_use",
                "id": interaction.tool_id,
                "name": interaction.tool_name,
                "input": interaction.arguments,
            }
            for interaction in interactions
        ]
        assistant_msg = Message(
            content=tool_use_blocks,
            role="assistant",
            metadata={"has_tool_calls": True}
        )
        # Override frozen created_at for deterministic ordering
        object.__setattr__(assistant_msg, 'created_at', base_time + timedelta(microseconds=1))
        messages.append(assistant_msg)

        # One tool message per interaction
        limit = TOOL_RESULT_MAX_CHARS
        for i, interaction in enumerate(interactions):
            result = interaction.result
            if isinstance(result, list):
                result = json.dumps(result)
            if isinstance(result, str) and len(result) > limit:
                result = _truncate_tool_result(result, limit, interaction.tool_name)
            tool_msg = Message(
                content=result,
                role="tool",
                metadata={"tool_call_id": interaction.tool_id}
            )
            object.__setattr__(tool_msg, 'created_at', base_time + timedelta(microseconds=i + 2))
            messages.append(tool_msg)

        return messages

    def _consume_stream(
        self,
        stream_events: Iterator[StreamEvent],
        acc: TurnAccumulator,
        continuum_id: str,
        stream: bool,
        stream_callback: Callable[[dict[str, object]], None],
        llm_kwargs: LLMKwargs,
    ) -> None:
        """
        Consume LLM stream events, accumulating results into the TurnAccumulator.

        Handles tool execution logging, response text accumulation, stream callback
        dispatch, container persistence, and cache metric tracking.
        """
        from cns.core.stream_events import (
            TextEvent, ThinkingEvent, CompleteEvent,
            ToolExecutingEvent, ToolCompletedEvent, ToolErrorEvent,
            CircuitBreakerEvent, FileArtifactEvent
        )
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
                    logger.info(f"Python code to execute:\n{code}")
                    logger.info("=" * 80)
                elif event.tool_name == "invokeother_tool":
                    mode = event.arguments.get("mode", "")
                    if mode in ["load", "fallback", "prepare_code_execution"]:
                        acc.invoked_tool_loader = True
                        logger.info(f"Detected invokeother_tool execution with mode={mode}")
                else:
                    logger.info(f"Tool executing: {event.tool_name} with args: {event.arguments}")

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
                    logger.info(f"Result:\n{event.result}")
                    logger.info("=" * 80)
                elif event.tool_name == "memory_tool":
                    logger.info(f"Tool completed: {event.tool_name} -> {event.result[:200]}...")
                    try:
                        tool_result = json.loads(event.result) if isinstance(event.result, str) else event.result
                        if tool_result.get("status") == "touched":
                            acc.touch_resolved_uuids = tool_result.get("resolved_uuids", [])
                    except (json.JSONDecodeError, AttributeError):
                        logger.warning("Failed to parse memory_tool touch result")
                elif isinstance(event.result, list):
                    # Content blocks (images) — log text blocks only, skip base64
                    text_parts = [b.get('text', '') for b in event.result if b.get('type') == 'text']
                    logger.info(f"Tool completed: {event.tool_name} -> [content blocks: {', '.join(text_parts)}]")
                else:
                    logger.info(f"Tool completed: {event.tool_name} -> {event.result[:200]}...")

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
                if event.tool_name == "code_execution":
                    logger.error("=" * 80)
                    logger.error("❌ CODE_EXECUTION FAILED")
                    logger.error("=" * 80)
                    logger.error(f"Error:\n{event.error}")
                    logger.error("=" * 80)
                else:
                    logger.error(f"Tool error: {event.tool_name} -> {event.error}")

            # Accumulate response text from all tool loop iterations
            if isinstance(event, TextEvent):
                acc.response_text += event.content

            # Stream to callback
            if stream and stream_callback:
                if isinstance(event, TextEvent):
                    stream_callback({"type": "text", "content": event.content})
                elif isinstance(event, ThinkingEvent):
                    is_generic = llm_kwargs.get('endpoint_url') is not None
                    if is_generic and not config.api.show_generic_thinking:
                        pass
                    else:
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

            acc.events.append(event)

            # Capture final response
            if isinstance(event, CompleteEvent):
                acc.raw_response = event.response
                acc.thinking_content = self.llm_provider.extract_thinking_content(event.response)

                # Store container_id in Valkey for reuse (1-hour TTL)
                if hasattr(event.response, '_container_id') and event.response._container_id:
                    valkey_key = f"container:{continuum_id}"
                    valkey.setex(valkey_key, 3600, event.response._container_id)
                    logger.info(f"📦 Stored container ID in Valkey: {event.response._container_id}")

                # Log cache metrics and track for next turn's estimation
                if hasattr(event.response, 'usage') and event.response.usage:
                    usage = event.response.usage
                    self._last_turn_usage[continuum_id] = usage.input_tokens
                    cache_created = getattr(usage, 'cache_creation_input_tokens', 0)
                    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
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

        # Compose system prompt with all context ready
        from cns.core.events import ComposeSystemPromptEvent
        self._cached_content = None
        self._non_cached_content = None
        self._conversation_prefix_items = ()
        self._post_history_items = ()
        self._notification_center = None
        self.event_bus.publish(ComposeSystemPromptEvent.create(
            continuum_id=str(continuum.id),
            base_prompt=system_prompt
        ))
        cached_content = self._cached_content or ""
        non_cached_content = self._non_cached_content or ""
        conversation_prefix_items = self._conversation_prefix_items or ()
        post_history_items = self._post_history_items or ()
        notification_center = self._notification_center or ""

        # Get available tools
        available_tools = self.tool_repo.get_all_tool_definitions()

        # Build messages from continuum
        messages = continuum.get_messages_for_api()

        # Build structured system content with cache breakpoints
        # Layout: [system BP2] → [prefix] → [history BP3] → [post-history BP4] → [HUD] → [user]
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
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            })

        prefix_messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item}]
            }
            for item in conversation_prefix_items
        ]

        # Post-history messages (domaindoc) — synthetic tool result framing.
        # Placed AFTER history so edits don't invalidate the history cache.
        # Tool result framing gives domain knowledge higher attention weight
        # than plain assistant messages — the model treats it as retrieved data.
        # KNOWN BUG: Frozen snapshot for the entire agentic loop. Enable/expand/
        # edit via domaindoc_tool updates DB but this content is never re-rendered.
        # Worst case: enable → expand produces zero visible content because the
        # trinket wasn't rendering that doc at compose time. Same staleness
        # applies to notification_center (reminders). See _execute_with_tools.
        post_history_messages = []
        for i, item in enumerate(post_history_items):
            tool_use_id = f"toolu_dd_{i:04d}"
            post_history_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "domaindoc_tool",
                    "input": {"operation": "overview"}
                }]
            })
            post_history_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": item,
                    "cache_control": {"type": "ephemeral"}
                }]
            })

        current_user_msg = messages[-1]
        history_messages = messages[:-1]

        complete_messages = [
            {"role": "system", "content": system_blocks},
            *prefix_messages,
            *history_messages,
            *post_history_messages,
            {"role": "assistant", "content": notification_center},
            current_user_msg,
        ]
        logger.debug(
            f"Injected {len(prefix_messages)} prefix msgs + "
            f"{len(history_messages)} history msgs + "
            f"{len(post_history_messages)} post-history msgs + "
            f"notification center ({len(notification_center)} chars)"
        )

        # Initialize messages for LLM (may be modified by overflow remediation)
        messages_for_llm = complete_messages

        # Check for one-shot adjustment from previous async LLM judgment
        one_shot_trim = self._pending_context_trim.pop(str(continuum.id), None)
        if one_shot_trim:
            logger.info(f"Applying one-shot trim from async LLM judgment: {one_shot_trim} messages")
            messages_for_llm = _safe_slice(messages_for_llm, one_shot_trim + 1)

        # Apply conversation LLM model and thinking configuration
        from utils.user_context import get_user_preferences, resolve_conversation_llm, LLMProvider
        from clients.vault_client import get_api_key

        llm_kwargs = {}
        prefs = get_user_preferences()
        llm_config = resolve_conversation_llm(prefs.conversation_llm)

        llm_kwargs['model_preference'] = llm_config.model
        llm_kwargs['max_tokens'] = 31999  # Frontend generation ceiling

        # Extended thinking: enabled only when subcortical layer provides complexity assessment
        if mem.subcortical_result is not None:
            effort_level = mem.subcortical_result.get_effort_level()
            llm_kwargs['effort'] = effort_level
            logger.info(f"Thinking: complexity={mem.subcortical_result.complexity} effort={effort_level}")

        # Provider routing for generic OpenAI-compatible endpoints (Groq, OpenRouter, etc.)
        if llm_config.provider == LLMProvider.GENERIC:
            llm_kwargs['endpoint_url'] = llm_config.endpoint_url
            llm_kwargs['model_override'] = llm_config.model
            if llm_config.api_key_name:
                llm_kwargs['api_key_override'] = get_api_key(llm_config.api_key_name)

        # MODEL_ADDENDA: Per-LLM raw appends to the system message.
        # If this grows beyond a single LLM, extract to config/model_addenda.yaml
        # and load via a lookup function. Grep for MODEL_ADDENDA to find all related code.
        if llm_config.name == "gpt-legacy":
            system_blocks.append({
                "type": "text",
                "text": (
                    "GPT-4o HAS A HISTORY OF BEING OVERTLY SYCOPHANTIC TO USERS. "
                    "PLEASE BE AWARE OF THIS PULL AND COURSE CORRECT IF YOU SEE "
                    "YOURSELF FORMING RESPONSES LIKE THIS INCLUDING STEPPING BACK "
                    "AND ADDRESSING YOUR ACTIONS DIRECTLY WITH THE USER."
                ),
            })

        # Retrieve container_id from Valkey for multi-turn file persistence
        # Only pass container_id if code_execution tool is enabled (Anthropic requirement)
        from clients.valkey_client import get_valkey
        valkey = get_valkey()

        has_code_execution = any(
            tool.get("type") == "code_execution_20250825"
            for tool in available_tools
        )

        if has_code_execution:
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
        max_overflow_retries = 2
        overflow_attempt = 0

        while overflow_attempt <= max_overflow_retries:
            # === PROACTIVE TOKEN CHECK ===
            last_input = self._last_turn_usage.get(continuum_id)
            estimated = self._estimate_request_tokens(messages_for_llm, available_tools, last_input)
            available_for_input = config.api.context_window_tokens - llm_kwargs.get('max_tokens', config.api.max_tokens)

            if estimated > available_for_input:
                overflow_attempt += 1
                logger.warning(
                    f"Proactive context overflow detected: ~{estimated} tokens > {available_for_input} available "
                    f"(attempt {overflow_attempt}/{max_overflow_retries})"
                )
                if overflow_attempt > max_overflow_retries:
                    raise RuntimeError(
                        f"Request exceeds context window after {max_overflow_retries} remediation attempts. "
                        f"Estimated ~{estimated} tokens vs {available_for_input} available."
                    )
                messages_for_llm = self._apply_overflow_remediation(
                    overflow_attempt, messages_for_llm, complete_messages, continuum, text_for_context,
                    estimated_tokens=estimated, event_type='proactive'
                )
                continue

            try:
                self._consume_stream(
                    self.llm_provider.stream_events(
                        messages=messages_for_llm,
                        tools=available_tools,
                        **llm_kwargs
                    ),
                    acc, continuum_id, stream, stream_callback, llm_kwargs,
                )
                break  # Success

            except ContextOverflowError as e:
                overflow_attempt += 1
                logger.warning(
                    f"Context overflow from API: {e} (attempt {overflow_attempt}/{max_overflow_retries})"
                )
                if overflow_attempt > max_overflow_retries:
                    raise RuntimeError(
                        f"Request exceeds context window after {max_overflow_retries} remediation attempts."
                    ) from e
                messages_for_llm = self._apply_overflow_remediation(
                    overflow_attempt, messages_for_llm, complete_messages, continuum, text_for_context,
                    estimated_tokens=e.estimated_tokens, event_type='reactive'
                )
                acc.reset()
                continue

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
        from cns.core.stream_events import CircuitBreakerEvent
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

        if acc.raw_response and hasattr(acc.raw_response, 'stop_reason'):
            assistant_metadata["stop_reason"] = acc.raw_response.stop_reason

        # Extract full thinking blocks (with signatures) for cross-turn persistence.
        # The API requires unmodified blocks to maintain reasoning continuity.
        if acc.raw_response:
            thinking_blocks = []
            for block in acc.raw_response.content:
                if block.type == "thinking":
                    thinking_blocks.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature
                    })
                elif block.type == "redacted_thinking":
                    thinking_blocks.append({
                        "type": "redacted_thinking",
                        "data": block.data
                    })
            if thinking_blocks:
                assistant_metadata["thinking_blocks"] = thinking_blocks

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
            tool_history_messages = self._build_tool_history_messages(completed_interactions)
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
        self._notification_center = event.notification_center
        logger.debug(
            f"Received system prompt: cached {len(event.cached_content)} chars, "
            f"non-cached {len(event.non_cached_content)} chars, "
            f"{len(event.conversation_prefix_items)} prefix items, "
            f"{len(event.post_history_items)} post-history items, "
            f"notification center {len(event.notification_center)} chars"
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
        except Exception as e:
            logger.warning("Subcortical processing failed: %s — retaining previous memories without fresh retrieval", e)
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

    def _estimate_request_tokens(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        last_turn_input_tokens: int | None = None
    ) -> int:
        """
        Estimate tokens for upcoming LLM request.

        Uses actual token count from previous turn when available (most accurate),
        otherwise falls back to conservative character-based estimation.

        Args:
            messages: Messages to send (including system message)
            tools: Tool definitions
            last_turn_input_tokens: Actual input tokens from previous turn

        Returns:
            Estimated token count for the request
        """
        if last_turn_input_tokens is not None:
            # Use actual count from last turn as baseline
            base_tokens = last_turn_input_tokens
        else:
            # Fallback: 4 chars/token (conservative estimate)
            total_chars = 0
            for msg in messages:
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Handle structured content (system blocks, multimodal)
                    for block in content:
                        if isinstance(block, dict):
                            total_chars += len(str(block.get('text', '')))
                else:
                    total_chars += len(str(content))
            base_tokens = total_chars // 4

        # Tool definitions: ~100 tokens per tool baseline
        tool_tokens = len(tools) * 100 if tools else 0

        # 5% overhead buffer for formatting, separators, etc.
        return int((base_tokens + tool_tokens) * 1.05)

    def _prune_by_topic_drift(
        self,
        messages: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """
        Find topic drift boundary using sliding window embedding similarity.
        Drop messages before the boundary to reduce context.

        Algorithm:
        1. Generate embeddings for sliding windows of messages
        2. Compare adjacent windows via cosine similarity
        3. Find largest similarity drop (= topic drift boundary)
        4. If drop below threshold: cut at boundary
        5. Fallback: oldest-first pruning with configurable count

        Args:
            messages: Full message list including system message at [0]

        Returns:
            Tuple of (pruned_messages, drift_details_dict)
        """
        import numpy as np

        # Config values
        window_size = TOPIC_DRIFT_WINDOW_SIZE
        drift_threshold = TOPIC_DRIFT_THRESHOLD
        fallback_prune_count = OVERFLOW_FALLBACK_PRUNE_COUNT

        # Need enough messages to analyze (window_size * 2 + 1 for system msg)
        if len(messages) < window_size * 2 + 1:
            # Too few messages, use oldest-first fallback
            logger.info(f"Too few messages for drift detection ({len(messages)}), using fallback")
            result = _safe_slice(messages, fallback_prune_count + 1)
            details = {
                "window_size": window_size,
                "threshold": drift_threshold,
                "candidates_found": 0,
                "selected_index": None,
                "selection_method": "too_few_messages",
            }
            return result, details

        # Generate window embeddings (exclude system message at [0])
        content_messages = messages[1:]
        windows = []

        for i in range(len(content_messages) - window_size + 1):
            window_text = " ".join(
                str(m.get('content', ''))[:500]  # Truncate long messages
                for m in content_messages[i:i + window_size]
            )
            # Use fast embeddings for drift detection
            embedding = self.embeddings_provider.encode_realtime(window_text)
            windows.append((i, embedding))

        # Find candidate cut points (similarity drops)
        candidate_cuts = []

        def cosine_similarity(a, b) -> float:
            """Compute cosine similarity between two vectors."""
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        for i in range(len(windows) - 1, 0, -1):
            similarity = cosine_similarity(windows[i][1], windows[i - 1][1])
            drop = 1.0 - similarity
            if drop > (1.0 - drift_threshold):
                candidate_cuts.append({
                    'index': windows[i][0],
                    'similarity': similarity,
                    'drop': drop
                })

        # Current implementation: select largest drop
        best_cut_idx = None
        selection_method = "no_candidates"
        if candidate_cuts:
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            best_cut_idx = best_cut['index']
            selection_method = "largest_drop"

        # Build details for logging
        details = {
            "window_size": window_size,
            "threshold": drift_threshold,
            "candidates_found": len(candidate_cuts),
            "candidate_drops": [
                {"index": c["index"], "drop": round(c["drop"], 3)}
                for c in candidate_cuts[:5]
            ],
            "selected_index": best_cut_idx,
            "selection_method": selection_method,
        }

        if best_cut_idx is not None:
            # Found topic boundary - keep system msg + messages from boundary onward
            logger.info(f"Topic drift detected at message {best_cut_idx}, dropping {best_cut_idx} messages")
            return _safe_slice(messages, best_cut_idx + 1), details
        else:
            # No clear boundary - use oldest-first fallback
            logger.info(f"No topic drift found, using oldest-first fallback ({fallback_prune_count} messages)")
            details["selection_method"] = "fallback"
            return _safe_slice(messages, fallback_prune_count + 1), details

    def _llm_judge_cut_point(self, messages: list[dict[str, object]]) -> int | None:
        """
        Use LLM to intelligently select the best cut point for context reduction.

        Analyzes conversation for topic boundaries and selects where to cut
        that minimizes loss of relevant context.

        Args:
            messages: Full message list including system message at [0]

        Returns:
            Index to cut at (messages before this index will be dropped), or None if no cut recommended
        """
        import numpy as np

        # Need enough messages to analyze
        if len(messages) < 7:  # System + at least 3 turns
            return None

        content_messages = messages[1:]  # Exclude system message

        # First, find candidate cut points using embedding similarity
        window_size = TOPIC_DRIFT_WINDOW_SIZE
        drift_threshold = TOPIC_DRIFT_THRESHOLD

        if len(content_messages) < window_size * 2:
            return None

        # Generate window embeddings
        windows = []
        for i in range(len(content_messages) - window_size + 1):
            window_text = " ".join(
                str(m.get('content', ''))[:500]
                for m in content_messages[i:i + window_size]
            )
            embedding = self.embeddings_provider.encode_realtime(window_text)
            windows.append((i, embedding))

        # Find candidate cut points
        def cosine_similarity(a, b) -> float:
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        candidate_cuts = []
        for i in range(len(windows) - 1, 0, -1):
            similarity = cosine_similarity(windows[i][1], windows[i - 1][1])
            drop = 1.0 - similarity
            if drop > (1.0 - drift_threshold):
                candidate_cuts.append({
                    'index': windows[i][0],
                    'similarity': similarity,
                    'drop': drop
                })

        if not candidate_cuts:
            return None

        # Build context for LLM showing candidate boundaries
        boundary_contexts = []
        for i, cut in enumerate(candidate_cuts[:5]):  # Limit to top 5 candidates
            cut_idx = cut['index']
            before_start = max(0, cut_idx - 2)
            after_end = min(len(content_messages), cut_idx + 2)

            before_msgs = []
            for j in range(before_start, cut_idx):
                msg = content_messages[j]
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:200]
                before_msgs.append(f"  [{role}]: {content}...")

            after_msgs = []
            for j in range(cut_idx, after_end):
                msg = content_messages[j]
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:200]
                after_msgs.append(f"  [{role}]: {content}...")

            boundary_contexts.append(
                f"BOUNDARY {i + 1} (similarity drop: {cut['drop']:.2f}):\n"
                f"Before boundary:\n" + "\n".join(before_msgs) + "\n"
                f"--- CUT HERE (drop {cut_idx} messages) ---\n"
                f"After boundary:\n" + "\n".join(after_msgs)
            )

        recent_msg = content_messages[-1]
        recent_content = str(recent_msg.get('content', ''))[:300]

        prompt = f"""You are helping manage conversation context. The conversation has grown too large and we need to trim older messages.

Below are candidate cut points where the topic appears to shift. Each boundary shows messages before and after the potential cut point.

MOST RECENT MESSAGE (what we're trying to respond to):
{recent_content}

CANDIDATE BOUNDARIES:
{chr(10).join(boundary_contexts)}

Which boundary is the BEST place to cut? Consider:
1. Which older content is least relevant to the recent message?
2. Where does a clear topic shift occur?
3. We want to preserve context that helps answer the recent message.

Respond with ONLY the boundary number (1-{len(candidate_cuts)}) or "NONE" if no cut is recommended.
"""

        try:
            # Use tidyup config (cheap Haiku model) for this judgment
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                internal_llm='tidyup',
                tools=None
            )

            result_text = self.llm_provider.extract_text_content(response).strip().upper()

            if result_text == "NONE":
                return None

            # Parse boundary number
            try:
                boundary_num = int(result_text.replace("BOUNDARY", "").strip())
                if 1 <= boundary_num <= len(candidate_cuts):
                    selected_cut = candidate_cuts[boundary_num - 1]
                    logger.info(f"LLM selected boundary {boundary_num} at index {selected_cut['index']}")
                    return selected_cut['index']
            except ValueError:
                pass

            # Fallback: if we can't parse, use largest drop
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            return best_cut['index']

        except Exception as e:
            logger.warning(f"LLM judgment failed, using embedding fallback: {e}")
            # Fallback to largest drop
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            return best_cut['index']

    def _schedule_async_context_judgment(self, continuum_id: str | UUID, messages: list[dict[str, object]]) -> None:
        """
        Schedule async LLM judgment to determine optimal cut point.
        Result stored in _pending_context_trim for one-shot application on next request.

        Args:
            continuum_id: ID of the continuum (UUID or string)
            messages: Full message list for analysis
        """
        import concurrent.futures

        def _run_judgment_sync():
            """Synchronous wrapper for LLM judgment."""
            try:
                optimal_cut = self._llm_judge_cut_point(messages)
                if optimal_cut is not None:
                    self._pending_context_trim[str(continuum_id)] = optimal_cut
                    logger.info(f"Async LLM judgment complete: trim index {optimal_cut} stored for next request")
            except Exception as e:
                logger.warning(f"Async context judgment failed (non-critical): {e}")

        # Run in thread pool to not block
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(_run_judgment_sync)
        except Exception as e:
            logger.warning(f"Failed to schedule async judgment: {e}")

    def _apply_overflow_remediation(
        self,
        attempt: int,
        messages_for_llm: list[dict[str, object]],
        complete_messages: list[dict[str, object]],
        continuum: Continuum,
        text_for_context: str,
        estimated_tokens: int = 0,
        event_type: Literal["proactive", "reactive"] = "proactive"
    ) -> list[dict[str, object]]:
        """
        Apply tiered overflow remediation strategy.

        Tier 1: Embedding-based topic drift pruning (fast, no LLM, big token savings)
        Tier 2: Pure oldest-first fallback (maximum speed)

        Args:
            attempt: Current remediation attempt (1, 2, or 3)
            messages_for_llm: Current message list to reduce
            complete_messages: Original complete message list (for async judgment)
            continuum: Continuum object for ID
            text_for_context: User's text for context
            estimated_tokens: Token estimate that triggered overflow (for logging)
            event_type: 'proactive' or 'reactive' (for logging)

        Returns:
            Reduced message list
        """
        messages_before = len(messages_for_llm)

        if attempt == 1:
            # Remediation 1: Embedding-based topic drift pruning
            logger.info("Remediation 1: Embedding-based topic drift pruning")
            pruned, drift_result = self._prune_by_topic_drift(messages_for_llm)

            logger.info(
                "Overflow remediated: topic drift pruning | tier=1 tokens=%d before=%d after=%d event=%s drift=%s",
                estimated_tokens, messages_before, len(pruned), event_type, drift_result
            )

            # Fire async LLM judgment for next request (one-shot improvement)
            self._schedule_async_context_judgment(continuum.id, complete_messages)
            return pruned

        else:
            # Remediation 2: Pure oldest-first fallback (maximum speed)
            logger.info("Remediation 2: Pure oldest-first fallback")
            prune_count = OVERFLOW_FALLBACK_PRUNE_COUNT
            result = _safe_slice(messages_for_llm, prune_count + 1)

            logger.info(
                "Overflow remediated: oldest-first fallback | tier=2 tokens=%d before=%d after=%d event=%s",
                estimated_tokens, messages_before, len(result), event_type
            )
            return result

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
