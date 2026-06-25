"""
Domain events for CNS.

Immutable event objects that represent state changes in the continuum domain.
Events enable loose coupling between components.

Event Categories:
- MessageEvent: Everything between user and AI, kicks off the processing chain
- ToolEvent: Tool execution, enabling, disabling, errors
- WorkingMemoryEvent: Manager updates, content changes, trinket operations
- ContinuumCheckpointEvent: Turn completion, segment lifecycle, system coordination

Future events should fit into these categories. Only create new categories if there's
a fundamentally different type of system interaction that doesn't fit the above.
Resist the urge to create one-off events - adapt existing categories instead.
"""
from __future__ import annotations

from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from utils.timezone_utils import utc_now

if TYPE_CHECKING:
    from cns.core.continuum import Continuum

@dataclass(frozen=True, kw_only=True)
class ContinuumEvent:
    """Base class for all continuum events."""
    continuum_id: str
    user_id: str
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class MessageEvent(ContinuumEvent):
    """Events related to message processing between user and AI."""
    pass


@dataclass(frozen=True)
class ToolEvent(ContinuumEvent):
    """Events related to tool execution and management."""
    pass


@dataclass(frozen=True)
class WorkingMemoryEvent(ContinuumEvent):
    """Events related to working memory and manager updates."""
    pass


@dataclass(frozen=True)
class ContinuumCheckpointEvent(ContinuumEvent):
    """Events for system coordination: turn completion, segment lifecycle."""
    pass


# Concrete implementations we need immediately

@dataclass(frozen=True)
class TurnCompletedEvent(ContinuumCheckpointEvent):
    """
    Continuum turn completed (user message + assistant response).

    Carries a reference to the continuum object so handlers can extract
    whatever data they need (messages, metadata, tool usage, etc.) without
    the event needing to know what each handler requires.

    ARCHITECTURAL NOTE: Events describing state changes should carry the changed state.
    Requiring handlers to re-fetch state introduces race conditions - the event may be
    published before persistence completes, causing handlers to see stale data. This is
    why TurnCompletedEvent carries the continuum object directly. When events carry data,
    handlers get correct state regardless of persistence timing.

    segment_turn_number is the authoritative turn count within the current segment,
    incremented at API entry point when real user messages arrive (not synthetic
    continuation prompts or injected system messages).
    """
    turn_number: int
    segment_turn_number: int  # Turn count within current segment (1-indexed)
    continuum: Continuum

    @classmethod
    def create(cls, continuum_id: str, turn_number: int,
               segment_turn_number: int, continuum: Continuum) -> TurnCompletedEvent:
        """
        Create turn completed event with auto-generated metadata.

        Args:
            continuum_id: Continuum identifier
            turn_number: Current turn number (calculated from message count)
            segment_turn_number: Turn number within current segment (1-indexed)
            continuum: Continuum object reference for handler access
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            turn_number=turn_number,
            segment_turn_number=segment_turn_number,
            continuum=continuum
        )


@dataclass(frozen=True, kw_only=True)
class ToolResultHistoryCommittedEvent(ContinuumCheckpointEvent):
    """Tool result messages committed to PostgreSQL and the hot Valkey cache."""

    tool_messages: tuple[Message, ...]

    @classmethod
    def create(
        cls,
        continuum_id: str,
        tool_messages: list[Message],
    ) -> ToolResultHistoryCommittedEvent:
        """Create a post-commit tool-result event for the current user."""
        from utils.user_context import get_current_user_id

        return cls(
            continuum_id=continuum_id,
            user_id=get_current_user_id(),
            tool_messages=tuple(tool_messages),
        )


@dataclass(frozen=True)
class ComposeSystemPromptEvent(WorkingMemoryEvent):
    """Request to compose the system prompt with current working memory state."""
    base_prompt: str

    @classmethod
    def create(cls, continuum_id: str, base_prompt: str) -> ComposeSystemPromptEvent:
        """Create compose system prompt event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            base_prompt=base_prompt
        )


@dataclass(frozen=True)
class SystemPromptComposedEvent(WorkingMemoryEvent):
    """System prompt has been composed and is ready for use."""
    cached_content: str
    non_cached_content: str
    conversation_prefix_items: tuple[str, ...]  # Each becomes assistant message before history
    post_history_items: tuple[str, ...]  # Each becomes assistant message after history (BP4)
    notification_center: dict[str, str]  # Named fields: section_name -> XML content

    @classmethod
    def create(
        cls,
        continuum_id: str,
        cached_content: str,
        non_cached_content: str,
        conversation_prefix_items: tuple[str, ...] = (),
        post_history_items: tuple[str, ...] = (),
        notification_center: dict[str, str] | None = None,
    ) -> SystemPromptComposedEvent:
        """Create system prompt composed event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            cached_content=cached_content,
            non_cached_content=non_cached_content,
            conversation_prefix_items=conversation_prefix_items,
            post_history_items=post_history_items,
            notification_center=notification_center
        )


@dataclass(frozen=True)
class UpdateTrinketEvent(WorkingMemoryEvent):
    """Request a specific trinket to update its content."""
    target_trinket: str
    context: dict[str, object]

    @classmethod
    def create(cls, continuum_id: str, target_trinket: str, context: dict[str, object] | None = None) -> UpdateTrinketEvent:
        """Create update trinket event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            target_trinket=target_trinket,
            context=context or {}
        )


@dataclass(frozen=True)
class TrinketContentEvent(WorkingMemoryEvent):
    """Trinket has published content for the system prompt."""
    variable_name: str
    content: str
    trinket_name: str
    cache_policy: bool = False

    @classmethod
    def create(
        cls,
        continuum_id: str,
        variable_name: str,
        content: str,
        trinket_name: str,
        cache_policy: bool = False,
    ) -> TrinketContentEvent:
        """Create trinket content event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            variable_name=variable_name,
            content=content,
            trinket_name=trinket_name,
            cache_policy=cache_policy,
        )


@dataclass(frozen=True)
class SegmentTimeoutEvent(ContinuumCheckpointEvent):
    """Inactivity threshold reached, trigger segment collapse."""
    segment_id: str
    inactive_duration_minutes: int
    local_hour: int  # User's local time for context

    @classmethod
    def create(cls, continuum_id: str, user_id: str, segment_id: str,
               inactive_duration_minutes: int, local_hour: int) -> SegmentTimeoutEvent:
        """
        Create segment timeout event with auto-generated metadata.

        Args:
            continuum_id: Continuum identifier
            user_id: User identifier
            segment_id: Segment identifier
            inactive_duration_minutes: Duration segment has been inactive
            local_hour: User's local hour for context
        """
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            segment_id=segment_id,
            inactive_duration_minutes=inactive_duration_minutes,
            local_hour=local_hour
        )


@dataclass(frozen=True)
class SegmentCollapsedEvent(ContinuumCheckpointEvent):
    """
    Segment collapsed into manifest with summary generated.

    Subscribers:
    - ForageTrinket: Clears all forage results when segment collapses
      (prevents stale forage results from leaking into new conversation contexts)
    """
    segment_id: str
    summary: str
    tools_used: list[str]

    @classmethod
    def create(cls, continuum_id: str, segment_id: str,
               summary: str, tools_used: list[str]) -> SegmentCollapsedEvent:
        """Create segment collapsed event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            segment_id=segment_id,
            summary=summary,
            tools_used=tools_used
        )


@dataclass(frozen=True)
class ManifestUpdatedEvent(ContinuumCheckpointEvent):
    """Manifest structure changed, notify cache invalidation."""
    segment_count: int

    @classmethod
    def create(cls, continuum_id: str, segment_count: int) -> ManifestUpdatedEvent:
        """Create manifest updated event with auto-generated metadata."""
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        return cls(
            continuum_id=continuum_id,
            user_id=user_id,
            event_id=str(uuid4()),
            occurred_at=utc_now(),
            segment_count=segment_count
        )
