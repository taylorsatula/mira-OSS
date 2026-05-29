"""
Continuum aggregate root for CNS.

Mutable aggregate that encapsulates business logic and state transitions.
ContinuumState is frozen, but the aggregate itself maintains a mutable
message cache that is appended to during processing and replaced on
session reload.
"""
from __future__ import annotations

import logging
from uuid import UUID, uuid4

from .message import Message, MessageMetadata, ContentBlock
from .state import ContinuumState, ContinuumStateDict
from .events import ContinuumEvent

logger = logging.getLogger(__name__)


class Continuum:
    """
    Continuum aggregate root.

    Manages continuum state and message cache. ContinuumState is frozen,
    but the message cache is mutable (appended during turns, replaced on reload).
    """

    def __init__(self, state: ContinuumState):
        """Initialize continuum with state."""
        self._state = state
        self._message_cache: list[Message] = []  # Hot cache of recent messages

    @classmethod
    def create_new(cls, user_id: str) -> Continuum:
        """Create a new continuum for user."""
        state = ContinuumState(
            id=uuid4(),
            user_id=user_id
        )
        return cls(state)
    
    @property
    def id(self) -> UUID:
        """Get continuum ID."""
        return self._state.id

    @property
    def user_id(self) -> str:
        """Get user ID."""
        return self._state.user_id

    @property
    def messages(self) -> list[Message]:
        """Get cached messages - must be initialized through ContinuumPool."""
        return self._message_cache

    def apply_cache(self, messages: list[Message]) -> None:
        """
        Apply an externally managed cache update.
        
        Used by segment cache loader to update the cache after operations
        like segment reconstruction and message loading.
        
        Args:
            messages: New message cache to apply
        """
        self._message_cache = messages
    
    def add_user_message(self, content: str | list[ContentBlock]) -> tuple[Message, list[ContinuumEvent]]:
        """
        Add user message to continuum.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        # Create message with original content for processing
        message = Message(content=content, role="user")

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        return message, []
    
    def add_assistant_message(self, content: str, metadata: MessageMetadata | None = None) -> tuple[Message, list[ContinuumEvent]]:
        """
        Add assistant message to continuum.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        # Validate content is not blank
        if not content or not content.strip():
            raise ValueError("Assistant message content cannot be blank or empty")

        # Create message
        message = Message(content=content, role="assistant", metadata=metadata or {})

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        return message, []
    
    def add_tool_message(self, content: str | list[ContentBlock], tool_call_id: str, is_error: bool = False) -> list[ContinuumEvent]:
        """
        Add tool result message to continuum.

        Returns:
            List of domain events (empty for tool messages)
        """
        # Create message
        message = Message(
            content=content,
            role="tool",
            tool_call_id=tool_call_id,
            is_error=is_error,
        )

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        # Tool messages don't generate events by themselves
        return []

    def add_tool_history(self, messages: list[Message]) -> None:
        """
        Insert tool history messages into the cache.

        Called after turn completion to persist tool use/result pairs
        so subsequent turns can see what tools returned.
        """
        self._message_cache.extend(messages)


    def to_dict(self) -> ContinuumStateDict:
        """Convert continuum to dictionary for persistence."""
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: ContinuumStateDict) -> Continuum:
        """Create continuum from dictionary."""
        state = ContinuumState.from_dict(data)
        return cls(state)
