"""
Continuum aggregate root for CNS.

Immutable continuum entity that encapsulates business logic
and state transitions without external dependencies.
"""
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
from uuid import UUID, uuid4

from .message import Message
from .state import ContinuumState
from .events import ContinuumEvent

logger = logging.getLogger(__name__)


class Continuum:
    """
    Continuum aggregate root.

    Immutable entity that manages continuum state and business rules.
    All state changes return new continuum instances and domain events.
    """

    def __init__(self, state: ContinuumState):
        """Initialize continuum with state."""
        self._state = state
        self._message_cache = []  # Hot cache of recent messages
        self._cumulative_tokens = 0  # Running token count for cache breakpoint calculation
        self._cached_up_to_tokens = 0  # Tracks how many tokens are already cached
        self._thinking_budget_preference: Optional[int] = None  # User's thinking budget preference for this conversation

    @classmethod
    def create_new(cls, user_id: str) -> 'Continuum':
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
    def messages(self) -> List[Message]:
        """Get cached messages - must be initialized through ContinuumPool."""
        return self._message_cache

    @property
    def last_touchstone(self) -> Optional[str]:
        """Get last generated touchstone from metadata."""
        return self._state.metadata.get('last_touchstone')

    @property
    def last_touchstone_embedding(self) -> Optional[List[float]]:
        """Get last generated touchstone embedding from metadata."""
        return self._state.metadata.get('last_touchstone_embedding')

    @property
    def thinking_budget_preference(self) -> Optional[int]:
        """
        Get user's thinking budget preference for this conversation.

        Returns:
            None: Use system config default
            0: Explicitly disable thinking
            Positive int: Explicitly enable with specific budget
        """
        return self._thinking_budget_preference

    def set_thinking_budget_preference(self, budget: Optional[int]) -> None:
        """
        Set user's thinking budget preference for this conversation.

        Args:
            budget: None (system default), 0 (disabled), or positive int (enabled with budget)

        Raises:
            ValueError: If budget is negative or not a valid value
        """
        if budget is not None and budget < 0:
            raise ValueError("Thinking budget must be None, 0, or a positive integer")
        self._thinking_budget_preference = budget
        logger.debug(f"Set thinking budget preference to {budget} for continuum {self.id}")

    def apply_cache(self, messages: List[Message]) -> None:
        """
        Apply an externally managed cache update.
        
        Used by hot cache manager to update the cache after operations
        like topic-based pruning and summary insertion.
        
        Args:
            messages: New message cache to apply
        """
        self._message_cache = messages
    
    def add_user_message(self, content: Union[str, List[Dict[str, Any]]]) -> tuple[Message, List[ContinuumEvent]]:
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
    
    def add_assistant_message(self, content: str, metadata: dict = None) -> tuple[Message, List[ContinuumEvent]]:
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
    
    def add_tool_message(self, content: str, tool_call_id: str) -> List[ContinuumEvent]:
        """
        Add tool result message to continuum.

        Returns:
            List of domain events (empty for tool messages)
        """
        # Create message
        message = Message(
            content=content,
            role="tool",
            metadata={"tool_call_id": tool_call_id}
        )

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        # Tool messages don't generate events by themselves
        return []

    def set_last_touchstone(self, touchstone: Union[str, Dict[str, str]], embedding: Optional[List[float]] = None) -> None:
        """
        Set last generated touchstone and its embedding in metadata.

        The touchstone is a semantic summary used for memory retrieval and is evolved
        across continuum turns. The embedding enables continuum-level similarity
        search and clustering. Both are stored in metadata for persistence and caching.

        Args:
            touchstone: Touchstone text to store
            embedding: Optional 384-dim embedding vector for similarity search
        """
        # Create new metadata dict with touchstone and embedding
        metadata = self._state.metadata.copy()
        metadata['last_touchstone'] = touchstone
        if embedding is not None:
            metadata['last_touchstone_embedding'] = embedding

        # Update state with new metadata
        self._state = ContinuumState(
            id=self._state.id,
            user_id=self._state.user_id,
            metadata=metadata
        )

    def add_tokens(self, token_count: int) -> None:
        """
        Add tokens to cumulative count for cache breakpoint calculation.

        Args:
            token_count: Number of tokens to add (typically from API response usage)
        """
        self._cumulative_tokens += token_count

    def mark_cached(self) -> None:
        """
        Mark that all cumulative tokens have been successfully cached.

        Called after a successful API response when cache_control was applied.
        """
        self._cached_up_to_tokens = self._cumulative_tokens
        logger.debug(f"Marked {self._cached_up_to_tokens} tokens as cached")
    
    def get_messages_for_api(self) -> List[dict]:
        """Get messages formatted for LLM API with proper prefixes and cache control."""
        from cns.services.segment_helpers import format_segment_for_display

        formatted_messages = []

        for message in self.messages:  # This uses the property which handles cache loading
            # Format content based on message type
            content = message.content

            # Apply display formatting for collapsed segments
            if (message.metadata.get('is_segment_boundary') and
                message.metadata.get('status') == 'collapsed'):
                content = format_segment_for_display(message)

            if message.role == "assistant" and message.metadata.get("has_tool_calls", False):
                # Assistant message with tool calls
                msg_dict = {
                    "role": "assistant",
                    "content": content
                }
                if "tool_calls" in message.metadata:
                    msg_dict["tool_calls"] = message.metadata["tool_calls"]
                formatted_messages.append(msg_dict)
            elif message.role == "tool":
                # Tool result message
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": message.metadata.get("tool_call_id"),
                    "content": content
                })
            elif message.role == "user" and isinstance(message.content, list):
                # User message with content blocks (multimodal)
                formatted_messages.append({
                    "role": "user",
                    "content": message.content  # Keep original for multimodal
                })
            else:
                # Standard text message
                formatted_messages.append({
                    "role": message.role,
                    "content": content
                })

        # Apply ultra-fine-grained caching: cache everything once we hit 1024 tokens
        # and have new tokens to cache
        if self._cumulative_tokens >= 1024 and self._cumulative_tokens > self._cached_up_to_tokens:
            # Find last assistant message and mark it for caching
            for i in range(len(formatted_messages) - 1, -1, -1):
                if formatted_messages[i]["role"] == "assistant":
                    content = formatted_messages[i]["content"]

                    # Ensure content is structured as blocks (required for cache_control)
                    if isinstance(content, str):
                        # Convert string content to structured block
                        content = [{"type": "text", "text": content}]

                    # Apply cache_control to the last content block
                    if isinstance(content, list) and len(content) > 0:
                        content[-1]["cache_control"] = {"type": "ephemeral"}
                        formatted_messages[i]["content"] = content
                        logger.debug(
                            f"Applied ultra-fine cache control to message {i} "
                            f"(will cache {self._cumulative_tokens} tokens, "
                            f"previously cached: {self._cached_up_to_tokens})"
                        )
                    break

        return formatted_messages
    
    def to_dict(self) -> dict:
        """Convert continuum to dictionary for persistence."""
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'Continuum':
        """Create continuum from dictionary."""
        state = ContinuumState.from_dict(data)
        return cls(state)