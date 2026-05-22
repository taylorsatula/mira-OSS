"""
Event-aware base trinket class.

Provides common functionality for all trinkets to participate in the
event-driven working memory system. Persists content to Valkey for
API access and monitoring.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

from clients.valkey_client import get_valkey_client
from utils.user_context import get_current_user_id
from utils.timezone_utils import utc_now, format_utc_iso

if TYPE_CHECKING:
    from cns.core.events import UpdateTrinketEvent, TurnCompletedEvent
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

# Valkey key prefix for trinket content storage
TRINKET_KEY_PREFIX = "trinkets"


class EventAwareTrinket(ABC):
    """
    Base class for event-driven trinkets.

    Trinkets inherit from this class to:
    1. Receive update requests via UpdateTrinketEvent
    2. Generate content when requested
    3. Publish their content via TrinketContentEvent

    Subclasses must define these class attributes:
        variable_name: str  — section name in the composed prompt
        cache_policy: bool  — whether to use prompt caching (default False)
    """

    # REQUIRED — subclasses must define (validated at init)
    variable_name: str

    # Optional — subclasses override as needed
    cache_policy: bool = False

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        """
        Initialize the trinket with event bus connection.

        Args:
            event_bus: CNS event bus for publishing content
            working_memory: Working memory instance for registration

        Raises:
            TypeError: If subclass doesn't define variable_name
        """
        if not getattr(self, 'variable_name', None):
            raise TypeError(
                f"{self.__class__.__name__} must define 'variable_name' class attribute"
            )

        self.event_bus = event_bus
        self.working_memory = working_memory

        # Register with working memory
        self.working_memory.register_trinket(self)

        logger.info(f"{self.__class__.__name__} initialized and registered")

    def handle_update_request(self, event: 'UpdateTrinketEvent') -> None:
        """
        Handle an update request from working memory.

        Generates content, persists to Valkey, and publishes it. Infrastructure
        failures propagate to the event handler in core.py for proper isolation.

        Args:
            event: UpdateTrinketEvent with context
        """
        from cns.core.events import TrinketContentEvent

        # Generate content - let infrastructure failures propagate
        content = self.generate_content(event.context)

        if content and content.strip():
            # Persist to Valkey for API access
            self._persist_to_valkey(content)

            self.event_bus.publish(TrinketContentEvent.create(
                continuum_id=event.continuum_id,
                variable_name=self.variable_name,
                content=content,
                trinket_name=self.__class__.__name__,
                cache_policy=self.cache_policy,
            ))
            logger.debug(f"{self.__class__.__name__} published content ({len(content)} chars)")
        else:
            # Content is empty - clear stale data from Valkey
            self._clear_from_valkey()

    def _persist_to_valkey(self, content: str) -> None:
        """
        Persist trinket content to Valkey for API access.

        Stores content in a user-scoped hash with metadata for monitoring.
        Uses hset_with_retry for transient failure handling.

        Args:
            content: Generated trinket content
        """
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"

        value = json.dumps({
            "content": content,
            "cache_policy": self.cache_policy,
            "updated_at": format_utc_iso(utc_now())
        })

        valkey = get_valkey_client()
        valkey.hset_with_retry(hash_key, self.variable_name, value)

    def _clear_from_valkey(self) -> None:
        """
        Remove this trinket's content from Valkey.

        Called when trinket content becomes empty (e.g., segment collapse,
        TTL expiry) to prevent stale data from being served via API.
        """
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"

        valkey = get_valkey_client()
        valkey.hdel_with_retry(hash_key, self.variable_name)
    
    @abstractmethod
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate content for this trinket.

        Subclasses must implement this method to generate their
        specific content based on the provided context.

        Args:
            context: Context from UpdateTrinketEvent

        Returns:
            Generated content string or empty string if no content
        """
        ...


class StatefulTrinket(EventAwareTrinket):
    """
    Base for trinkets that maintain turn-scoped state with automatic expiry.

    Handles per-turn concerns (turn tracking, item expiry). Segment collapse
    is handled centrally by WorkingMemory._flush_stateful_trinkets() — trinkets
    never subscribe to SegmentCollapsedEvent directly.

    Subclasses implement:
    - _expire_items(): Remove stale items each turn, return True if any removed
    - _clear_all_state(): Reset all in-memory state (called by WorkingMemory on collapse)
    """

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._turn_counts: dict[str, int] = {}

        self.event_bus.subscribe('TurnCompletedEvent', self._on_turn_completed)

    @property
    def current_turn(self) -> int:
        """Current turn number for the active user (from ambient context)."""
        return self._turn_counts.get(get_current_user_id(), 0)

    @current_turn.setter
    def current_turn(self, value: int) -> None:
        """Set turn number for the active user."""
        self._turn_counts[get_current_user_id()] = value

    def clear_user_turn(self, user_id: str) -> None:
        """Remove a user's turn counter entry."""
        self._turn_counts.pop(user_id, None)

    def _on_turn_completed(self, event: 'TurnCompletedEvent') -> None:
        """Update turn counter and expire stale items."""
        self.current_turn = event.turn_number
        if self._expire_items():
            self._refresh_content()

    def _refresh_content(self) -> None:
        """Trigger a content regeneration after state changes."""
        self.working_memory.publish_trinket_update(
            target_trinket=self.__class__.__name__,
            context={"action": "lifecycle_refresh"}
        )

    @abstractmethod
    def _expire_items(self) -> bool:
        """Remove expired items. Return True if anything was removed."""
        ...

    @abstractmethod
    def _clear_all_state(self) -> None:
        """Clear all in-memory state for segment collapse."""
        ...