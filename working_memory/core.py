"""
Event-driven working memory core.

Coordinates trinkets and system prompt composition through CNS events.
All operations are synchronous - events are published and handled immediately.
"""
import json
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from clients.valkey_client import get_valkey_client
from utils.user_context import get_current_user_id, get_user_preferences, set_current_user_id
from utils.timezone_utils import utc_now, format_utc_iso, format_relationship_duration
from .composer import SystemPromptComposer
from .trinkets.base import EventAwareTrinket, TRINKET_KEY_PREFIX, StatefulTrinket
from .types import AllTrinketStates, TrinketState

if TYPE_CHECKING:
    from cns.core.events import (
        ComposeSystemPromptEvent,
        SegmentCollapsedEvent,
        TrinketContentEvent,
        UpdateTrinketEvent,
    )
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Event-driven working memory coordinator.
    
    This class orchestrates the system prompt composition by:
    1. Managing trinket registrations
    2. Routing UpdateTrinketEvent to specific trinkets
    3. Collecting trinket content via TrinketContentEvent
    4. Composing and publishing the final system prompt
    
    All operations are synchronous to maintain simplicity.
    """
    
    def __init__(self, event_bus: 'EventBus'):
        """
        Initialize working memory with event bus connection.

        Args:
            event_bus: CNS event bus for subscribing and publishing
        """
        self.event_bus = event_bus
        self.composer = SystemPromptComposer()
        self._trinkets: Dict[str, EventAwareTrinket] = {}
        self._current_continuum_id: Optional[str] = None
        self._portrait_cache: dict[str, str] = {}

        # Subscribe to core events
        # System prompt composition request
        self.event_bus.subscribe('ComposeSystemPromptEvent', self._handle_compose_prompt)
        
        # Trinket update routing
        self.event_bus.subscribe('UpdateTrinketEvent', self._handle_update_trinket)
        
        # Trinket content collection
        self.event_bus.subscribe('TrinketContentEvent', self._handle_trinket_content)

        # Centralized segment collapse flush
        self.event_bus.subscribe('SegmentCollapsedEvent', self._flush_stateful_trinkets)

        logger.debug("Subscribed to working memory events")
        logger.info("WorkingMemory initialized")
        
    def register_trinket(self, trinket: EventAwareTrinket) -> None:
        """
        Register a trinket with working memory.

        Args:
            trinket: Trinket instance to register
        """
        trinket_name = trinket.__class__.__name__
        self._trinkets[trinket_name] = trinket
        logger.info(f"Registered trinket: {trinket_name}")
    
    def _handle_compose_prompt(self, event: 'ComposeSystemPromptEvent') -> None:
        """
        Handle request to compose system prompt.

        This triggers all trinkets to update and then composes the final prompt.
        """
        from cns.core.events import UpdateTrinketEvent, SystemPromptComposedEvent
        
        # Store context for future trinket updates
        self._current_continuum_id = event.continuum_id

        # Substitute template variables in system prompt
        prefs = get_user_preferences()
        first_name = (prefs.first_name or '').strip() or "friend"
        logger.debug(f"Personalizing system prompt for '{first_name}'")

        personalized_prompt = event.base_prompt.replace("{first_name}", first_name)

        user_id = get_current_user_id()
        if user_id not in self._portrait_cache:
            from cns.services.portrait_service import read_portrait
            portrait = read_portrait(user_id)
            self._portrait_cache[user_id] = portrait or ""
        portrait = self._portrait_cache[user_id]
        portrait_text = f"\n{portrait}" if portrait else ""
        personalized_prompt = personalized_prompt.replace("{user_context}", portrait_text)

        # Replace {relative time since account creation} with computed duration
        if prefs.created_at:
            duration = format_relationship_duration(prefs.created_at)
        else:
            duration = "some time"
        personalized_prompt = personalized_prompt.replace(
            "{relative time since account creation}", duration
        )

        # Resolve current model substrate (re-resolved each turn so mid-chat model switches are visible)
        from utils.user_context import resolve_conversation_llm
        llm_config = resolve_conversation_llm(prefs.conversation_llm)
        personalized_prompt = personalized_prompt.replace("{model_id}", llm_config.model)
        personalized_prompt = personalized_prompt.replace("{model_name}", llm_config.description)

        # Set base prompt
        self.composer.set_base_prompt(personalized_prompt)
        
        # Clear previous sections except base
        self.composer.clear_sections(preserve_base=True)

        # Request updates from all registered trinkets
        for trinket_name in self._trinkets.keys():
            self.event_bus.publish(UpdateTrinketEvent.create(
                continuum_id=event.continuum_id,
                target_trinket=trinket_name,
                context={}
            ))
        
        # After all trinkets have updated (synchronously), compose the prompt
        structured = self.composer.compose()

        # Publish composed prompt event with structured content
        self.event_bus.publish(SystemPromptComposedEvent.create(
            continuum_id=event.continuum_id,
            cached_content=structured['cached_content'],
            non_cached_content=structured['non_cached_content'],
            conversation_prefix_items=tuple(structured['conversation_prefix_items']),
            post_history_items=tuple(structured['post_history_items']),
            notification_center=structured['notification_center']
        ))

        nc_chars = sum(len(v) for v in structured['notification_center'].values())
        logger.info(
            f"Composed system prompt: cached {len(structured['cached_content'])} chars, "
            f"non-cached {len(structured['non_cached_content'])} chars, "
            f"{len(structured['conversation_prefix_items'])} prefix items, "
            f"{len(structured['post_history_items'])} post-history items, "
            f"HUD {len(structured['notification_center'])} fields ({nc_chars} chars)"
        )
    
    def _handle_update_trinket(self, event: 'UpdateTrinketEvent') -> None:
        """
        Route update request to specific trinket.

        Event handler continues processing even if individual trinkets fail,
        but distinguishes infrastructure failures from logic errors for observability.
        """
        trinket = self._trinkets.get(event.target_trinket)
        if not trinket:
            logger.warning(f"No trinket registered with name: {event.target_trinket}")
            return

        try:
            trinket.handle_update_request(event)
            logger.debug(f"Routed update to {event.target_trinket}")
        except Exception as e:
            # Event handler continues - isolate trinket failures
            # Use exception type to distinguish infrastructure from logic errors
            error_type = type(e).__name__
            if 'Database' in error_type or 'Valkey' in error_type or 'Connection' in error_type:
                logger.error(
                    f"Infrastructure failure in trinket {event.target_trinket}: {e}",
                    exc_info=True,
                    extra={'error_category': 'infrastructure'}
                )
            else:
                logger.error(
                    f"Trinket {event.target_trinket} failed: {e}",
                    exc_info=True,
                    extra={'error_category': 'logic'}
                )
    
    def _handle_trinket_content(self, event: 'TrinketContentEvent') -> None:
        """
        Handle trinket content updates.

        Trinkets publish their sections which we add to the composer.
        Placement is determined by SECTION_LAYOUT in the composer.
        """

        self.composer.add_section(
            event.variable_name,
            event.content,
            cache_policy=event.cache_policy,
        )
        logger.debug(f"Received content for '{event.variable_name}' from {event.trinket_name}")
    
    def _flush_stateful_trinkets(self, event: 'SegmentCollapsedEvent') -> None:
        """
        Centralized flush of all stateful trinkets on segment collapse.

        Clears in-memory state and Valkey cache for every StatefulTrinket,
        scoped to the current user. Also invalidates the portrait cache
        so re-synthesized portraits are picked up on next compose.
        """
        user_id = get_current_user_id()
        flushed = []
        for name, trinket in self._trinkets.items():
            if isinstance(trinket, StatefulTrinket):
                trinket._clear_all_state()
                trinket.clear_user_turn(user_id)
                trinket._clear_from_valkey()
                flushed.append(name)

        self.invalidate_portrait(user_id)

        if flushed:
            logger.info(f"Flushed {len(flushed)} stateful trinkets on segment collapse: {', '.join(flushed)}")

    def invalidate_portrait(self, user_id: str) -> None:
        """Invalidate the portrait cache for a user after external mutation."""
        self._portrait_cache.pop(user_id, None)

    def invalidate_trinket(self, trinket_name: str, user_id: str) -> None:
        """
        Invalidate a trinket's Valkey cache for a specific user.

        Used by external services (lora_service, etc.) after persisting changes
        that would otherwise require a restart or segment collapse to pick up.
        No-op if the trinket name is unknown.
        """
        trinket = self._trinkets.get(trinket_name)
        if not isinstance(trinket, EventAwareTrinket):
            logger.warning("Cannot invalidate unknown trinket: %s", trinket_name)
            return

        # _clear_from_valkey reads user_id from contextvar — temporarily swap it
        prev_user_id = get_current_user_id()
        try:
            set_current_user_id(user_id)
            trinket._clear_from_valkey()
        finally:
            if prev_user_id is not None:
                set_current_user_id(prev_user_id)

    def publish_trinket_update(self, target_trinket: str, context: dict[str, Any] | None = None) -> None:
        """
        Publish an update request for a specific trinket.

        This is used when external events need to trigger specific trinket updates.
        For example, when memories are surfaced, we update ProactiveMemoryTrinket.
        
        Args:
            target_trinket: Name of the trinket class to update
            context: Optional context data for the trinket
        """
        if not self._current_continuum_id:
            logger.warning("No active continuum context for trinket update")
            return
        
        from cns.core.events import UpdateTrinketEvent

        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=self._current_continuum_id,
            target_trinket=target_trinket,
            context=context or {}
        ))
    
    def get_trinket(self, name: str) -> EventAwareTrinket | None:
        """
        Get a registered trinket by name.

        Used when components need direct access to trinket state (e.g., orchestrator
        accessing ProactiveMemoryTrinket's cached memories for retention evaluation).

        Args:
            name: Trinket class name (e.g., 'ProactiveMemoryTrinket')

        Returns:
            Trinket instance or None if not found
        """
        return self._trinkets.get(name)

    def get_trinket_state(self, section_name: str) -> TrinketState | None:
        """
        Get cached state of a single trinket from Valkey.

        Args:
            section_name: The trinket's section name (e.g., 'user_information')

        Returns:
            Dict with section_name, content, cache_policy, last_updated
            or None if not found
        """
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"

        valkey = get_valkey_client()
        json_value = valkey.hget_with_retry(hash_key, section_name)

        if json_value is None:
            return None

        try:
            data = json.loads(json_value)
            return {
                "section_name": section_name,
                "content": data.get("content", ""),
                "cache_policy": data.get("cache_policy", False),
                "last_updated": data.get("updated_at")
            }
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in trinket cache for section: {section_name}")
            return None

    def get_all_trinket_states(self) -> AllTrinketStates:
        """
        Get cached state of all trinkets from Valkey.

        Queries each section individually and compiles results.

        Returns:
            Dict with trinkets list and metadata
        """
        user_id = get_current_user_id()
        hash_key = f"{TRINKET_KEY_PREFIX}:{user_id}"

        valkey = get_valkey_client()
        section_names = valkey._client.hkeys(hash_key)

        trinkets = []
        for section_name in section_names:
            state = self.get_trinket_state(section_name)
            if state:
                trinkets.append(state)

        return {
            "trinkets": trinkets,
            "meta": {
                "trinket_count": len(trinkets),
                "loaded_at": format_utc_iso(utc_now())
            }
        }