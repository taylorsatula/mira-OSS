"""Proactive memory trinket for displaying relevant long-term memories."""
import logging
from typing import List, Dict, Any, TYPE_CHECKING

from utils.user_context import get_current_user_id
from utils.tag_parser import format_memory_id
from .base import EventAwareTrinket

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


class ProactiveMemoryTrinket(EventAwareTrinket):
    """
    Displays surfaced memories in the notification center.

    This trinket formats memories passed via context into
    a structured section for the sliding notification center.
    """

    variable_name = "relevant_memories"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        """Initialize with per-user memory cache."""
        super().__init__(event_bus, working_memory)
        self._user_memories: dict[str, list[dict[str, Any]]] = {}

    def get_cached_memories(self) -> List[Dict[str, Any]]:
        """
        Get the currently cached memories for the active user.

        Used by the orchestrator for memory retention evaluation —
        previous turn's surfaced memories are evaluated for continued relevance.

        Returns:
            List of memory dicts from the previous turn
        """
        return self._user_memories.get(get_current_user_id(), [])
    
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate memory content from context.

        Args:
            context: Update context containing 'memories' list

        Returns:
            Formatted memories section or empty string if no memories
        """
        user_id = get_current_user_id()

        # Update cache if memories are provided
        if 'memories' in context:
            self._user_memories[user_id] = context['memories']

        memories = self._user_memories.get(user_id, [])
        if not memories:
            return ""

        # Format memories for prompt
        memory_content = self._format_memories_for_prompt(memories)

        logger.debug(f"Formatted {len(memories)} memories for display")
        return memory_content
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories as XML with nested linked_memories elements."""
        if not memories:
            return ""

        parts = ["<surfaced_memories>"]
        primary_ids = {memory.get('id', '') for memory in memories if memory.get('id')}
        seen_linked_ids = set(primary_ids)

        for memory in memories:
            parts.append(self._format_primary_memory_xml(memory, seen_linked_ids))

        parts.append("</surfaced_memories>")
        return "\n".join(parts)

    def _format_primary_memory_xml(
        self,
        memory: Dict[str, Any],
        seen_linked_ids: set[str] | None = None,
    ) -> str:
        """Format a primary memory as XML with nested linked_memories."""
        from utils.timezone_utils import format_relative_time, parse_time_string, format_datetime

        raw_id = memory.get('id', '')
        formatted_id = format_memory_id(raw_id) if raw_id else 'unknown'
        text = memory.get('text', '')

        # Build attributes
        attrs = [f'id="{formatted_id}"']

        # Mark global memories for clear identification
        if memory.get('source') == 'global':
            attrs.append('source="global"')

        similarity = memory.get('similarity_score')
        if similarity is not None and similarity > 0.75:
            attrs.append(f'relevance="{int(similarity * 100)}"')

        parts = [f"<memory {' '.join(attrs)}>"]
        parts.append(f"<text>{text}</text>")

        # Created time
        if memory.get('created_at'):
            created_dt = parse_time_string(memory['created_at'])
            relative_time = format_relative_time(created_dt)
            parts.append(f"<created>{relative_time}</created>")

        # Temporal info
        temporal_attrs = []
        if memory.get('expires_at'):
            expires_dt = parse_time_string(memory['expires_at'])
            expiry_date = format_datetime(expires_dt, 'date')
            temporal_attrs.append(f'expires="{expiry_date}"')
        if memory.get('happens_at'):
            happens_dt = parse_time_string(memory['happens_at'])
            event_date = format_datetime(happens_dt, 'date')
            temporal_attrs.append(f'happens="{event_date}"')
        if temporal_attrs:
            parts.append(f"<temporal {' '.join(temporal_attrs)}/>")

        # Annotations (contextual notes)
        annotations = memory.get('annotations', [])
        if annotations:
            parts.append("<annotations>")
            for ann in annotations:
                text = ann.get('text', '')
                created = ann.get('created_at', '')
                if created:
                    ann_dt = parse_time_string(created)
                    relative = format_relative_time(ann_dt)
                    parts.append(f'<note added="{relative}">{text}</note>')
                else:
                    parts.append(f'<note>{text}</note>')
            parts.append("</annotations>")

        # Linked memories as compact inline context
        linked_memories = memory.get('linked_memories', [])
        if linked_memories:
            linked_context = self._format_linked_context(linked_memories, seen_linked_ids)
            if linked_context:
                parts.append(linked_context)

        parts.append("</memory>")
        return "\n".join(parts)

    def _format_linked_context(
        self,
        linked_memories: List[Dict[str, Any]],
        seen_linked_ids: set[str] | None = None,
    ) -> str:
        """
        Format linked memories as compact inline context annotations.

        Deduplicates by target ID across the full surfaced-memory block. Primary
        memory IDs are pre-seeded into seen_linked_ids, so a memory already shown
        as a primary is not repeated as linked context under another primary.
        Renders ~80 chars per line instead of ~200+ chars per full XML block.
        """
        if not linked_memories:
            return ""

        if seen_linked_ids is None:
            seen_linked_ids = set()

        # Dedup by target ID — keep first occurrence per target across primaries
        seen_ids: dict = {}
        for linked in linked_memories:
            target_id = linked.get('id', '')
            if not target_id or target_id in seen_linked_ids:
                continue
            seen_ids[target_id] = linked
            seen_linked_ids.add(target_id)

        lines = []
        for linked in seen_ids.values():
            raw_id = linked.get('id', '')
            formatted_id = format_memory_id(raw_id) if raw_id else ''
            text = linked.get('text', '')
            link_meta = linked.get('link_metadata', {})
            link_type = link_meta.get('link_type', '')
            id_suffix = f", {formatted_id}" if formatted_id else ""
            lines.append(f"Also: {text} ({link_type}{id_suffix})")

        return "<context>\n" + "\n".join(lines) + "\n</context>"