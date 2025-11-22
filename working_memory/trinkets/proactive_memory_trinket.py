"""Proactive memory trinket for displaying relevant long-term memories."""
import logging
from typing import List, Dict, Any

from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


class ProactiveMemoryTrinket(EventAwareTrinket):
    """
    Displays surfaced memories in working memory.
    
    This trinket formats memories passed via context into
    a structured section for the system prompt.
    """
    
    def __init__(self, event_bus, working_memory):
        """Initialize with memory cache."""
        super().__init__(event_bus, working_memory)
        self._cached_memories = []  # Store memories between updates
    
    def _get_variable_name(self) -> str:
        """Proactive memory publishes to 'relevant_memories'."""
        return "relevant_memories"
    
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate memory content from context.
        
        Args:
            context: Update context containing 'memories' list
            
        Returns:
            Formatted memories section or empty string if no memories
        """
        # Update cache if memories are provided
        if 'memories' in context:
            self._cached_memories = context['memories']
        
        # Use cached memories
        if not self._cached_memories:
            return ""
        
        # Format memories for prompt
        memory_content = self._format_memories_for_prompt(self._cached_memories)
        
        logger.debug(f"Formatted {len(self._cached_memories)} memories for display")
        return memory_content
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories with hierarchical tree structure showing linked relationships."""
        if not memories:
            return ""

        content_parts = ["=== SURFACED MEMORIES ==="]
        content_parts.append("")

        for memory in memories:
            # Format primary memory
            content_parts.extend(self._format_primary_memory(memory))
            content_parts.append("")

        return "\n".join(content_parts)

    def _format_primary_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Format a primary memory with its linked memories in tree structure."""
        lines = []

        # Memory header with optional confidence
        confidence = memory.get('confidence') or memory.get('similarity_score')
        if confidence is not None and confidence > 0.75:
            conf_percentage = int(confidence * 100)
            lines.append(f"[AUTOMATICALLY SURFACED SUBCONSCIOUS MEMORY | CONFIDENCE: {conf_percentage}%]")
        else:
            lines.append("[AUTOMATICALLY SURFACED SUBCONSCIOUS MEMORY]")

        lines.append(f"ID: {memory.get('id', 'unknown')}")
        lines.append(f"Text: \"{memory.get('text', '')}\")")

        if memory.get('created_at'):
            from datetime import datetime
            from utils.timezone_utils import format_relative_time

            # Parse ISO timestamp and format as relative time
            created_dt = datetime.fromisoformat(memory['created_at'])
            relative_time = format_relative_time(created_dt)
            lines.append(f"Created: {relative_time}")

        # Format temporal info if present
        temporal_info = self._format_temporal_info(memory)
        if temporal_info:
            lines.append(temporal_info)

        # Format linked memories
        linked_memories = memory.get('linked_memories', [])
        if linked_memories:
            lines.append("")
            lines.extend(self._format_linked_memories(linked_memories, is_last_group=True))

        return lines

    def _format_linked_memories(
        self,
        linked_memories: List[Dict[str, Any]],
        indent: str = "",
        is_last_group: bool = False,
        current_depth: int = 1,
        max_display_depth: int = 2
    ) -> List[str]:
        """
        Recursively format linked memories with tree structure.

        NOTE: max_display_depth is distinct from traversal depth:
        - Traversal depth (config.max_link_traversal_depth): How deep to walk the graph
        - Display depth (this parameter): How many levels to show in output

        We may traverse 3-4 levels deep to discover important memories,
        but only display Primary + 2 levels to avoid context window bloat.

        Args:
            linked_memories: List of linked memory dicts
            indent: Current indentation string
            is_last_group: Whether this is the last group at this level
            current_depth: Current display depth (1-indexed)
            max_display_depth: Maximum depth to display (default 2 = Primary + 2 levels)
        """
        lines = []

        # Stop display if we've reached max depth
        if current_depth > max_display_depth:
            return lines

        for i, linked in enumerate(linked_memories):
            is_last = (i == len(linked_memories) - 1)

            # Tree symbols
            if is_last:
                branch = "└─"
                continuation = "   "
            else:
                branch = "├─"
                continuation = "│  "

            # Link metadata
            link_meta = linked.get('link_metadata', {})
            link_type = link_meta.get('link_type', 'unknown')
            confidence = link_meta.get('confidence')

            # Format header with confidence only if over 75%
            if confidence is not None and confidence > 0.75:
                conf_percentage = int(confidence * 100)
                header = f"{indent}{branch} [^ LINKED MEMORY - LINK TYPE: {link_type} | CONFIDENCE: {conf_percentage}%]"
            else:
                header = f"{indent}{branch} [^ LINKED MEMORY - LINK TYPE: {link_type}]"

            lines.append(header)

            # Memory details with continuation indentation
            detail_indent = indent + continuation
            lines.append(f"{detail_indent}ID: {linked.get('id', 'unknown')}")
            lines.append(f"{detail_indent}Text: \"{linked.get('text', '')}\")")

            # Nested linked memories (recursive)
            nested_linked = linked.get('linked_memories', [])
            if nested_linked:
                lines.append(f"{detail_indent}")
                lines.extend(
                    self._format_linked_memories(
                        nested_linked,
                        indent=detail_indent,
                        is_last_group=True,
                        current_depth=current_depth + 1,
                        max_display_depth=max_display_depth
                    )
                )

            # Add spacing between siblings (but not after last one)
            if not is_last:
                lines.append(f"{indent}│")

        return lines
    
    def _format_temporal_info(self, memory: dict) -> str:
        """Format temporal metadata for a memory."""
        from utils.timezone_utils import format_datetime
        
        temporal_parts = []
        
        if memory.get('expires_at'):
            expiry_date = format_datetime(memory['expires_at'], 'date')
            temporal_parts.append(f"expires: {expiry_date}")
        
        if memory.get('happens_at'):
            event_date = format_datetime(memory['happens_at'], 'date')
            temporal_parts.append(f"happens: {event_date}")
        
        if temporal_parts:
            return f" *({', '.join(temporal_parts)})*"
        return ""