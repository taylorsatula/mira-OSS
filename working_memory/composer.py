"""
System Prompt Composer

Handles in-memory composition of system prompts by collecting
sections from trinkets and assembling them in a defined order.
"""
import logging
import re
from typing import Dict, List, NamedTuple

from working_memory.types import ComposedPrompt

logger = logging.getLogger(__name__)

# Placement constants
PLACEMENT_SYSTEM = "system"
PLACEMENT_CONVERSATION_PREFIX = "conversation_prefix"
PLACEMENT_POST_HISTORY = "post_history"
PLACEMENT_NOTIFICATION = "notification"

# Centralized layout — one place to control placement and ordering.
# Sections not listed here still appear (in system, at end) with a warning,
# so new trinkets work immediately without touching this file.
SECTION_LAYOUT: Dict[str, List[str]] = {
    PLACEMENT_SYSTEM: [
        'base_prompt',
        'behavioral_directives',
        'tool_availability',
        'location_context',
        'conversation_manifest',
    ],
    PLACEMENT_CONVERSATION_PREFIX: [
        'context_compaction',
    ],
    PLACEMENT_POST_HISTORY: [
        'domaindoc',
    ],
    PLACEMENT_NOTIFICATION: [
        'datetime_section',
        'async_activity',
        'active_reminders',
        'inbox_status',
        'forage_results',
        'whilethecatsaway_results',
        'relevant_memories',
        'peanutgallery_guidance',
    ],
}


class SectionData(NamedTuple):
    """Data for a single section."""
    content: str
    cache_policy: bool


SECTION_SEPARATOR = "\n\n---\n\n"
STRIP_EMPTY_SECTIONS = True


class SystemPromptComposer:
    """
    Composes system prompts by collecting and ordering sections.

    This composer provides a clean interface for trinkets to contribute
    sections and handles the final assembly in a predictable order.
    Placement and ordering are defined by SECTION_LAYOUT.
    """

    def __init__(self):
        """Initialize the composer."""
        self._sections: Dict[str, SectionData] = {}

        logger.info("SystemPromptComposer initialized")

    def set_base_prompt(self, prompt: str) -> None:
        """
        Set the base system prompt.

        Args:
            prompt: Base system prompt that always appears first
        """
        # Add delimiter after base prompt to visually separate from injected content
        delimiter = "═" * 60
        scaffolding_note = (
            "Everything after this delimiter is part of MIRA's scaffolding, "
            "injected to provide additional context during the reply."
        )
        delimited_prompt = f"{prompt}\n\n{delimiter}\n{scaffolding_note}\n{delimiter}"

        self._sections['base_prompt'] = SectionData(
            content=delimited_prompt,
            cache_policy=True,
        )
        logger.debug(f"Set base prompt ({len(prompt)} chars)")

    def add_section(
        self,
        name: str,
        content: str,
        cache_policy: bool = False,
    ) -> None:
        """
        Add or update a section.

        Args:
            name: Section name (e.g., 'datetime_section', 'active_reminders')
            content: Section content (can include formatting)
            cache_policy: Whether this section should be cached (default False)
        """
        if not content or not content.strip():
            logger.debug(f"Skipping empty section '{name}'")
            return

        self._sections[name] = SectionData(
            content=content,
            cache_policy=cache_policy,
        )
        logger.debug(f"Added section '{name}' ({len(content)} chars)")

    def clear_sections(self, preserve_base: bool = True) -> None:
        """
        Clear all sections.

        Args:
            preserve_base: If True, keeps the base_prompt section
        """
        base_data = self._sections.get('base_prompt') if preserve_base else None
        self._sections.clear()
        if base_data:
            self._sections['base_prompt'] = base_data
        logger.debug(f"Cleared sections (preserved_base={preserve_base})")

    def compose(self) -> ComposedPrompt:
        """
        Compose system prompt, prefix, post-history, and notification center content.

        Placement and ordering are driven by SECTION_LAYOUT. Sections not in the
        layout default to system placement at the end, with a warning.

        Returns:
            Dictionary with:
            - 'cached_content': Static system prompt content (base_prompt, behavioral_directives, tool_availability)
            - 'non_cached_content': Non-cached system content
            - 'conversation_prefix_items': Pre-history assistant messages
            - 'post_history_items': Post-history assistant messages (domaindoc, BP4)
            - 'notification_center': Dict mapping section names to XML content for HUD injection
        """
        if not self._sections:
            logger.warning("No sections to compose! This means no base prompt.")
            return {
                "cached_content": "",
                "non_cached_content": "",
                "conversation_prefix_items": [],
                "post_history_items": [],
                "notification_center": {}
            }

        cached_parts = []
        non_cached_parts = []
        conversation_prefix_items = []
        post_history_items = []
        processed = set()

        # Route sections by SECTION_LAYOUT (defines both placement and ordering)
        for placement, names in SECTION_LAYOUT.items():
            for name in names:
                if name not in self._sections:
                    continue
                processed.add(name)
                section = self._sections[name]
                if STRIP_EMPTY_SECTIONS and not section.content.strip():
                    continue
                self._route_section(
                    placement, section,
                    cached_parts, non_cached_parts,
                    conversation_prefix_items, post_history_items,
                )

        # Stragglers: sections not in SECTION_LAYOUT default to system
        for name, section in self._sections.items():
            if name in processed:
                continue
            logger.warning(f"Section '{name}' not in SECTION_LAYOUT, appending to system")
            if STRIP_EMPTY_SECTIONS and not section.content.strip():
                continue
            self._route_section(
                PLACEMENT_SYSTEM, section,
                cached_parts, non_cached_parts,
                conversation_prefix_items, post_history_items,
            )

        # Build notification center as named-field dict from SECTION_LAYOUT order
        notification_center = self._build_notification_center(SECTION_LAYOUT[PLACEMENT_NOTIFICATION])

        # Join and clean system prompt content
        cached_content = self._clean_content(SECTION_SEPARATOR.join(cached_parts))
        non_cached_content = self._clean_content(SECTION_SEPARATOR.join(non_cached_parts))

        nc_chars = sum(len(v) for v in notification_center.values())
        logger.info(
            f"Composed: {len(cached_parts)} cached ({len(cached_content)} chars), "
            f"{len(non_cached_parts)} non-cached ({len(non_cached_content)} chars), "
            f"{len(conversation_prefix_items)} prefix items, "
            f"{len(post_history_items)} post-history items, "
            f"{len(notification_center)} notification fields ({nc_chars} chars)"
        )

        return {
            "cached_content": cached_content,
            "non_cached_content": non_cached_content,
            "conversation_prefix_items": conversation_prefix_items,
            "post_history_items": post_history_items,
            "notification_center": notification_center
        }

    def _route_section(
        self,
        placement: str,
        section: SectionData,
        cached_parts: List[str],
        non_cached_parts: List[str],
        conversation_prefix_items: List[str],
        post_history_items: List[str],
    ) -> None:
        """Route a section to the appropriate output list based on placement.

        Notification sections are not routed here — they're collected directly
        from self._sections by _build_notification_center().
        """
        if placement == PLACEMENT_POST_HISTORY:
            post_history_items.append(section.content)
        elif placement == PLACEMENT_CONVERSATION_PREFIX:
            conversation_prefix_items.append(section.content)
        elif placement == PLACEMENT_NOTIFICATION:
            # Notification sections handled by _build_notification_center()
            pass
        elif section.cache_policy:
            cached_parts.append(section.content)
        else:
            non_cached_parts.append(section.content)

    def _build_notification_center(self, section_names: List[str]) -> Dict[str, str]:
        """
        Build notification center as a dict mapping section names to their XML content.

        Only sections with non-empty content are included. Sections appear in
        SECTION_LAYOUT order so the model sees them consistently.

        Args:
            section_names: Ordered list of section names (from PLACEMENT_NOTIFICATION)

        Returns:
            Dict mapping section name -> XML content, empty dict if no content
        """
        result: Dict[str, str] = {}
        for name in section_names:
            section = self._sections.get(name)
            if section and section.content.strip():
                result[name] = section.content
        return result

    def _clean_content(self, content: str) -> str:
        """Clean up excessive whitespace in content."""
        # Replace 3+ newlines with exactly 2 newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
