"""
System Prompt Composer

Handles in-memory composition of system prompts by collecting
sections from trinkets and assembling them in a defined order.
"""
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComposerConfig:
    """Configuration for the system prompt composer."""
    section_order: List[str] = field(default_factory=lambda: [
        'base_prompt',           # Cached (static system prompt)
        'user_information',      # Cached (user's overarching knowledge - name, preferences, etc.)
        'conversation_manifest', # Cached (segment summaries - changes infrequently)
        # IMPORTANT: All cached trinkets (cache_policy=True) must be sequential above this line
        # for Claude's prefix caching to work efficiently. Non-cached trinkets go below.
        'datetime_section',
        'active_reminders',
        'punchclock_status',
        'tool_guidance',
        'relevant_memories',
        'workflow_guidance',
        'temporal_context'
    ]) # NOTE: These can't be autoconfigured bc we're explicitly setting composition order
    section_separator: str = "\n\n"
    strip_empty_sections: bool = True


class SystemPromptComposer:
    """
    Composes system prompts by collecting and ordering sections.

    This composer provides a clean interface for trinkets to contribute
    sections and handles the final assembly in a predictable order.
    """

    def __init__(self, config: Optional[ComposerConfig] = None):
        """
        Initialize the composer with configuration.

        Args:
            config: Composer configuration. If None, uses defaults.
        """
        self.config = config or ComposerConfig()
        self._sections: Dict[str, str] = {}
        self._cache_policies: Dict[str, bool] = {}  # Track cache policy per section

        logger.info(f"SystemPromptComposer initialized with {len(self.config.section_order)} ordered sections")
    
    def set_base_prompt(self, prompt: str) -> None:
        """
        Set the base system prompt.

        Args:
            prompt: Base system prompt that always appears first
        """
        self._sections['base_prompt'] = prompt
        self._cache_policies['base_prompt'] = True  # Base prompt is always cached
        logger.debug(f"Set base prompt ({len(prompt)} chars, cache=True)")
    
    def add_section(self, name: str, content: str, cache_policy: bool = False) -> None:
        """
        Add or update a section for the system prompt.

        Args:
            name: Section name (e.g., 'datetime_section', 'active_reminders')
            content: Section content (can include formatting)
            cache_policy: Whether this section should be cached (default False)
        """
        if not content or not content.strip():
            logger.debug(f"Skipping empty section '{name}'")
            return

        self._sections[name] = content
        self._cache_policies[name] = cache_policy
        logger.debug(f"Added section '{name}' ({len(content)} chars, cache={cache_policy})")
    
    def remove_section(self, name: str) -> bool:
        """
        Remove a section from the composer.
        
        Args:
            name: Section name to remove
            
        Returns:
            True if section was removed, False if not found
        """
        if name in self._sections:
            del self._sections[name]
            logger.debug(f"Removed section '{name}'")
            return True
        return False
        
    ## @CLAUDE: Only mentioned once. Dead code?
    
    def clear_sections(self, preserve_base: bool = True) -> None:
        """
        Clear all sections.

        Args:
            preserve_base: If True, keeps the base_prompt section
        """
        base_prompt = self._sections.get('base_prompt') if preserve_base else None
        base_cache_policy = self._cache_policies.get('base_prompt') if preserve_base else None
        self._sections.clear()
        self._cache_policies.clear()
        if base_prompt:
            self._sections['base_prompt'] = base_prompt
            self._cache_policies['base_prompt'] = base_cache_policy
        logger.debug(f"Cleared sections (preserved_base={preserve_base})")
    
    def compose(self) -> Dict[str, str]:
        """
        Compose the system prompt into structured sections grouped by cache policy.

        Returns:
            Dictionary with 'cached_content' and 'non_cached_content' keys
        """
        if not self._sections:
            logger.warning("No sections to compose! (Something is amiss. This means no base prompt too.")
            return {"cached_content": "", "non_cached_content": ""}

        # Separate sections by cache policy while maintaining order
        cached_parts = []
        non_cached_parts = []

        # Process sections in configured order
        for section_name in self.config.section_order:
            if section_name in self._sections:
                content = self._sections[section_name]
                if self.config.strip_empty_sections and not content.strip():
                    continue

                # Group by cache policy
                cache_policy = self._cache_policies.get(section_name, False)
                if cache_policy:
                    cached_parts.append(content)
                else:
                    non_cached_parts.append(content)

        # Add any sections not in the configured order
        extra_sections = set(self._sections.keys()) - set(self.config.section_order)
        if extra_sections:
            logger.warning(f"Found sections not in configured order: {extra_sections}")
            for section_name in sorted(extra_sections):
                content = self._sections[section_name]
                if self.config.strip_empty_sections and not content.strip():
                    continue

                cache_policy = self._cache_policies.get(section_name, False)
                if cache_policy:
                    cached_parts.append(content)
                else:
                    non_cached_parts.append(content)

        # Join and clean each group
        cached_content = self._clean_content(self.config.section_separator.join(cached_parts))
        non_cached_content = self._clean_content(self.config.section_separator.join(non_cached_parts))

        logger.info(f"Composed structured prompt: {len(cached_parts)} cached sections ({len(cached_content)} chars), {len(non_cached_parts)} non-cached sections ({len(non_cached_content)} chars)")

        return {
            "cached_content": cached_content,
            "non_cached_content": non_cached_content
        }

    def _clean_content(self, content: str) -> str:
        """Clean up excessive whitespace in content."""
        # Replace 3+ newlines with exactly 2 newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def get_sections(self) -> Dict[str, str]:
        """
        Get all current sections.
        
        Returns:
            Dictionary of section names to content
        """
        return self._sections.copy()
        
    ## @CLAUDE: This is also only referenced once and its in a testfile
    
    def get_section_names(self) -> List[str]:
        """
        Get names of all current sections.
        
        Returns:
            List of section names currently in composer
        """
        return list(self._sections.keys())
        
    ## @CLAUDE: Also maybe dead code
    
    def has_section(self, name: str) -> bool:
        """
        Check if a section exists.
        
        Args:
            name: Section name to check
            
        Returns:
            True if section exists
        """
        return name in self._sections

    ## @CLAUDE: Also maybe dead code. idk.