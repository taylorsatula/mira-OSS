"""Type definitions for working memory structures.

TypedDicts for the well-known dict shapes produced by composer.py and core.py.
"""
from typing import TypedDict


class ComposedPrompt(TypedDict):
    """Output of SystemPromptComposer.compose()."""
    cached_content: str
    non_cached_content: str
    conversation_prefix_items: list[str]
    post_history_items: list[str]
    notification_center: dict[str, str]  # Named fields: section_name -> XML content


class TrinketState(TypedDict):
    """Single trinket's cached state from Valkey (returned by get_trinket_state)."""
    section_name: str
    content: str
    cache_policy: bool
    last_updated: str | None


class TrinketStatesMeta(TypedDict):
    """Metadata block within AllTrinketStates."""
    trinket_count: int
    loaded_at: str


class AllTrinketStates(TypedDict):
    """Output of WorkingMemory.get_all_trinket_states()."""
    trinkets: list[TrinketState]
    meta: TrinketStatesMeta
