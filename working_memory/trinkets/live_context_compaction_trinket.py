"""
Conversation-prefix trinket for live active-context compaction.

The trinket renders the single Valkey-only continuation brief produced by
LiveContextCompactionService. It owns no summarization logic.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from cns.services.live_context_compaction_service import LiveContextCompactionStore
from utils.user_context import get_current_user_id
from working_memory.trinkets.base import StatefulTrinket


class LiveContextCompactionTrinket(StatefulTrinket):
    """Render the current live compaction brief before raw chat history."""

    variable_name = "context_compaction"
    cache_policy = False

    def __init__(
        self,
        event_bus: Any,
        working_memory: Any,
    ) -> None:
        self.store = LiveContextCompactionStore()
        super().__init__(event_bus, working_memory)

    def handle_update_request(self, event: Any) -> None:
        context = dict(event.context or {})
        context["continuum_id"] = event.continuum_id
        super().handle_update_request(replace(event, context=context))

    def generate_content(self, context: dict[str, Any]) -> str:
        artifact = self.store.get(get_current_user_id())
        if artifact is None:
            return ""
        if artifact.continuum_id != context.get("continuum_id"):
            return ""
        return artifact.brief_text

    def _expire_items(self) -> bool:
        """Live compaction artifacts do not expire by turn count."""
        return False

    def _clear_all_state(self) -> None:
        """Clear the Valkey artifact for the current user on segment collapse."""
        self.store.clear(get_current_user_id())
