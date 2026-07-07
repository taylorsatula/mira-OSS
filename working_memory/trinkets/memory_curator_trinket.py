"""
Memory curator trinket.

Displays MemoryCuratorAgent results — what the curator linked, merged, or
archived — so the primary LLM and operator can see curation activity.

Lifecycle:
  success   -> persists until segment collapse (or next curation run replaces it)
  timeout/failed -> auto-expires after 5 turns (same as ForageTrinket)

The trinket is keyed by task_id (one per curation run). Integration and floor
runs are tracked independently.
"""
import logging
from typing import Dict, Any, TYPE_CHECKING

from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

ERROR_TTL_TURNS = 5


class MemoryCuratorTrinket(StatefulTrinket):
    """
    Displays memory curator results as they arrive.

    Multiple curation runs are tracked independently by task_id. The primary
    LLM sees the per-memory decisions (linked / merged / archived / stand-alone)
    so it knows the current state of the memory graph.
    """

    variable_name = "memory_curation"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._user_results: dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def active_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the active user's curator results."""
        user_id = get_current_user_id()
        if user_id not in self._user_results:
            self._user_results[user_id] = {}
        return self._user_results[user_id]

    def _expire_items(self) -> bool:
        """Remove error/timeout results past their display window."""
        expired = [
            task_id for task_id, result in self.active_results.items()
            if result['type'] in ('timeout', 'failed')
            and self.current_turn > result.get('display_until_turn', 0)
        ]
        for task_id in expired:
            del self.active_results[task_id]
        return bool(expired)

    def _clear_all_state(self) -> None:
        """Clear curator results for the current user on segment collapse."""
        results = self._user_results.pop(get_current_user_id(), None)
        if results:
            logger.info(f"Clearing {len(results)} curator results on segment collapse")

    def handle_update_request(self, event) -> None:
        """Process incoming curator completion events by status."""
        context = event.context
        task_id = context.get('task_id')
        status = context.get('status')

        if not task_id:
            super().handle_update_request(event)
            return

        if status == 'success':
            self.active_results[task_id] = {
                'type': 'success',
                'data': context,
                'received_turn': self.current_turn,
            }
        elif status == 'timeout':
            self.active_results[task_id] = {
                'type': 'timeout',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + ERROR_TTL_TURNS,
            }
        elif status == 'failed':
            self.active_results[task_id] = {
                'type': 'failed',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + ERROR_TTL_TURNS,
            }
        else:
            # Unknown status — ignore
            return

        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate XML content showing all active curator results."""
        parts = []

        for task_id, result in self.active_results.items():
            result_type = result['type']

            if result_type == 'success':
                parts.append(self._format_success(task_id, result['data']))
            elif result_type in ('timeout', 'failed'):
                if self.current_turn <= result.get('display_until_turn', 0):
                    parts.append(self._format_error(task_id, result))

        if parts:
            return "<memory_curation>\n" + "\n".join(parts) + "\n</memory_curation>"
        return ""

    def _format_success(self, task_id: str, data: Dict[str, Any]) -> str:
        mode = data.get('mode', 'unknown')
        summary = data.get('summary', '')
        segment_id = data.get('segment_id', '')

        header = f'<result type="success" task_id="{task_id}" mode="{mode}"'
        if segment_id:
            header += f' segment="{segment_id}"'
        header += '>'

        return f"{header}\n{summary}\n</result>"

    def _format_error(self, task_id: str, result: Dict[str, Any]) -> str:
        data = result['data']
        result_type = result['type']
        turns_remaining = result['display_until_turn'] - self.current_turn
        mode = data.get('mode', 'unknown')

        if result_type == 'timeout':
            return (
                f'<result type="timeout" task_id="{task_id}" mode="{mode}" '
                f'turns_remaining="{turns_remaining}">\n'
                f"Memory curator timed out.\n"
                f"</result>"
            )

        return (
            f'<result type="failed" task_id="{task_id}" mode="{mode}" '
            f'turns_remaining="{turns_remaining}">\n'
            f"Memory curator failed.\n"
            f"</result>"
        )
