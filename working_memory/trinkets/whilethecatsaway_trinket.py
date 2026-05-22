"""
WhileTheCatsAway results trinket.

Displays background curiosity research results in the notification center.
Results auto-expire after a configurable number of turns — they're
informational summaries of what the agent learned and stored, not
working documents that need explicit dismissal.

Lifecycle: success | timeout | failed → auto-expire after RESULT_TTL_TURNS.
"""
import logging
from typing import Dict, Any, TYPE_CHECKING

from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

# All results auto-expire — these are informational, not actionable.
RESULT_TTL_TURNS = 8


class WhileTheCatsAwayTrinket(StatefulTrinket):
    """
    Surfaces whilethecatsaway agent results: what was explored,
    what was learned, and what memories were stored.

    Results auto-expire after RESULT_TTL_TURNS turns. No dismiss
    mechanism — the trinket clears itself.
    """

    variable_name = "whilethecatsaway_results"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._user_results: dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def active_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the active user's results. Lazy-initializes on access to support in-place mutation."""
        user_id = get_current_user_id()
        if user_id not in self._user_results:
            self._user_results[user_id] = {}
        return self._user_results[user_id]

    def _expire_items(self) -> bool:
        """Remove results past their display window."""
        expired = [
            task_id for task_id, result in self.active_results.items()
            if self.current_turn > result.get('display_until_turn', 0)
        ]
        for task_id in expired:
            del self.active_results[task_id]
        return bool(expired)

    def _clear_all_state(self) -> None:
        """Clear results for the current user on segment collapse."""
        results = self._user_results.pop(get_current_user_id(), None)
        if results:
            logger.info(
                f"Clearing {len(results)} "
                "whilethecatsaway results on segment collapse"
            )

    def handle_update_request(self, event) -> None:
        """Process incoming agent results by status."""
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
                'display_until_turn': self.current_turn + RESULT_TTL_TURNS,
            }

        elif status == 'timeout':
            self.active_results[task_id] = {
                'type': 'timeout',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + RESULT_TTL_TURNS,
            }

        elif status == 'failed':
            self.active_results[task_id] = {
                'type': 'failed',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + RESULT_TTL_TURNS,
            }

        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate XML content showing completed research results."""
        parts = []

        for task_id, result in self.active_results.items():
            if self.current_turn > result.get('display_until_turn', 0):
                continue

            result_type = result['type']
            data = result['data']

            if result_type == 'success':
                parts.append(self._format_success(task_id, data))
            else:
                parts.append(self._format_error(task_id, result))

        if parts:
            return (
                "<whilethecatsaway_results>\n"
                + "\n".join(parts)
                + "\n</whilethecatsaway_results>"
            )
        return ""

    def _format_success(self, task_id: str, data: Dict[str, Any]) -> str:
        topic = data.get('topic', '')
        summary = data.get('result', '')
        return (
            f'<result type="success" task_id="{task_id}" '
            f'topic="{topic}">\n'
            f"{summary}\n"
            f"</result>"
        )

    def _format_error(self, task_id: str, result: Dict[str, Any]) -> str:
        data = result['data']
        result_type = result['type']
        topic = data.get('topic', '')
        turns_remaining = result['display_until_turn'] - self.current_turn

        if result_type == 'timeout':
            return (
                f'<result type="timeout" task_id="{task_id}" '
                f'topic="{topic}" '
                f'turns_remaining="{turns_remaining}">\n'
                f"Background research timed out.\n"
                f"</result>"
            )

        error = data.get('error', 'Unknown error')
        return (
            f'<result type="failed" task_id="{task_id}" '
            f'topic="{topic}" '
            f'turns_remaining="{turns_remaining}">\n'
            f"Background research failed: {error}\n"
            f"</result>"
        )
