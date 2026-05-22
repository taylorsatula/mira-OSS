"""
Forage results trinket.

Displays background forage agent results in the notification center.
Handles multiple concurrent forages with status lifecycle:
  pending -> in_progress (stacked) -> success | timeout | failed
  success -> dismissed (via forage_tool dismiss action)

Success results persist until explicitly dismissed or segment collapse.
Error/timeout results auto-expire after 5 turns.
"""
import logging
from typing import Dict, Any, TYPE_CHECKING

from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

# Turns before error/timeout results auto-expire
ERROR_TTL_TURNS = 5


class ForageTrinket(StatefulTrinket):
    """
    Displays forage agent results as they arrive.

    Multiple concurrent forages are tracked independently by task_id.
    The primary LLM sees task_ids in the output so it can dismiss
    stale or unhelpful results via forage_tool's dismiss parameter.
    """

    variable_name = "forage_results"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._user_results: dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def active_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the active user's forage results. Lazy-initializes on access to support in-place mutation."""
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
        """Clear forage results for the current user on segment collapse."""
        results = self._user_results.pop(get_current_user_id(), None)
        if results:
            logger.info(f"Clearing {len(results)} forage results on segment collapse")

    def handle_update_request(self, event) -> None:
        """
        Process incoming forage results by status.

        Updates internal state, then delegates to parent for content
        generation and publication.
        """
        context = event.context
        task_id = context.get('task_id')
        status = context.get('status')

        if not task_id:
            # Lifecycle refresh from StatefulTrinket — just regenerate content
            super().handle_update_request(event)
            return

        if status == 'dismissed':
            if task_id in self.active_results:
                del self.active_results[task_id]
                logger.debug(f"Dismissed forage result {task_id[:8]}")
            super().handle_update_request(event)
            return

        if status == 'in_progress':
            # Don't downgrade terminal states (late overwatch thread)
            existing = self.active_results.get(task_id)
            if existing and existing['type'] in ('success', 'failed', 'timeout'):
                return
            # Stack summaries — accumulate per-iteration lines so Mira
            # sees the full research arc on next working memory refresh
            summary = context.get('summary', '')
            iteration = context.get('iteration', 0)
            if existing and existing['type'] == 'in_progress':
                summaries = existing['data'].get('summaries', [])
            else:
                summaries = []
            summaries.append({'iteration': iteration, 'text': summary})
            self.active_results[task_id] = {
                'type': 'in_progress',
                'data': {
                    'query': context.get('query', ''),
                    'iteration': iteration,
                    'max_iterations': context.get('max_iterations', 0),
                    'summaries': summaries,
                },
                'received_turn': self.current_turn,
            }
            super().handle_update_request(event)
            return

        if status == 'pending':
            self.active_results[task_id] = {
                'type': 'pending',
                'data': context,
                'received_turn': self.current_turn,
            }

        elif status == 'success':
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

        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate XML content showing all active forage results."""
        parts = []

        for task_id, result in self.active_results.items():
            result_type = result['type']

            if result_type == 'in_progress':
                parts.append(self._format_in_progress(task_id, result['data']))

            elif result_type == 'pending':
                parts.append(self._format_pending(task_id, result['data']))

            elif result_type == 'success':
                parts.append(self._format_success(task_id, result['data']))

            elif result_type in ('timeout', 'failed'):
                if self.current_turn <= result.get('display_until_turn', 0):
                    parts.append(self._format_error(task_id, result))

        if parts:
            return "<forage_results>\n" + "\n".join(parts) + "\n</forage_results>"
        return ""

    def _format_in_progress(self, task_id: str, data: Dict[str, Any]) -> str:
        query = data.get('query', '')
        iteration = data.get('iteration', 0)
        max_iter = data.get('max_iterations', 0)
        summaries = data.get('summaries', [])
        lines = [f"[{s['iteration']}] {s['text']}" for s in summaries]
        return (
            f'<result type="in_progress" task_id="{task_id}" query="{query}" '
            f'iteration="{iteration}/{max_iter}">\n'
            + "\n".join(lines) + "\n"
            f"</result>"
        )

    def _format_pending(self, task_id: str, data: Dict[str, Any]) -> str:
        query = data.get('query', '')
        return (
            f'<result type="pending" task_id="{task_id}" query="{query}">\n'
            f"Foraging in progress...\n"
            f"</result>"
        )

    def _format_success(self, task_id: str, data: Dict[str, Any]) -> str:
        query = data.get('query', '')
        written_result = data.get('result', '')
        iterations = data.get('iterations', 0)
        return (
            f'<result type="success" task_id="{task_id}" query="{query}" iterations="{iterations}">\n'
            f"{written_result}\n"
            f"</result>"
        )

    def _format_error(self, task_id: str, result: Dict[str, Any]) -> str:
        data = result['data']
        result_type = result['type']
        turns_remaining = result['display_until_turn'] - self.current_turn
        query = data.get('query', '')

        if result_type == 'timeout':
            elapsed = data.get('elapsed', 0)
            iteration = data.get('iteration', 0)
            return (
                f'<result type="timeout" task_id="{task_id}" query="{query}" '
                f'turns_remaining="{turns_remaining}">\n'
                f"Forage timed out after {elapsed:.0f}s at iteration {iteration}.\n"
                f"</result>"
            )

        error = data.get('error', 'Unknown error')
        error_type = data.get('error_type', 'Error')
        return (
            f'<result type="failed" task_id="{task_id}" query="{query}" '
            f'turns_remaining="{turns_remaining}">\n'
            f"Forage failed: {error_type} — {error}\n"
            f"</result>"
        )
