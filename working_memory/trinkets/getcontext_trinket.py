"""
GetContext search results trinket.

Displays asynchronous context search results when they become available.
Handles multiple concurrent searches, displays errors for 5 turns,
and clears all state on segment collapse.
"""
import json
import logging
from typing import Dict, Any

from working_memory.trinkets.base import EventAwareTrinket
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class GetContextTrinket(EventAwareTrinket):
    """
    Displays context search results from getcontext_tool.

    This trinket:
    - Handles multiple concurrent searches in the same segment
    - Displays success results until segment collapse
    - Displays error/timeout messages for 5 turns
    - Clears all state when segment collapses
    """

    def __init__(self, event_bus, working_memory):
        """Initialize with state tracking and event subscriptions."""
        super().__init__(event_bus, working_memory)

        # State tracking for multiple concurrent results
        self.active_results = {}  # task_id -> {type, data, received_turn, display_until_turn}
        self.current_turn = 0

        # Subscribe to events
        self.event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)
        self.event_bus.subscribe('SegmentCollapsedEvent', self._handle_segment_collapsed)

        logger.info("GetContextTrinket initialized with multi-result support and segment cleanup")

    def _get_variable_name(self) -> str:
        """GetContext publishes to 'context_search_results'."""
        return "context_search_results"

    def handle_update_request(self, event) -> None:
        """
        Process incoming search results (success/timeout/failure).

        Stores result in active_results state, then triggers content generation.
        This allows multiple searches to accumulate and display together.
        """
        context = event.context
        task_id = context.get('task_id')
        status = context.get('status', 'success')

        if not task_id:
            logger.warning("Received update without task_id, ignoring")
            return

        # Store result in state based on type
        if status == 'pending':
            self.active_results[task_id] = {
                'type': 'pending',
                'data': context,
                'received_turn': self.current_turn
            }
            logger.debug(f"Stored pending result for task {task_id[:8]}")

        elif status == 'success':
            self.active_results[task_id] = {
                'type': 'success',
                'data': context['summary'],
                'received_turn': self.current_turn
            }
            logger.debug(f"Stored success result for task {task_id[:8]}")

        elif status == 'timeout':
            self.active_results[task_id] = {
                'type': 'timeout',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + 5
            }
            logger.debug(f"Stored timeout result for task {task_id[:8]}, will display for 5 turns")

        elif status == 'failed':
            self.active_results[task_id] = {
                'type': 'failed',
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': self.current_turn + 5
            }
            logger.debug(f"Stored failure result for task {task_id[:8]}, will display for 5 turns")

        # Trigger content generation and publication
        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate content showing ALL active search results.

        Formats successes and currently-visible errors with clear separation.
        Context parameter is ignored - we format all active state.
        """
        parts = []

        # Collect all displayable results
        for task_id, result in list(self.active_results.items()):
            if result['type'] == 'pending':
                parts.append(self._format_pending_result(result['data']))

            elif result['type'] == 'success':
                parts.append(self._format_success_result(result['data']))

            elif result['type'] in ['timeout', 'failed']:
                # Only show if still within display window
                if self.current_turn <= result.get('display_until_turn', 0):
                    parts.append(self._format_error_result(result))

        # Join with clear separator
        if parts:
            return "\n\n---\n\n".join(parts)

        return ""

    def _format_success_result(self, summary: Dict[str, Any]) -> str:
        """Format successful search results."""
        header = f"📎 Context Search: {summary.get('query', 'Unknown query')}"

        # Get the summary content
        summary_text = summary.get('summary', '')

        # Format key findings
        findings = []
        for finding in summary.get('key_findings', []):
            finding_text = f"• {finding.get('point', '')}"
            if finding.get('source'):
                finding_text += f" (source: {finding['source']})"
            findings.append(finding_text)

        # Combine parts
        parts = [header]
        if summary_text:
            parts.append(summary_text)
        if findings:
            parts.append("\n" + "\n".join(findings))
        if summary.get('limitations'):
            parts.append(f"\nNote: {summary['limitations']}")

        return "\n\n".join(parts)

    def _format_error_result(self, result: Dict[str, Any]) -> str:
        """Format error result with remaining display time."""
        data = result['data']
        result_type = result['type']
        turns_remaining = result['display_until_turn'] - self.current_turn

        if result_type == 'timeout':
            return (
                f"⚠️ Context search timed out: '{data['query']}'\n\n"
                f"The search exceeded 5 minutes (stopped at iteration {data['iteration']} "
                f"with {data['findings_count']} findings). "
                f"Try rephrasing your query or tell the user the search was inconclusive.\n\n"
                f"(This message will disappear in {turns_remaining} turn{'s' if turns_remaining != 1 else ''})"
            )
        else:  # failed
            return (
                f"⚠️ Context search failed: '{data['query']}'\n\n"
                f"Error: {data['error']} ({data['error_type']})\n\n"
                f"Try rephrasing the query or tell the user you encountered an error.\n\n"
                f"(This message will disappear in {turns_remaining} turn{'s' if turns_remaining != 1 else ''})"
            )

    def _format_pending_result(self, data: Dict[str, Any]) -> str:
        """Format pending/searching result."""
        query = data.get('query', 'Unknown query')
        search_scope = data.get('search_scope', [])
        search_mode = data.get('search_mode', 'standard')

        scope_text = ', '.join(search_scope) if search_scope else 'all sources'

        return (
            f"🔍 Searching for context: '{query}'\n\n"
            f"Mode: {search_mode} | Sources: {scope_text}\n\n"
            f"Results will appear here when the search completes..."
        )

    def _handle_turn_completed(self, event) -> None:
        """
        Handle turn completion: update turn counter and cleanup expired errors.

        Errors display for 5 turns after received, then auto-cleanup.
        Success results stay until segment collapse.
        """
        self.current_turn = event.turn_number

        # Find expired error messages (past their display window)
        expired = [
            task_id for task_id, result in self.active_results.items()
            if result['type'] in ['timeout', 'failed']
            and self.current_turn > result.get('display_until_turn', 0)
        ]

        # Remove expired errors
        for task_id in expired:
            del self.active_results[task_id]
            logger.debug(f"Cleaned up expired error for task {task_id[:8]}")

        # If we cleaned up anything, trigger a content refresh
        if expired:
            self.working_memory.publish_trinket_update(
                target_trinket="GetContextTrinket",
                context={"action": "cleanup_completed"}
            )

    def _handle_segment_collapsed(self, event) -> None:
        """
        Clear all search results when segment collapses.

        This ensures old search results don't leak into new segments.
        Segment collapse means a new conversation context begins, so
        previous search results are no longer relevant.
        """
        if self.active_results:
            logger.info(
                f"Clearing {len(self.active_results)} search results on segment collapse "
                f"(segment_id: {event.segment_id})"
            )
            self.active_results.clear()
            self.current_turn = 0

            # Trigger content refresh to clear from working memory
            self.working_memory.publish_trinket_update(
                target_trinket="GetContextTrinket",
                context={"action": "segment_collapsed"}
            )

    def cleanup(self) -> None:
        """Clean up trinket resources."""
        self.active_results.clear()
        super().cleanup()
