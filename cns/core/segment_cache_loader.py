"""
Segment Cache Loader for CNS.

Loads context for new sessions with segment summaries and session boundary markers.
"""
import logging
from datetime import datetime
from typing import List, Optional

from cns.core.message import Message
from cns.services.segment_helpers import create_collapse_marker, create_session_boundary_marker
from config import config

logger = logging.getLogger(__name__)


class SegmentCacheLoader:
    """
    Loads messages for new sessions with session boundary markers.

    When a continuum expires from Valkey cache (new session), this loads
    recent messages and adds a session boundary marker.
    """

    def __init__(self, repository):
        """
        Initialize the cache manager.

        Args:
            repository: Continuum repository for persistence
        """
        self.repository = repository

    def load_session_cache(self, continuum_id: str, user_id: str) -> List[Message]:
        """
        Load cache for a new session with boundary marker.

        When a session expires (after 1 hour idle), this loads:
        1. Collapse marker (indicates older messages available through search)
        2. Collapsed segment summaries (past conversations)
        3. Last 3 user/assistant turns from before the active segment (continuity)
        4. Session boundary (marks where the break occurred)
        5. Active segment messages (unconsolidated current conversation)

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            [collapse_marker, summaries, continuity_turns, session_boundary, active_messages]
        """
        # Set user context for private methods to use
        from utils.user_context import set_current_user_id
        set_current_user_id(user_id)

        # Step 1: Load collapsed segment summaries (past conversations)
        # Uses complexity-based selection for optimal information density
        segment_summaries = self._load_segment_summaries(continuum_id)

        # Step 2: Load continuity messages (last 3 turns before active sentinel)
        continuity_messages = self._load_continuity_messages(continuum_id, turn_count=3)

        # Step 3: Create collapse marker to indicate older searchable content
        collapse_marker = create_collapse_marker()

        # Step 4: Load active segment messages (current unconsolidated conversation)
        active_segment_messages = self._load_active_segment_messages(continuum_id)

        # Step 5: Create session boundary marking the break
        boundary = create_session_boundary_marker(segment_summaries)

        # Step 6: Assemble in order - collapse marker first, then summaries, continuity, boundary, and active messages
        messages = [collapse_marker] + segment_summaries + continuity_messages + [boundary] + active_segment_messages

        logger.info(
            f"Loaded session cache for continuum {continuum_id}: "
            f"collapse marker + {len(segment_summaries)} summaries + "
            f"{len(continuity_messages)} continuity + boundary + {len(active_segment_messages)} active"
        )

        return messages

    def _load_segment_summaries(self, continuum_id: str) -> List[Message]:
        """
        Load recent collapsed segment sentinels using complexity-based selection.

        Queries the most recent segments and selects based on accumulated complexity
        score to optimize information density within context window constraints.

        Args:
            continuum_id: Continuum ID

        Returns:
            List of collapsed segment sentinels in chronological order

        Raises:
            DatabaseError: If database query fails
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()

        messages = self._select_summaries_by_complexity(continuum_id, user_id)

        logger.debug(
            f"Loaded {len(messages)} segment summaries for continuum {continuum_id} "
            f"using complexity-based selection"
        )
        return messages

    def _select_summaries_by_complexity(
        self,
        continuum_id: str,
        user_id: str
    ) -> List[Message]:
        """
        Select segment summaries based on complexity score accumulation.

        Queries a window of recent collapsed segments and iteratively selects
        them from newest to oldest, accumulating complexity scores until the
        total exceeds the configured limit or max count is reached.

        Complexity scores optimize information density - a few complex segments
        provide more value than many simple ones within the same token budget.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            Selected segment summaries in chronological order (oldest to newest)

        Raises:
            DatabaseError: If database query fails
        """
        # Get configuration values
        complexity_limit = config.system.session_summary_complexity_limit
        max_count = config.system.session_summary_max_count
        query_window = config.system.session_summary_query_window

        # Query window of recent segments (newest first)
        candidates = self.repository.find_collapsed_segments(
            continuum_id,
            user_id,
            limit=query_window
        )

        if not candidates:
            logger.debug(f"No collapsed segments found for continuum {continuum_id}")
            return []

        # Select segments by accumulating complexity
        selected = []
        total_complexity = 0

        for segment in candidates:
            # Get complexity score, default to 2 (moderate) if missing
            complexity = segment.metadata.get('complexity_score', 2)

            # Check if adding this segment would exceed limit
            # Allow if we have 0 segments (always include at least one)
            if total_complexity + complexity > complexity_limit and len(selected) > 0:
                logger.debug(
                    f"Stopping selection: total_complexity={total_complexity}, "
                    f"next_complexity={complexity} would exceed limit={complexity_limit}"
                )
                break

            # Add segment
            selected.append(segment)
            total_complexity += complexity

            # Check if we've reached max count
            if len(selected) >= max_count:
                logger.debug(
                    f"Stopping selection: reached max_count={max_count}"
                )
                break

        logger.info(
            f"Selected {len(selected)}/{len(candidates)} segments "
            f"(total_complexity={total_complexity}/{complexity_limit}, "
            f"limit={'complexity' if len(selected) < max_count else 'count'})"
        )

        # Return in chronological order (oldest first)
        # Candidates are newest-first, so reverse the selected slice
        return list(reversed(selected))

    def _load_active_segment_messages(self, continuum_id: str) -> List[Message]:
        """
        Load all messages from the active segment.

        Active segment is one that hasn't been collapsed yet (status='active').
        Returns all real conversation messages after the active sentinel.

        Args:
            continuum_id: Continuum ID

        Returns:
            List of messages in chronological order, or empty list if no active segment

        Raises:
            DatabaseError: If database query fails
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()

        # Find the active segment sentinel
        active_sentinel = self.repository.find_active_segment(continuum_id, user_id)

        if not active_sentinel:
            logger.debug(f"No active segment found for continuum {continuum_id}")
            return []

        # Load all messages after the sentinel
        messages = self.repository.load_segment_messages(continuum_id, user_id, active_sentinel.created_at)

        logger.debug(f"Loaded {len(messages)} active segment messages for continuum {continuum_id}")
        return messages

    def _load_continuity_messages(
        self,
        continuum_id: str,
        turn_count: int
    ) -> List[Message]:
        """
        Load last N user/assistant turns before the active segment sentinel.

        This provides conversational continuity by showing the tail end of the
        previous collapsed segment. Working backwards from the sentinel, we find
        the last N assistant messages and their corresponding user messages.

        Args:
            continuum_id: Continuum ID
            turn_count: Number of user/assistant pairs to load

        Returns:
            Last N user/assistant message pairs in chronological order

        Raises:
            DatabaseError: If database query fails
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()
        messages = self.repository.load_continuity_messages(continuum_id, user_id, turn_count)

        logger.debug(
            f"Loaded {len(messages)} continuity messages "
            f"({len(messages) // 2} turns) before active sentinel"
        )

        return messages
