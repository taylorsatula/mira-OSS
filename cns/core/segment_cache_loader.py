"""
Segment Cache Loader for CNS.

Loads context for new sessions with segment summaries and session boundary markers.
"""
from __future__ import annotations

import logging

from cns.core.message import Message
from cns.infrastructure.continuum_repository import ContinuumRepository
from cns.services.segment_helpers import create_collapse_marker, create_session_boundary_marker

logger = logging.getLogger(__name__)

# Two-tier session cache settings
SESSION_SUMMARY_COMPLEXITY_LIMIT = 4.5  # Max total complexity for Tier 1 extended summaries
SESSION_SUMMARY_MAX_COUNT = 4           # Max Tier 1 extended summaries
SESSION_SUMMARY_QUERY_WINDOW = 14       # Recent segments to query for selection
SESSION_PRECIS_MAX_COUNT = 4            # Max Tier 2 precis-only summaries


class SegmentCacheLoader:
    """
    Loads messages for new sessions with session boundary markers.

    When a continuum expires from Valkey cache (new session), this loads
    recent messages and adds a session boundary marker.
    """

    def __init__(self, repository: ContinuumRepository):
        """
        Initialize the cache manager.

        Args:
            repository: Continuum repository for persistence
        """
        self.repository = repository
        self._primer_turns = self._load_behavioral_primer()

    def _load_behavioral_primer(self) -> list[Message]:
        """Load behavioral primer dialogue and create Message objects.

        Parsed once at init. Messages are reused across all load_session_cache()
        calls — they're never persisted, so shared instances are fine.
        """
        from config.prompts.loader import load_prompt
        raw = load_prompt("behavioral_primer.txt")
        messages = []
        for block in raw.split("---"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n", 1)
            role = lines[0].strip("[] \n")
            content = lines[1].strip() if len(lines) > 1 else ""
            messages.append(Message(
                content=content,
                role=role,
                metadata={'system_notification': True, 'notification_type': 'behavioral_primer'}
            ))
        return messages

    def load_session_cache(self, continuum_id: str, user_id: str) -> list[Message]:
        """
        Load cache for a new session with boundary marker.

        When a session expires (after 1 hour idle), this loads:
        1. Collapse marker (indicates older messages available through search)
        2. Collapsed segment summaries (past conversations)
        3. Last 5 user/assistant turns from before the active segment (continuity)
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

        # Step 2: Load continuity messages (last 5 turns before active sentinel)
        continuity_messages = self._load_continuity_messages(continuum_id, turn_count=2)

        # Step 3: Create collapse marker to indicate older searchable content
        collapse_marker = create_collapse_marker()

        # Step 4: Load active segment messages (current unconsolidated conversation)
        active_segment_messages = self._load_active_segment_messages(continuum_id)

        # Step 5: Create session boundary marking the break
        boundary = create_session_boundary_marker(segment_summaries)

        # Step 5.5: Behavioral primer (only when collapsed segments provide conversation history)
        primer_turns = self._primer_turns if segment_summaries else []

        # Step 6: Assemble in order - collapse marker first, then summaries, primer, continuity, boundary, and active messages
        messages = [collapse_marker] + segment_summaries + primer_turns + continuity_messages + [boundary] + active_segment_messages

        logger.info(
            f"Loaded session cache for continuum {continuum_id}: "
            f"collapse marker + {len(segment_summaries)} summaries + "
            f"{len(primer_turns)} primer + {len(continuity_messages)} continuity + "
            f"boundary + {len(active_segment_messages)} active"
        )

        return messages

    def _load_segment_summaries(self, continuum_id: str) -> list[Message]:
        """
        Load recent collapsed segment sentinels using two-tier selection.

        Tier 1: Extended summaries selected by accumulated complexity score.
        Tier 2: Additional segments using precis only for broader lookback.

        Tier 2 entries are marked with display_mode='precis' in metadata
        (ephemeral — not persisted) so the display layer formats them differently.

        Args:
            continuum_id: Continuum ID

        Returns:
            List of collapsed segment sentinels in chronological order

        Raises:
            DatabaseError: If database query fails
        """
        from utils.user_context import get_current_user_id
        user_id = get_current_user_id()

        tier1, tier2 = self._select_two_tier_summaries(continuum_id, user_id)

        # Mark segments with display_mode for format routing (ephemeral, not persisted)
        marked_tier1 = [s.with_metadata(display_mode='extended') for s in tier1]
        marked_tier2 = [s.with_metadata(display_mode='precis') for s in tier2]

        # Tier 2 holds older segments (precis), Tier 1 holds newer ones (extended).
        # Concatenate oldest-first so the combined list is chronological end-to-end —
        # downstream consumers (session boundary marker, LLM narrative reading) rely
        # on segment_summaries[-1] being the most recent segment.
        messages = marked_tier2 + marked_tier1

        logger.info(
            f"Loaded {len(tier1)} extended + {len(marked_tier2)} precis summaries "
            f"for continuum {continuum_id}"
        )
        return messages

    def _select_two_tier_summaries(
        self,
        continuum_id: str,
        user_id: str
    ) -> tuple[list[Message], list[Message]]:
        """
        Two-tier segment selection for session cache.

        Tier 1: Extended summaries up to complexity limit (full synopsis).
        Tier 2: Additional segments using precis only (broader lookback).

        Both tiers draw from the same candidate pool, newest-first.
        Tier 2 picks up where Tier 1 stopped.

        Args:
            continuum_id: Continuum ID
            user_id: User ID

        Returns:
            (tier1, tier2) — both in chronological order (oldest to newest)

        Raises:
            DatabaseError: If database query fails
        """
        complexity_limit = SESSION_SUMMARY_COMPLEXITY_LIMIT
        max_count = SESSION_SUMMARY_MAX_COUNT
        precis_max = SESSION_PRECIS_MAX_COUNT
        query_window = SESSION_SUMMARY_QUERY_WINDOW

        # Query window of recent segments (returns oldest first, we reverse for selection)
        candidates = self.repository.find_collapsed_segments(
            continuum_id,
            user_id,
            limit=query_window
        )

        if not candidates:
            logger.debug(f"No collapsed segments found for continuum {continuum_id}")
            return [], []

        # Tier 1: Extended summaries (newest first, complexity accumulation)
        tier1 = []
        total_complexity = 0
        tier1_stop_index = 0

        newest_first = list(reversed(candidates))
        for i, segment in enumerate(newest_first):
            complexity = segment.metadata.get('complexity_score', 2)

            if total_complexity + complexity > complexity_limit and len(tier1) > 0:
                tier1_stop_index = i
                break

            tier1.append(segment)
            total_complexity += complexity
            tier1_stop_index = i + 1

            if len(tier1) >= max_count:
                break

        # Tier 2: Precis-only (continue from where Tier 1 stopped)
        tier2 = []
        for segment in newest_first[tier1_stop_index:]:
            precis = segment.metadata.get('precis', '')
            if not precis:
                continue
            tier2.append(segment)
            if len(tier2) >= precis_max:
                break

        logger.info(
            f"Selected {len(tier1)} extended (complexity={total_complexity}/{complexity_limit}) "
            f"+ {len(tier2)} precis from {len(candidates)} candidates"
        )

        # Return both in chronological order (oldest first)
        return list(reversed(tier1)), list(reversed(tier2))

    def _load_active_segment_messages(self, continuum_id: str) -> list[Message]:
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
    ) -> list[Message]:
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
