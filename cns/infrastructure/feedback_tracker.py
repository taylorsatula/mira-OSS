"""
Feedback synthesis tracking for the user model pipeline.

Determines when user model synthesis should run using modular arithmetic
on cumulative_activity_days (every N use-days). Stores a snapshot of the
activity day counter when synthesis last ran — completely stateless,
no counters to reset.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TypedDict

from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now
from utils.user_activity import get_user_cumulative_activity_days


class LoraContent(TypedDict):
    """User model XML and check-in status returned by get_lora_content()."""
    synthesis_xml: str | None
    needs_checkin: bool


class TrackingStatus(TypedDict):
    """Return type for get_tracking_status()."""
    use_days_since_synthesis: int
    last_synthesis_at: datetime | None
    has_previous_synthesis: bool
    needs_checkin: bool


logger = logging.getLogger(__name__)


class FeedbackTracker:
    """
    Tracks synthesis timing for the user model pipeline.

    Uses modular arithmetic on users.cumulative_activity_days to determine
    when synthesis should run. The activity_days_at_last_synthesis column
    stores a snapshot of the counter at the last synthesis run.
    """

    def should_synthesize(self, user_id: str, threshold: int = 7) -> bool:
        """
        Check if user model synthesis should run.

        Uses modular arithmetic: cumulative_activity_days % threshold == 0.
        Dedup guard: only fires if activity_days has advanced past the last snapshot.

        Args:
            user_id: User ID
            threshold: Use-day interval between synthesis runs (default 7)

        Returns:
            True if synthesis should run
        """
        activity_days = get_user_cumulative_activity_days(user_id)

        if activity_days <= 0:
            return False

        if activity_days % threshold != 0:
            return False

        # Dedup: ensure we haven't already synthesized at this activity day
        session_manager = get_shared_session_manager()
        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT activity_days_at_last_synthesis
                FROM feedback_synthesis_tracking
                WHERE user_id = %s
            """, (user_id,))

            if not result:
                return False

            last_snapshot = result.get('activity_days_at_last_synthesis', 0)
            should_run = activity_days > last_snapshot

            if should_run:
                logger.info("User %s: synthesis due (activity_days=%d, last_snapshot=%d, threshold=%d)",
                           user_id, activity_days, last_snapshot, threshold)

            return should_run

    def get_last_synthesis_output(self, user_id: str) -> str | None:
        """
        Get the XML output from the previous synthesis run.

        Args:
            user_id: User ID

        Returns:
            Previous user model XML, or None if no previous synthesis
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT last_synthesis_output
                FROM feedback_synthesis_tracking
                WHERE user_id = %s
            """, (user_id,))

            if result:
                return result.get('last_synthesis_output')
            return None

    def mark_synthesized(
        self,
        user_id: str,
        synthesis_output: str,
        needs_checkin: bool = False
    ) -> None:
        """
        Mark synthesis as complete by snapshotting current activity days.

        Args:
            user_id: User ID
            synthesis_output: User model XML (stored for next run)
            needs_checkin: Whether the model contains check-in topics
        """
        session_manager = get_shared_session_manager()
        now = utc_now()
        activity_days = get_user_cumulative_activity_days(user_id)

        with session_manager.get_session(user_id) as session:
            session.execute_single("""
                UPDATE feedback_synthesis_tracking
                SET activity_days_at_last_synthesis = %s,
                    last_synthesis_at = %s,
                    last_synthesis_output = %s,
                    needs_checkin = %s
                WHERE user_id = %s
            """, (activity_days, now, synthesis_output, needs_checkin, user_id))

            logger.info("User %s: synthesis complete, snapshot activity_days=%d, needs_checkin=%s",
                       user_id, activity_days, needs_checkin)

    def get_tracking_status(self, user_id: str) -> TrackingStatus:
        """
        Get current tracking status for a user.

        Computes use_days_since_synthesis on the fly from the snapshot
        to preserve the API contract for data.py and actions.py consumers.

        Args:
            user_id: User ID

        Returns:
            Dict with use_days_since_synthesis (computed), last_synthesis_at,
            has_previous_synthesis, needs_checkin
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT activity_days_at_last_synthesis,
                       last_synthesis_at, needs_checkin,
                       last_synthesis_output IS NOT NULL as has_previous_synthesis
                FROM feedback_synthesis_tracking
                WHERE user_id = %s
            """, (user_id,))

            if result:
                activity_days = get_user_cumulative_activity_days(user_id)
                last_snapshot = result.get('activity_days_at_last_synthesis', 0)
                return {
                    'use_days_since_synthesis': activity_days - last_snapshot,
                    'last_synthesis_at': result.get('last_synthesis_at'),
                    'has_previous_synthesis': result.get('has_previous_synthesis', False),
                    'needs_checkin': result.get('needs_checkin', False)
                }
            return {
                'use_days_since_synthesis': 0,
                'last_synthesis_at': None,
                'has_previous_synthesis': False,
                'needs_checkin': False
            }

    def get_lora_content(self, user_id: str) -> LoraContent:
        """
        Get user model XML and check-in status in a single query.

        Used by LoraTrinket to reduce database round-trips.

        Args:
            user_id: User ID

        Returns:
            Dict with 'synthesis_xml' and 'needs_checkin' keys
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT last_synthesis_output, needs_checkin
                FROM feedback_synthesis_tracking
                WHERE user_id = %s
            """, (user_id,))

            if result:
                return {
                    'synthesis_xml': result.get('last_synthesis_output'),
                    'needs_checkin': result.get('needs_checkin', False)
                }
            return {'synthesis_xml': None, 'needs_checkin': False}

    def set_synthesis_output(self, user_id: str, xml: str) -> None:
        """
        Update the synthesis output (user model XML) for a user.

        Also updates last_synthesis_at timestamp.

        Args:
            user_id: User ID
            xml: User model XML content
        """
        session_manager = get_shared_session_manager()
        now = utc_now()

        with session_manager.get_session(user_id) as session:
            existing = session.execute_single("""
                SELECT user_id FROM feedback_synthesis_tracking WHERE user_id = %s
            """, (user_id,))

            if existing:
                session.execute_single("""
                    UPDATE feedback_synthesis_tracking
                    SET last_synthesis_output = %s, last_synthesis_at = %s
                    WHERE user_id = %s
                """, (xml, now, user_id))
            else:
                session.execute_single("""
                    INSERT INTO feedback_synthesis_tracking
                        (user_id, activity_days_at_last_synthesis, last_synthesis_output, last_synthesis_at)
                    VALUES (%s, 0, %s, %s)
                """, (user_id, xml, now))

        logger.debug("User %s: updated synthesis_output", user_id)

    def reset_synthesis(self, user_id: str) -> None:
        """
        Clear synthesis output and snapshot current activity days.

        Args:
            user_id: User ID
        """
        session_manager = get_shared_session_manager()
        activity_days = get_user_cumulative_activity_days(user_id)

        with session_manager.get_session(user_id) as session:
            session.execute_single("""
                UPDATE feedback_synthesis_tracking
                SET last_synthesis_output = NULL,
                    activity_days_at_last_synthesis = %s,
                    needs_checkin = FALSE
                WHERE user_id = %s
            """, (activity_days, user_id))

        logger.debug("User %s: reset synthesis output, snapshot activity_days=%d", user_id, activity_days)

    def acknowledge_checkin(self, user_id: str, checkin_response: str) -> None:
        """
        Store check-in feedback and clear the needs_checkin flag.

        Idempotent: overwrites any previous checkin_response (supports iterative
        refinement where the user corrects Mira's paraphrase across turns).

        Args:
            user_id: User ID
            checkin_response: User-confirmed feedback text from the check-in
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            session.execute_single("""
                UPDATE feedback_synthesis_tracking
                SET needs_checkin = FALSE,
                    checkin_response = %s
                WHERE user_id = %s
            """, (checkin_response, user_id))

        logger.info("User %s: check-in acknowledged, needs_checkin=FALSE", user_id)

    def get_and_clear_checkin_response(self, user_id: str) -> str | None:
        """
        Atomically read and clear stored check-in feedback.

        Called during synthesis to incorporate user feedback, then null the
        column so it's not consumed again in a subsequent cycle.

        Args:
            user_id: User ID

        Returns:
            Stored feedback text, or None if no feedback pending
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                UPDATE feedback_synthesis_tracking
                SET checkin_response = NULL
                WHERE user_id = %s AND checkin_response IS NOT NULL
                RETURNING checkin_response
            """, (user_id,))

            if result:
                feedback = result.get('checkin_response')
                if feedback:
                    logger.info("User %s: consumed check-in feedback for synthesis", user_id)
                    return feedback
            return None

    def initialize_user(self, user_id: str) -> None:
        """
        Initialize feedback tracking for a new user.

        Creates the feedback_synthesis_tracking row with defaults.
        Called during account creation.

        Args:
            user_id: User ID
        """
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            session.execute_single("""
                INSERT INTO feedback_synthesis_tracking
                    (user_id, activity_days_at_last_synthesis)
                VALUES (%s, 0)
                ON CONFLICT (user_id) DO NOTHING
            """, (user_id,))

        logger.info("User %s: initialized feedback tracking", user_id)
