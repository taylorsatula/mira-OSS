"""User information trinket for displaying cached user data in system prompt."""
import logging
from typing import Dict, Any

from .base import EventAwareTrinket
from utils.user_context import get_current_user_id, get_current_user, update_current_user
from utils.database_session_manager import get_shared_session_manager

logger = logging.getLogger(__name__)


class UserInfoTrinket(EventAwareTrinket):
    """
    Displays cached user information from overarching_knowledge field.

    This trinket loads user information once per session and caches it
    both at the application level (user context) and LLM level (prompt caching).
    The overarching_knowledge field contains structured user data like name,
    preferences, and other semi-static information.
    """

    # User info is cacheable (changes infrequently)
    cache_policy = True

    def _get_variable_name(self) -> str:
        """User info publishes to 'user_information'."""
        return "user_information"

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate user information content from database.

        Uses two-level caching:
        1. Session-level: Checks user context cache first (rapidpath)
        2. Database query: Falls back to DB on cache miss

        Args:
            context: Update context (user_id available via contextvars)

        Returns:
            Formatted user information section or empty string if no data

        Raises:
            DatabaseError: If database query fails (infrastructure failure)
        """
        # Get user_id from contextvars (set by auth middleware)
        user_id = get_current_user_id()
        if not user_id:
            logger.warning("UserInfoTrinket called without user_id in context")
            return ""

        # Check if already cached in session context (rapidpath)
        user_data = get_current_user()
        if 'overarching_knowledge' in user_data:
            knowledge = user_data['overarching_knowledge']
            logger.debug(f"UserInfoTrinket using cached knowledge for user {user_id}")
        else:
            # Query database (first time this session)
            logger.debug(f"UserInfoTrinket querying database for user {user_id}")
            session_manager = get_shared_session_manager()

            with session_manager.get_session(user_id) as session:
                result = session.execute_single(
                    "SELECT overarching_knowledge FROM users WHERE id = %(user_id)s",
                    {'user_id': user_id}
                )
                knowledge = result.get('overarching_knowledge', '') if result else ''

            # Cache in session context for subsequent turns
            update_current_user({'overarching_knowledge': knowledge})

        # Return empty if no knowledge stored
        if not knowledge or not knowledge.strip():
            logger.debug(f"No overarching_knowledge for user {user_id}")
            return ""

        # Format with clear header
        return f"USER INFORMATION:\n{knowledge}"
