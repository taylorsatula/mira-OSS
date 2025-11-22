"""
User activity tracking - engagement metrics and login tracking.

Handles user engagement data including:
- Activity day tracking (first message of day detection)
- Cumulative activity days (vacation-proof user engagement metric)
- Login timestamp tracking
- Granular activity logging for analytics

This module owns user engagement concerns separate from authentication.
"""
import logging
import pytz
from typing import Optional

from utils.timezone_utils import utc_now
from utils.user_context import get_current_user, update_current_user, get_user_timezone
from utils.database_session_manager import get_shared_session_manager

logger = logging.getLogger(__name__)


def increment_user_activity_day(user_id: str) -> int:
    """
    Increment user's cumulative activity day count if first message of their day.

    ⚠️  FIRST MESSAGE OF DAY HOOK POINT ⚠️
    This method detects the first message a user sends in their local day.
    Insert beginning-of-day actions here (morning summaries, daily notifications, etc).

    Uses user's local timezone to determine day boundaries, so "first message of day"
    is semantically correct regardless of where the user is or when they message.

    Rapidpath optimization: Context caching ensures only first call per session hits DB.

    Args:
        user_id: User ID to increment

    Returns:
        Updated cumulative_activity_days count

    Raises:
        ValueError: If user not found
    """
    # Rapidpath: Check if already incremented today in this session
    user_data = get_current_user()
    if user_data.get('_activity_day_incremented_today'):
        return user_data.get('cumulative_activity_days', 0)

    # Get user's local date (not server date)
    user_tz = pytz.timezone(get_user_timezone())
    user_local_date = utc_now().astimezone(user_tz).date()

    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        # Check if this is first message of user's day
        current_user = session.execute_single("""
            SELECT cumulative_activity_days, last_activity_date
            FROM users
            WHERE id = %(user_id)s
        """, {'user_id': user_id})

        if not current_user:
            raise ValueError(f"User {user_id} not found for activity day increment")

        current_days = current_user.get('cumulative_activity_days', 0) or 0
        last_date = current_user.get('last_activity_date')

        # Check if already active today (user's local date)
        if last_date and last_date >= user_local_date:
            # Already counted today - rapidpath for subsequent messages
            session.execute_update("""
                INSERT INTO user_activity_days (user_id, activity_date, first_message_at, message_count)
                VALUES (%(user_id)s, %(activity_date)s, %(timestamp)s, 1)
                ON CONFLICT (user_id, activity_date)
                DO UPDATE SET message_count = user_activity_days.message_count + 1
            """, {
                'user_id': user_id,
                'activity_date': user_local_date,
                'timestamp': utc_now()
            })

            # Cache for rapidpath on next call
            update_current_user({
                'cumulative_activity_days': current_days,
                '_activity_day_incremented_today': True
            })

            return current_days

        # ========================================================================
        # 🌅 FIRST MESSAGE OF USER'S DAY - Insert daily actions here
        # ========================================================================

        logger.info(f"First message of day for user {user_id} (local date: {user_local_date})")

        # Update last login timestamp to reflect daily activity
        update_user_login(user_id)

        # TODO: Add beginning-of-day actions here:
        # - Morning summary generation
        # - Daily notification triggers
        # - Streak tracking
        # - Time-of-day context updates

        # ========================================================================

        # New activity day - increment cumulative count
        new_count = current_days + 1

        session.execute_update("""
            UPDATE users
            SET cumulative_activity_days = %(new_count)s,
                last_activity_date = %(activity_date)s
            WHERE id = %(user_id)s
        """, {
            'user_id': user_id,
            'new_count': new_count,
            'activity_date': user_local_date
        })

        # Track in granular table
        session.execute_update("""
            INSERT INTO user_activity_days (user_id, activity_date, first_message_at, message_count)
            VALUES (%(user_id)s, %(activity_date)s, %(timestamp)s, 1)
            ON CONFLICT (user_id, activity_date)
            DO UPDATE SET message_count = user_activity_days.message_count + 1
        """, {
            'user_id': user_id,
            'activity_date': user_local_date,
            'timestamp': utc_now()
        })

        # Cache for rapidpath
        update_current_user({
            'cumulative_activity_days': new_count,
            '_activity_day_incremented_today': True
        })

        logger.debug(f"User {user_id} activity day incremented to {new_count}")
        return new_count


def get_user_cumulative_activity_days(user_id: str) -> int:
    """
    Get user's cumulative activity days count.

    Args:
        user_id: User ID to query

    Returns:
        Cumulative activity days (0 if no activity)

    Raises:
        ValueError: If user not found
    """
    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        result = session.execute_single("""
            SELECT cumulative_activity_days
            FROM users
            WHERE id = %(user_id)s
        """, {'user_id': user_id})

        if not result:
            raise ValueError(f"User {user_id} not found")

        return result.get('cumulative_activity_days', 0) or 0


def update_user_login(user_id: str) -> bool:
    """
    Update user's last login timestamp.

    Args:
        user_id: User ID to update

    Returns:
        True if update successful
    """
    session_manager = get_shared_session_manager()
    with session_manager.get_admin_session() as session:
        rows_updated = session.execute_update("""
            UPDATE users
            SET last_login_at = %(login_time)s
            WHERE id = %(user_id)s
        """, {
            'user_id': user_id,
            'login_time': utc_now()
        })
        return rows_updated > 0
