"""
User context management using contextvars.

This module provides transparent user context that works for both:
- Single-user scenarios (CLI): Context set once and persists
- Multi-user scenarios (web): Context isolated per request automatically

Uses Python's contextvars which provides automatic isolation for
concurrent operations while working identically for single-threaded use.
"""

import contextvars
from typing import Dict, Any, Optional

# Context variable for current user data
_user_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'user_context', 
    default=None
)


def set_current_user_id(user_id: str) -> None:
    """
    Set current user ID in context (standardized key: 'user_id').
    """
    current = _user_context.get() or {}
    current["user_id"] = user_id
    _user_context.set(current)


def get_current_user_id() -> str:
    """
    Get current user ID from context (reads 'user_id').
    """
    context = _user_context.get()
    if not context or "user_id" not in context:
        raise RuntimeError("No user context set. Ensure authentication is properly initialized.")
    return context["user_id"]


def set_current_user_data(user_data: Dict[str, Any]) -> None:
    """
    Set complete user data in context.
    Standardizes to 'user_id' and does not maintain legacy 'id'.
    """
    data = user_data.copy()
    if "user_id" not in data and "id" in data:
        # Normalize legacy key to standardized key
        data["user_id"] = data.pop("id")
    current = _user_context.get() or {}
    current.update(data)
    _user_context.set(current)


def get_current_user() -> Dict[str, Any]:
    """
    Get current user data from context.
    
    Returns:
        Copy of current user data dictionary
        
    Raises:
        RuntimeError: If no user context is set
    """
    context = _user_context.get()
    if not context:
        raise RuntimeError("No user context set. Ensure authentication is properly initialized.")
    return context.copy()


def update_current_user(updates: Dict[str, Any]) -> None:
    """
    Update current user data with new values.
    
    Args:
        updates: Dictionary of updates to apply
    """
    current = _user_context.get() or {}
    current.update(updates)
    _user_context.set(current)


def clear_user_context() -> None:
    """
    Clear the current user context.
    
    Useful for cleanup or testing scenarios.
    """
    _user_context.set(None)


def has_user_context() -> bool:
    """
    Check if user context is currently set.

    Returns:
        True if user context exists, False otherwise
    """
    context = _user_context.get()
    return context is not None and "user_id" in context


def get_user_timezone() -> str:
    """
    Get current user's timezone with fallback to system default.

    Context caching ensures we only query the database once per session,
    with subsequent calls returning the cached value.

    Returns:
        IANA timezone name (user preference if set, otherwise system default)

    Raises:
        RuntimeError: If no user context is set
        Exception: If database query fails (propagates database errors)
    """
    # Rapidpath: Check if already cached in context
    try:
        user_data = get_current_user()
        if 'timezone' in user_data:
            return user_data['timezone']
    except RuntimeError:
        raise RuntimeError("No user context set. Cannot get timezone without user context.")

    # Not cached - query database
    user_id = get_current_user_id()

    from clients.postgres_client import PostgresClient
    db = PostgresClient('mira_service')

    # Let database errors propagate - caller can decide fallback strategy
    result = db.execute_single(
        "SELECT timezone FROM users WHERE id = %s",
        (user_id,)
    )

    # Determine timezone (preference or system default)
    if result and result['timezone']:
        timezone = result['timezone']
    else:
        from config import config
        timezone = config.system.timezone

    # Cache for subsequent calls
    update_current_user({'timezone': timezone})

    return timezone


def set_user_timezone(timezone: str) -> None:
    """
    Update user's timezone preference in database.

    Args:
        timezone: IANA timezone name to set as user preference
    """
    user_id = get_current_user_id()

    from clients.postgres_client import PostgresClient
    db = PostgresClient('mira_service')

    db.execute_update(
        "UPDATE users SET timezone = %s WHERE id = %s",
        (timezone, user_id)
    )

    # Invalidate cache so next get_user_timezone() reflects the update
    update_current_user({'timezone': timezone})


def get_user_cumulative_activity_days() -> int:
    """
    Get current user's cumulative activity days with context caching.

    This is the canonical way to get "how many days" for scoring calculations.
    Returns activity days (not calendar days) to ensure vacation-proof decay.

    Context caching ensures we only query the database once per session,
    with subsequent calls returning the cached value.

    Returns:
        Cumulative activity days for current user

    Raises:
        RuntimeError: If no user context is set
    """
    # Check if already cached in context
    try:
        user_data = get_current_user()
        if 'cumulative_activity_days' in user_data:
            return user_data['cumulative_activity_days']
    except RuntimeError:
        raise RuntimeError("No user context set. Cannot get activity days without user context.")

    # Not cached - query user activity module and cache result
    user_id = get_current_user_id()

    from utils.user_activity import get_user_cumulative_activity_days as get_activity_days
    activity_days = get_activity_days(user_id)

    # Cache for subsequent calls
    update_current_user({'cumulative_activity_days': activity_days})

    return activity_days
