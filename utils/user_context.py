"""
User context management using contextvars.

This module provides transparent user context that works for both:
- Single-user scenarios (CLI): Context set once and persists
- Multi-user scenarios (web): Context isolated per request automatically

Uses Python's contextvars which provides automatic isolation for
concurrent operations while working identically for single-threaded use.
"""

import contextvars
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

# Context variable for current user data
_user_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'user_context',
    default=None
)

# Context variable for current segment ID (set during conversation processing)
_current_segment_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_segment_id',
    default=None
)

# Context variable for request cancellation (set by WebSocket handler, checked by LLM provider)
_cancel_event: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    'cancel_event',
    default=None
)


def set_cancel_event(event: threading.Event) -> None:
    """Set the cancellation event for the current request."""
    _cancel_event.set(event)


def get_cancel_event() -> Optional[threading.Event]:
    """Get the cancellation event for the current request, or None if not set."""
    return _cancel_event.get(None)


def check_cancelled() -> None:
    """Raise GenerationCancelled if the current request has been cancelled.

    Lightweight check — call between stream chunks, before tool execution,
    and at agentic loop boundaries.
    """
    evt = _cancel_event.get(None)
    if evt is not None and evt.is_set():
        from cns.core.stream_events import GenerationCancelled
        raise GenerationCancelled()


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


def get_current_segment_id() -> Optional[str]:
    """
    Get current segment ID from context.

    Returns None if no segment is active (first message before segment creation).
    """
    return _current_segment_id.get()


def set_current_segment_id(segment_id: str) -> contextvars.Token:
    """
    Set current segment ID in context.

    Called at conversation entry points (websocket/HTTP chat handlers)
    when the active segment is known.

    Returns:
        Token that can be used with reset_current_segment_id()
    """
    return _current_segment_id.set(segment_id)


# ============================================================
# ConversationLLM - Database-backed conversation LLM definitions
# ============================================================

class LLMProvider(str, Enum):
    """LLM provider routing type."""
    ANTHROPIC = "anthropic"  # Direct Anthropic SDK
    GENERIC = "generic"      # OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, etc.)


@dataclass(frozen=True)
class ConversationLLMConfig:
    """LLM configuration for a user-facing conversation model."""
    name: str
    model: str
    thinking_budget: int
    description: str
    display_order: int
    provider: LLMProvider = LLMProvider.ANTHROPIC
    endpoint_url: Optional[str] = None
    api_key_name: Optional[str] = None
    hidden: bool = False


# Module-level cache for conversation LLMs (loaded once per process)
_conversation_llm_cache: Optional[dict[str, ConversationLLMConfig]] = None


def get_conversation_llms() -> dict[str, ConversationLLMConfig]:
    """
    Get all available conversation LLM configs from database.
    Cached at module level (configs rarely change).
    """
    global _conversation_llm_cache
    if _conversation_llm_cache is not None:
        return _conversation_llm_cache

    from clients.postgres_client import PostgresClient
    from utils.instance import database_name
    db = PostgresClient(database_name())

    results = db.execute_query(
        "SELECT name, model, thinking_budget, description, display_order, provider, endpoint_url, api_key_name, hidden FROM conversation_llm ORDER BY display_order"
    )

    _conversation_llm_cache = {
        row['name']: ConversationLLMConfig(
            name=row['name'],
            model=row['model'],
            thinking_budget=row['thinking_budget'],
            description=row['description'] or '',
            display_order=row['display_order'],
            provider=LLMProvider(row['provider']),
            endpoint_url=row['endpoint_url'],
            api_key_name=row['api_key_name'],
            hidden=row.get('hidden', False) or False
        )
        for row in results
    }
    return _conversation_llm_cache


def resolve_conversation_llm(name: str) -> ConversationLLMConfig:
    """Get LLM config for a conversation LLM name."""
    conversation_llms = get_conversation_llms()
    if name not in conversation_llms:
        raise ValueError(f"Unknown conversation LLM: {name}")
    return conversation_llms[name]



# ============================================================
# InternalLLM - Database-backed internal LLM configurations
# ============================================================

@dataclass(frozen=True)
class InternalLLMConfig:
    """Internal LLM configuration for system operations (not user-facing).

    Single source of truth for all LLM-tuning params. Both build_batch_params()
    and generate_response(internal_llm=) read from these fields.
    """
    name: str
    model: str
    endpoint_url: str
    api_key_name: Optional[str]
    description: str
    max_tokens: int
    effort: Optional[str] = None  # 'low'|'medium'|'high'|'max'


_internal_llm_cache: dict[tuple[str, str], InternalLLMConfig] | None = None


def load_internal_llm_configs() -> None:
    """Load internal LLM configs at startup. Call during app boot."""
    global _internal_llm_cache
    from clients.postgres_client import PostgresClient
    from utils.instance import database_name
    db = PostgresClient(database_name())
    results = db.execute_query(
        "SELECT name, tier, model, endpoint_url, api_key_name, description, "
        "max_tokens, effort FROM internal_llm"
    )
    _internal_llm_cache = {
        (row['name'], row['tier']): InternalLLMConfig(
            name=row['name'],
            model=row['model'],
            endpoint_url=row['endpoint_url'],
            api_key_name=row['api_key_name'],
            description=row['description'] or '',
            max_tokens=row['max_tokens'],
            effort=row.get('effort'),
        )
        for row in results
    }


def get_internal_llm(name: str) -> InternalLLMConfig:
    """Get internal LLM config by name, resolved by user's card-on-file status.

    Every config has exactly two rows (free + cof). Resolution is a single
    exact-match lookup — no fallback chain.

    In OSS mode (no billing module), defaults to 'cof' tier.
    """
    if _internal_llm_cache is None:
        raise RuntimeError("Internal LLM configs not loaded. Call load_internal_llm_configs() at startup.")

    tier = _resolve_user_internal_tier()
    key = (name, tier)

    config = _internal_llm_cache.get(key)
    if config:
        return config

    raise KeyError(f"No internal_llm config for '{name}' with tier='{tier}'")


def _resolve_user_internal_tier() -> str:
    """Determine whether the current user gets 'free' or 'cof' internal models.

    Returns 'cof' if user has a payment method on file, 'free' otherwise.
    In OSS mode (no billing module), returns 'cof' (OSS schema seeds
    both tiers with the same models, so the value doesn't matter — but
    'cof' gives the best-available config by convention).
    """
    try:
        from billing import get_billing_backend  # noqa: F401
    except ImportError:
        return 'cof'  # OSS mode — no billing, use cof-tier configs

    if not has_user_context():
        # Expected at startup (factory init, singleton services caching LLM configs).
        # If this appears during request handling, a service is resolving LLM config
        # outside user context — that's a bug.
        logger.warning("get_internal_llm() called without user context (expected at startup only), defaulting to 'cof' tier")
        return 'cof'

    user_id = get_current_user_id()

    from clients.postgres_client import PostgresClient
    from utils.instance import database_name
    db = PostgresClient(database_name(), admin=True)
    result = db.execute_single(
        "SELECT stripe_payment_method_id FROM users WHERE id = %s",
        (user_id,),
    )
    if result and result.get("stripe_payment_method_id"):
        return 'cof'
    return 'free'


# ============================================================
# UserPreferences - Database-backed user settings
# ============================================================

class UserPreferences(BaseModel):
    """
    User preferences and profile data loaded from database.
    Cached in Valkey with invalidation on updates.
    """
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: str = Field(default="America/Chicago")
    temperature_unit: str = Field(default="fahrenheit")
    memory_manipulation_enabled: bool = Field(default=True)
    conversation_llm: str = Field(default="primary")
    created_at: Optional[datetime] = None


def get_user_preferences() -> UserPreferences:
    """
    Get current user's preferences with Valkey caching.

    Cache hierarchy:
    1. Valkey (shared across all contexts - WebSocket, HTTP, etc.)
    2. Database (source of truth)

    Valkey cache is invalidated on preference updates, ensuring
    all contexts see changes immediately.
    """
    import json
    from clients.valkey_client import get_valkey_client
    from clients.postgres_client import PostgresClient

    user_id = get_current_user_id()
    cache_key = f"user_prefs:{user_id}"

    # Check Valkey cache first (shared across all contexts)
    valkey = get_valkey_client()
    cached = valkey.get(cache_key)
    if cached:
        data = json.loads(cached)
        return UserPreferences(**data)

    # Cache miss - fetch from database
    from utils.instance import database_name
    db = PostgresClient(database_name())
    result = db.execute_single(
        """SELECT first_name, last_name, timezone, temperature_unit, memory_manipulation_enabled, conversation_llm, created_at
           FROM users WHERE id = %s""",
        (user_id,)
    )

    prefs = UserPreferences(
        first_name=result.get('first_name'),
        last_name=result.get('last_name'),
        timezone=result.get('timezone') or 'America/Chicago',
        temperature_unit=result.get('temperature_unit') or 'fahrenheit',
        memory_manipulation_enabled=result.get('memory_manipulation_enabled', True),
        conversation_llm=result.get('conversation_llm') or 'minimax',
        created_at=result.get('created_at'),
    )

    # Cache in Valkey with 5-minute TTL (safety net - invalidation handles freshness)
    valkey.set(cache_key, prefs.model_dump_json(), ex=300)

    return prefs


def update_user_preference(field: str, value: Any) -> UserPreferences:
    """
    Update a single preference field in database and invalidate cache.

    Args:
        field: Preference field name (timezone, conversation_llm, etc.)
        value: New value for the field

    Returns:
        Updated UserPreferences object
    """
    if field not in UserPreferences.model_fields:
        raise ValueError(f"Unknown preference field: {field}")

    user_id = get_current_user_id()

    from clients.postgres_client import PostgresClient
    from clients.valkey_client import get_valkey_client

    from utils.instance import database_name
    db = PostgresClient(database_name())

    db.execute_update(
        f"UPDATE users SET {field} = %s WHERE id = %s",
        (value, user_id)
    )

    # Invalidate Valkey cache - next get_user_preferences() will fetch fresh
    valkey = get_valkey_client()
    valkey.delete(f"user_prefs:{user_id}")

    return get_user_preferences()


# ============================================================
# Activity tracking (not a preference - computed value)
# ============================================================

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
