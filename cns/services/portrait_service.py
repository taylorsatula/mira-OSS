"""
Portrait synthesis service.

Generates a prose portrait of the user from collapsed segment summaries and
stores it on the users table in PostgreSQL. The portrait populates the
{user_context} placeholder in the system prompt <user> section.

Triggered by segment collapse: _process_portrait_synthesis() in
segment_collapse_handler.py calls should_synthesize_portrait() to check the
use-day gate, then synthesize_and_store() to generate and persist.
WorkingMemory reads the stored portrait via read_portrait() (no on-demand
synthesis — the collapse chain is the sole trigger).
"""
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from clients.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Minimum collapsed segments required before generating a portrait.
# Below this threshold the LLM has too little signal to produce useful output.
MIN_SUMMARIES = 3

# Fetch all collapsed segments within the user's last N activity days.
# Replaces the old flat LIMIT — gives consistent temporal coverage regardless
# of how many segments a user produces per day.
ACTIVITY_DAY_WINDOW = 20

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"

# Module-level state — loaded once per process
_system_prompt: Optional[str] = None
_user_template: Optional[str] = None
_llm_provider: Optional[LLMProvider] = None

# In-memory dedup: tracks {user_id: activity_days_at_last_synthesis}.
# Prevents redundant LLM calls when multiple segments collapse on the same
# qualifying activity day. Clears on process restart — at most one extra
# synthesis per user, which is harmless for a portrait refresh.
_portrait_last_synthesized: dict[str, int] = {}


def should_synthesize_portrait(user_id: str, threshold: int) -> bool:
    """
    Check whether portrait synthesis should run for this user.

    Uses modular arithmetic on cumulative_activity_days with an in-memory
    dedup guard to prevent duplicate synthesis within a process lifetime.
    """
    from utils.user_context import get_user_cumulative_activity_days

    activity_days = get_user_cumulative_activity_days()
    if activity_days <= 0 or activity_days % threshold != 0:
        return False

    return activity_days > _portrait_last_synthesized.get(user_id, 0)


def _load_prompts() -> None:
    """Load portrait prompt templates (lazy, once per process)."""
    global _system_prompt, _user_template

    if _system_prompt is not None:
        return

    with open(_PROMPTS_DIR / "portrait_synthesis_system.txt") as f:
        system = f.read().strip()
    with open(_PROMPTS_DIR / "portrait_synthesis_user.txt") as f:
        user = f.read().strip()

    _system_prompt = system
    _user_template = user


def _get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider(enable_prompt_caching=False)
    return _llm_provider


def synthesize_and_store(user_id: str) -> bool:
    """
    Generate a portrait from the user's collapsed segment summaries and persist it.

    Returns True if synthesis was attempted (LLM was called), False if skipped
    due to insufficient data. Raises on infrastructure failures so the collapse
    handler can log and continue.

    Args:
        user_id: UUID string for the user (contextvar must already be set by caller)
    """
    _load_prompts()

    summaries = _fetch_summaries(user_id)

    if len(summaries) < MIN_SUMMARIES:
        logger.info(
            "Portrait skipped for user %s: %d summaries available, need %d",
            user_id, len(summaries), MIN_SUMMARIES
        )
        return False

    portrait = _call_llm(summaries)
    _save_portrait(user_id, portrait)

    # Mark dedup after successful save
    from utils.user_context import get_user_cumulative_activity_days
    _portrait_last_synthesized[user_id] = get_user_cumulative_activity_days()

    logger.info(
        "Portrait synthesized for user %s: %d summaries → %d chars",
        user_id, len(summaries), len(portrait)
    )
    return True


def read_portrait(user_id: str) -> str:
    """
    Read the stored portrait for a user from PostgreSQL.

    Returns "" when no portrait exists or if the read fails. This is the correct
    fallback — working memory treats it identically to the pre-portrait state.
    The portrait is optional content, not required infrastructure.
    """
    try:
        from clients.postgres_client import PostgresClient
        from utils.instance import database_name
        db = PostgresClient(database_name())
        row = db.execute_single(
            "SELECT portrait FROM users WHERE id = %s",
            (user_id,)
        )
        return (row.get("portrait") or "") if row else ""
    except Exception as e:
        logger.warning("Portrait read failed for user %s: %s", user_id, e)
        return ""


def _fetch_summaries(user_id: str) -> list[str]:
    """
    Fetch collapsed segment summaries from the last ACTIVITY_DAY_WINDOW activity days.

    Uses a user-scoped session — the caller has already set the user context
    via set_current_user_id(), so RLS enforces isolation on both
    user_activity_days and messages tables.
    """
    from utils.database_session_manager import get_shared_session_manager

    session_manager = get_shared_session_manager()
    with session_manager.get_session(user_id) as session:
        rows = session.execute_query(
            """
            SELECT content FROM messages
            WHERE metadata->>'is_segment_boundary' = 'true'
              AND metadata->>'status' = 'collapsed'
              AND created_at >= COALESCE(
                  (SELECT activity_date FROM user_activity_days
                   ORDER BY activity_date DESC
                   LIMIT 1 OFFSET %(offset)s),
                  '1970-01-01'::date
              )
            ORDER BY created_at DESC
            """,
            {"offset": ACTIVITY_DAY_WINDOW - 1},
        )
    return [row["content"] for row in rows]


def _call_llm(summaries: list[str]) -> str:
    """Generate portrait text from numbered segment summaries."""
    assert _system_prompt is not None
    assert _user_template is not None

    numbered = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
    user_message = _user_template.replace("{segment_summaries}", numbered)

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": user_message}],
        system_override=_system_prompt,
        internal_llm="portrait",
        allow_negative=True,  # Background system task — not user-initiated
    )

    return llm.extract_text_content(response).strip()


def _save_portrait(user_id: str, content: str) -> None:
    """Write portrait to the users table in PostgreSQL."""
    from clients.postgres_client import PostgresClient
    from utils.timezone_utils import utc_now
    from utils.instance import database_name

    db = PostgresClient(database_name())
    db.execute_update(
        "UPDATE users SET portrait = %s, portrait_generated_at = %s WHERE id = %s",
        (content, utc_now(), UUID(user_id)),
    )
