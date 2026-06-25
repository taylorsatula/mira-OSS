"""
Auto-generate one-sentence summaries for DomainDoc sections.

When a DomainDoc section is created or updated (via tool or web UI), this service
generates a brief summary and stores it directly on the section record.

The Section Index is rendered dynamically by the domaindoc trinket from these
stored summaries - no separate index storage needed.
"""
import logging
from typing import Optional

from clients.llm_provider import LLMProvider
from utils.userdata_manager import UserDataManager

logger = logging.getLogger(__name__)

# Module-level state for lazy-loaded prompts
_system_prompt: Optional[str] = None
_user_template: Optional[str] = None
_llm_provider: Optional[LLMProvider] = None


def _load_prompts() -> None:
    """Load prompt templates from config files (lazy, once per process)."""
    global _system_prompt, _user_template

    if _system_prompt is not None:
        return  # Already loaded

    from config.prompts.loader import load_prompt
    _system_prompt = load_prompt("domaindoc_summary_system.txt")
    _user_template = load_prompt("domaindoc_summary_user.txt")


def _get_llm_provider() -> LLMProvider:
    """Get or create the LLM provider singleton."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider


def update_section_summary(
    db: UserDataManager,
    section_id: int,
    section_header: str,
    content: str
) -> None:
    """
    Generate summary for a section and store it on the section record.

    Called after section content is created or modified in both DomaindocTool
    and actions.py.

    Args:
        db: User's database manager
        section_id: ID of the section to update
        section_header: Header of the section (used in summary prompt)
        content: Content to summarize
    """
    # Skip empty content
    if not content or not content.strip():
        return

    try:
        summary = _generate_summary(section_header, content)
        _store_summary(db, section_id, summary)
    except Exception as e:
        # Log and continue - don't block the save operation
        logger.warning(f"Failed to generate summary for section '{section_header}': {e}")


def _generate_summary(header: str, content: str) -> str:
    """
    Generate one-sentence summary via fast LLM.

    Uses Gemini Flash via OpenRouter for speed and cost efficiency.
    Truncates content to prevent excessive token usage.
    """
    _load_prompts()
    assert _system_prompt is not None
    assert _user_template is not None

    # Truncate content to ~2000 chars to keep token count reasonable
    truncated_content = content[:2000] + "..." if len(content) > 2000 else content

    messages = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": _user_template.format(
            section_header=header,
            content=truncated_content
        )}
    ]

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=messages,
        internal_llm='analysis',
        allow_negative=True  # System task — background summary generation
    )

    return llm.extract_text_content(response).strip()


def _store_summary(db: UserDataManager, section_id: int, summary: str) -> None:
    """Store summary directly on the section record."""
    from utils.timezone_utils import utc_now, format_utc_iso

    now = format_utc_iso(utc_now())
    db.update(
        "domaindoc_sections",
        {"encrypted__summary": summary, "updated_at": now},
        "id = :id",
        {"id": section_id}
    )


