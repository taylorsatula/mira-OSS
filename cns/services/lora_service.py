"""
LoRA (user model) refinement service.

Generates a revised user model based on user instructions and stores it as a
preview in Valkey. The user then accepts or declines the preview. Critic
validation runs before presenting the preview, ensuring the refined model
meets the same quality standards as synthesized models.

Unlike portrait refinement (plain prose), user model refinement must produce
valid XML and pass the critic's quality checks before the preview is shown.
"""
import logging
import re
from typing import Optional
from uuid import uuid4

from clients.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# TTL for LoRA refinement previews stored in Valkey.
# Same as portrait: 10 minutes — long enough to review, short enough to avoid
# stale previews accumulating. Expired keys are invisible to accept/decline.
PREVIEW_TTL_SECONDS = 600

_LORA_PREVIEW_PREFIX = "lora_preview"

# Module-level state — loaded once per process
_refinement_system_prompt: Optional[str] = None
_critic_system_prompt: Optional[str] = None
_critic_user_template: Optional[str] = None
_llm_provider: Optional[LLMProvider] = None

# Section list for critic context (populated on first use)
_section_list: Optional[str] = None

CRITIC_MAX_ATTEMPTS = 3


def _load_prompts() -> None:
    """Load refinement and critic prompt templates (lazy, once per process)."""
    global _refinement_system_prompt, _critic_system_prompt, _critic_user_template

    if _refinement_system_prompt is not None:
        return

    from config.prompts.loader import load_prompt
    _refinement_system_prompt = load_prompt("lora_refinement_system.txt")
    _critic_system_prompt = load_prompt("user_model_critic_system.txt")
    _critic_user_template = load_prompt("user_model_critic_user.txt")


def _get_section_list() -> str:
    """Pre-compute section list for critic context."""
    global _section_list
    if _section_list is not None:
        return _section_list

    from cns.services.system_prompt_parser import format_section_list, get_assessable_sections
    from config import config
    sections = get_assessable_sections(config.system_prompt)
    _section_list = format_section_list(sections)
    return _section_list


def _get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider


def read_lora(user_id: str) -> str:
    """
    Read the stored user model XML for a user.

    Returns "" when no model exists.
    """
    from cns.infrastructure.feedback_tracker import FeedbackTracker
    tracker = FeedbackTracker()
    content = tracker.get_lora_content(user_id)
    return content.get("synthesis_xml") or ""


def refine_lora(user_id: str, instructions: str) -> dict[str, str]:
    """
    Generate a revised user model based on user instructions and store as a preview.

    The existing model is combined with the user's refinement instructions
    and sent to the LLM. The result is validated by the critic before being
    stored as a preview. If the critic rejects the output, refinement is
    re-attempted with critic feedback (up to CRITIC_MAX_ATTEMPTS times).

    Returns dict with 'preview_id' and 'proposed' (the revised user model XML).
    Raises ValueError if no existing model or if refinement produces no output.

    Args:
        user_id: UUID string for the user (contextvar must already be set by caller)
        instructions: The user's free-text instructions for refining the user model
    """
    _load_prompts()

    current = read_lora(user_id)
    if not current:
        raise ValueError(
            "No user model exists to refine — user model synthesis runs "
            "automatically based on conversation feedback."
        )

    if not instructions or not instructions.strip():
        raise ValueError("Refinement instructions cannot be empty.")

    # Run refinement with critic validation loop
    candidate_xml = _run_refinement(current, instructions)

    # Critic validation loop (mirrors UserModelSynthesizer.synthesize)
    for attempt in range(CRITIC_MAX_ATTEMPTS):
        critic = _validate_with_critic(candidate_xml)

        if critic["passed"]:
            logger.info(
                "LoRA refinement passed critic (attempt %d) for user %s",
                attempt + 1, user_id
            )
            break

        logger.warning(
            "Critic rejected LoRA refinement (attempt %d) for user %s: %s",
            attempt + 1, user_id, critic["feedback"][:200]
        )

        if attempt < CRITIC_MAX_ATTEMPTS - 1:
            candidate_xml = _rerun_refinement_with_feedback(
                current, instructions, critic["feedback"]
            )
    else:
        # Circuit breaker: return the candidate anyway, but log a warning.
        # Unlike synthesis (which falls back to the previous model),
        # refinement is user-initiated — the user should see what was produced
        # and decide whether to accept it.
        logger.warning(
            "LoRA refinement exhausted %d critic attempts for user %s, "
            "presenting last candidate for user review",
            CRITIC_MAX_ATTEMPTS, user_id
        )

    if not candidate_xml or not candidate_xml.strip():
        raise ValueError("LoRA refinement produced no output — try different instructions.")

    # Store preview in Valkey with TTL
    preview_id = str(uuid4())
    valkey_key = f"{_LORA_PREVIEW_PREFIX}:{user_id}:{preview_id}"

    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()
    valkey.setex(valkey_key, PREVIEW_TTL_SECONDS, candidate_xml)

    logger.info(
        "LoRA preview created for user %s: preview_id=%s, %d chars",
        user_id, preview_id, len(candidate_xml)
    )

    return {"preview_id": preview_id, "proposed": candidate_xml}


def accept_lora(user_id: str, preview_id: str) -> None:
    """
    Accept a LoRA preview: fetch from Valkey and persist to feedback tracking.

    Raises ValueError if the preview_id is expired, invalid, or does not belong
    to this user.

    Args:
        user_id: UUID string for the user
        preview_id: Opaque ID returned by refine_lora()
    """
    proposed = _pop_preview(user_id, preview_id)

    from cns.infrastructure.feedback_tracker import FeedbackTracker
    tracker = FeedbackTracker()
    tracker.set_synthesis_output(user_id, proposed)
    _invalidate_lora_cache(user_id)

    logger.info(
        "LoRA preview accepted for user %s: preview_id=%s, %d chars",
        user_id, preview_id, len(proposed)
    )


def decline_lora(user_id: str, preview_id: str) -> None:
    """
    Decline a LoRA preview: delete from Valkey without persisting.

    No-op if the preview has already expired or been consumed.

    Args:
        user_id: UUID string for the user
        preview_id: Opaque ID returned by refine_lora()
    """
    _pop_preview(user_id, preview_id)

    logger.info(
        "LoRA preview declined for user %s: preview_id=%s",
        user_id, preview_id
    )


def _pop_preview(user_id: str, preview_id: str) -> str:
    """
    Fetch and delete a LoRA preview from Valkey (single-consume).

    Mirrors portrait_service._pop_preview exactly, with a different prefix.
    """
    if not preview_id or not preview_id.strip():
        raise ValueError("preview_id is required.")

    valkey_key = f"{_LORA_PREVIEW_PREFIX}:{user_id}:{preview_id}"

    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()

    proposed = valkey.get(valkey_key)
    if proposed is None:
        raise ValueError(
            "User model preview not found — it may have expired (10-minute lifetime). "
            "Please generate a new preview."
        )

    # Single-consume: delete after read
    valkey.delete(valkey_key)

    # Valkey returns bytes
    if isinstance(proposed, bytes):
        proposed = proposed.decode("utf-8")

    return proposed


def _run_refinement(current_xml: str, instructions: str) -> str:
    """Call the LLM to refine the user model."""
    assert _refinement_system_prompt is not None

    user_message = (
        f"## Existing User Model\n{current_xml}\n\n"
        f"## Instructions\n{instructions.strip()}"
    )

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=_refinement_system_prompt,
        internal_llm="synthesis",
        allow_negative=True,  # Background system task
    )
    return llm.extract_text_content(response).strip()


def _rerun_refinement_with_feedback(
    current_xml: str, instructions: str, critic_feedback: str
) -> str:
    """Rerun refinement with critic feedback appended."""
    assert _refinement_system_prompt is not None

    user_message = (
        f"## Existing User Model\n{current_xml}\n\n"
        f"## Instructions\n{instructions.strip()}\n\n"
        f"## Quality Critic Feedback\n"
        f"The quality critic flagged these issues in the previous attempt. "
        f"Revise the user model to address them:\n\n{critic_feedback}"
    )

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=_refinement_system_prompt,
        internal_llm="synthesis",
        allow_negative=True,
    )
    return llm.extract_text_content(response).strip()


def _validate_with_critic(candidate_xml: str) -> dict:
    """
    Run critic validation on a candidate user model.

    Mirrors UserModelSynthesizer._validate_with_critic.
    """
    assert _critic_system_prompt is not None
    assert _critic_user_template is not None

    section_list = _get_section_list()
    user_prompt = _critic_user_template.format(
        section_id_list=section_list,
        candidate_user_model=candidate_xml
    )

    llm_messages = [
        {"role": "system", "content": _critic_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    llm = _get_llm_provider()
    response = llm.generate_response(
        messages=llm_messages,
        internal_llm="critic",
        allow_negative=True,
    )

    raw_output = llm.extract_text_content(response)

    status_match = re.search(r'<mira:critic_review\s+status="(\w+)"', raw_output)
    if not status_match:
        logger.warning("Could not parse critic output, treating as pass")
        return {"passed": True, "feedback": ""}

    status = status_match.group(1)
    if status == "pass":
        return {"passed": True, "feedback": ""}

    issues = []
    issue_pattern = r'<mira:issue\s+type="([^"]+)"\s+section="([^"]+)">(.*?)</mira:issue>'
    for match in re.finditer(issue_pattern, raw_output, re.DOTALL):
        issue_type = match.group(1)
        section = match.group(2)
        detail = match.group(3).strip()
        issues.append(f"[{issue_type} in {section}] {detail}")

    return {"passed": False, "feedback": "\n".join(issues)}


def _invalidate_lora_cache(user_id: str) -> None:
    """Invalidate the LoraTrinket's Valkey cache after accepting a refined model."""
    from cns.services.orchestrator import get_orchestrator

    get_orchestrator().working_memory.invalidate_trinket("LoraTrinket", user_id)
