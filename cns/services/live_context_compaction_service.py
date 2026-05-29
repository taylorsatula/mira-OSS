"""
Asynchronous live compaction for active-chat provider context.

This service produces one Valkey-only frozen continuation brief for older
messages in the active segment. Segment collapse remains the durable memory
pipeline; this service only shapes future provider requests.
"""
from __future__ import annotations

import html
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clients.llm_provider import LLMProvider
from clients.valkey_client import ValkeyClient, get_valkey_client
from cns.core.continuum import Continuum
from cns.core.message import (
    ContentBlock,
    Message,
    preprocess_content_blocks,
)
from cns.core.message_formatter import format_messages_for_api
from utils.timezone_utils import format_utc_iso, parse_utc_time_string, utc_now
from utils.user_context import get_current_user_id
from config.config_manager import config

if TYPE_CHECKING:
    from cns.infrastructure.continuum_repository import ContinuumRepository

logger = logging.getLogger(__name__)

LIVE_CONTEXT_COMPACTION_KEY_PREFIX = "live_context_compaction"
LIVE_CONTEXT_COMPACTION_LOCK_PREFIX = "live_context_compaction_lock"
LIVE_CONTEXT_COMPACTION_LOCK_TTL_SECONDS = 300


@dataclass(frozen=True)
class LiveContextCompactionArtifact:
    """One frozen continuation brief for an active continuum."""

    continuum_id: str
    covered_start: datetime
    covered_end: datetime
    brief_text: str
    compaction_count: int
    updated_at: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveContextCompactionArtifact":
        """Parse a Valkey JSON payload into a typed artifact."""
        return cls(
            continuum_id=str(data["continuum_id"]),
            covered_start=parse_utc_time_string(str(data["covered_start"])),
            covered_end=parse_utc_time_string(str(data["covered_end"])),
            brief_text=str(data["brief_text"]),
            compaction_count=int(data["compaction_count"]),
            updated_at=parse_utc_time_string(str(data["updated_at"])),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the artifact to the canonical Valkey payload shape."""
        return {
            "continuum_id": self.continuum_id,
            "covered_start": format_utc_iso(self.covered_start),
            "covered_end": format_utc_iso(self.covered_end),
            "brief_text": self.brief_text,
            "compaction_count": self.compaction_count,
            "updated_at": format_utc_iso(self.updated_at),
        }


class LiveContextCompactionStore:
    """Read, replace, and clear the per-user live compaction artifact."""

    def __init__(self, valkey: ValkeyClient | None = None) -> None:
        self.valkey = valkey or get_valkey_client()

    def get(self, user_id: str) -> LiveContextCompactionArtifact | None:
        """Return the current artifact for user_id, or None when absent."""
        raw = self.valkey.get(self._artifact_key(user_id))
        if raw is None:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Live context compaction artifact must be a JSON object")
        return LiveContextCompactionArtifact.from_dict(data)

    def store_replacement(
        self,
        *,
        user_id: str,
        continuum_id: str,
        covered_start: datetime,
        covered_end: datetime,
        brief_text: str,
        prior_artifact: LiveContextCompactionArtifact | None,
    ) -> LiveContextCompactionArtifact:
        """
        Replace the user's artifact with a single rolling continuation brief.

        First compaction starts at count 1. Repeated compactions keep the
        original covered_start and increment compaction_count.
        """
        artifact = LiveContextCompactionArtifact(
            continuum_id=continuum_id,
            covered_start=(
                prior_artifact.covered_start
                if prior_artifact is not None
                else covered_start
            ),
            covered_end=covered_end,
            brief_text=brief_text,
            compaction_count=(
                prior_artifact.compaction_count + 1
                if prior_artifact is not None
                else 1
            ),
            updated_at=utc_now(),
        )
        self.valkey.set(self._artifact_key(user_id), json.dumps(artifact.to_dict()))
        return artifact

    def clear(self, user_id: str) -> bool:
        """Delete the user's live compaction artifact."""
        return self.valkey.delete(self._artifact_key(user_id))

    def acquire_lock(
        self,
        user_id: str,
        ttl_seconds: int = LIVE_CONTEXT_COMPACTION_LOCK_TTL_SECONDS,
    ) -> bool:
        """Acquire the non-blocking per-user compaction lock."""
        return bool(
            self.valkey.set(
                self._lock_key(user_id),
                "1",
                nx=True,
                ex=ttl_seconds,
            )
        )

    def release_lock(self, user_id: str) -> bool:
        """Release the per-user compaction lock."""
        return self.valkey.delete(self._lock_key(user_id))

    @staticmethod
    def _artifact_key(user_id: str) -> str:
        return f"{LIVE_CONTEXT_COMPACTION_KEY_PREFIX}:{user_id}"

    @staticmethod
    def _lock_key(user_id: str) -> str:
        return f"{LIVE_CONTEXT_COMPACTION_LOCK_PREFIX}:{user_id}"



def _wrap_brief(raw_brief: str, compaction_range: LiveCompactionRange) -> str:
    """Wrap the LLM-generated brief in the standard compaction tag with header and recovery instruction."""
    covered_start = format_utc_iso(compaction_range.artifact_covered_start)
    covered_end = format_utc_iso(compaction_range.covered_end)
    lines = [
        f'<mira:compacted_active_context covered_start="{covered_start}" covered_end="{covered_end}">',
        (
            f"EARLIER ACTIVE CONTEXT FROM {covered_start} TO {covered_end} WAS COMPACTED "
            "BECAUSE THE CONTEXT WINDOW LIMIT WAS BEING APPROACHED."
        ),
        (
            f"This block is historical and frozen as of {covered_end}. "
            "Later visible messages are more recent and take precedence if they conflict with this block."
        ),
        "",
        "If exact wording, old tool output, or prior message order matters, use continuum_tool with:",
        "operation=search",
        "search_mode=messages",
        f"start_time={covered_start}",
        f"end_time={covered_end}",
        "",
        raw_brief,
        "</mira:compacted_active_context>",
    ]
    return "\n".join(lines)

TOOL_RESULT_TRUNCATION_LIMIT = 500


def filter_messages_for_live_context(
    messages: Sequence[Message],
    artifact: LiveContextCompactionArtifact | None,
    continuum_id: str,
) -> list[Message]:
    """
    Return a request-local message list with compacted ordinary messages omitted.

    Messages are omitted only when an artifact for this continuum covers their
    timestamp. Segment boundaries, system notifications, and compaction synopsis
    scaffolding are always preserved.
    """
    if artifact is None or artifact.continuum_id != continuum_id:
        return list(messages)

    return [
        message
        for message in messages
        if not _is_covered_ordinary_message(message, artifact)
    ]


def _is_covered_ordinary_message(
    message: Message,
    artifact: LiveContextCompactionArtifact,
) -> bool:
    if message.metadata.get("is_segment_boundary"):
        return False
    if message.metadata.get("system_notification"):
        return False
    if message.metadata.get("is_compaction_synopsis"):
        return False
    return artifact.covered_start < message.created_at <= artifact.covered_end


@dataclass(frozen=True)
class LiveCompactionRange:
    """Selected DB-backed range for one live compaction run."""

    new_range_start: datetime
    artifact_covered_start: datetime
    covered_end: datetime
    messages: list[Message]
    prior_artifact: LiveContextCompactionArtifact | None


def format_messages_for_live_compaction(messages: Sequence[Message]) -> str:
    """
    Format messages with plain XML role tags for the live compactor LLM.

    The output intentionally contains no message IDs, timestamps, tool_call_id
    attributes, or error-status attributes.
    """
    parts: list[str] = []

    for message in messages:
        if message.metadata.get("system_notification"):
            continue
        if message.metadata.get("is_segment_boundary"):
            continue
        if message.metadata.get("is_compaction_synopsis"):
            continue

        if message.role == "tool":
            content = _tool_result_text_for_compaction(message.content)
        else:
            content = _ordinary_message_text_for_compaction(message.content)

        if not content.strip():
            continue

        tag = message.role
        parts.append(f"<{tag}>{html.escape(content, quote=False)}</{tag}>")

    return "\n".join(parts)


def _ordinary_message_text_for_compaction(content: str | list[ContentBlock]) -> str:
    preprocessed = preprocess_content_blocks(content)
    text_parts = list(preprocessed.text_parts)
    if preprocessed.image_count:
        text_parts.insert(0, f"[{preprocessed.image_count} image(s) shared]")
    return " ".join(part for part in text_parts if part)


def _tool_result_text_for_compaction(content: str | list[ContentBlock]) -> str:
    if isinstance(content, str):
        text = content
    else:
        preprocessed = preprocess_content_blocks(content)
        text_parts = list(preprocessed.text_parts)
        if preprocessed.image_count:
            text_parts.insert(0, f"[{preprocessed.image_count} image(s) shared]")
        text = " ".join(part for part in text_parts if part)

    if len(text) <= TOOL_RESULT_TRUNCATION_LIMIT:
        return text
    return (
        text[:TOOL_RESULT_TRUNCATION_LIMIT]
        + f"\n[Tool result truncated to {TOOL_RESULT_TRUNCATION_LIMIT} chars for live compaction]"
    )


class LiveContextCompactionService:
    """Selects active-chat ranges and refreshes the rolling Valkey brief."""

    def __init__(
        self,
        *,
        continuum_repo: "ContinuumRepository",
        store: LiveContextCompactionStore,
        llm_provider: LLMProvider,
    ) -> None:
        self.continuum_repo = continuum_repo
        self.store = store
        self.llm_provider = llm_provider
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="live-context-compaction",
        )
        self._load_prompts()

    def _load_prompts(self) -> None:
        prompts_dir = Path("config/prompts")
        system_path = prompts_dir / "live_context_compaction_system.txt"
        user_path = prompts_dir / "live_context_compaction_user.txt"
        if not system_path.exists() or not user_path.exists():
            raise FileNotFoundError(f"Live context compaction prompts not found in {prompts_dir}")
        self._system_template = system_path.read_text().strip()
        self._user_template = user_path.read_text().strip()

    def should_schedule(self, estimated_tokens: int, available_for_input: int) -> bool:
        """Return True when the request is near enough to the provider limit."""
        return (
            available_for_input - estimated_tokens
            <= config.api.compaction_trigger_buffer_tokens
        )

    def schedule(self, continuum_id: str) -> bool:
        """
        Schedule one non-blocking compaction attempt for the active user.

        Returns True when a worker was submitted. Lock contention or optional
        scheduling failure returns False and leaves the active turn alone.
        """
        user_id = get_current_user_id()
        try:
            if not self.store.acquire_lock(user_id):
                logger.debug("Live context compaction already running for user %s", user_id)
                return False
        except Exception:
            logger.error("Failed to acquire live context compaction lock", exc_info=True)
            return False

        ctx = copy_context()
        try:
            self._executor.submit(
                ctx.run,
                self._run_compaction_with_lock,
                continuum_id,
                user_id,
            )
            return True
        except Exception:
            self.store.release_lock(user_id)
            logger.error("Failed to schedule live context compaction", exc_info=True)
            return False

    def _run_compaction_with_lock(self, continuum_id: str, user_id: str) -> None:
        try:
            self.compact_once(continuum_id)
        except Exception:
            logger.error("Live context compaction failed", exc_info=True)
        finally:
            try:
                self.store.release_lock(user_id)
            except Exception:
                logger.error("Failed to release live context compaction lock", exc_info=True)

    def compact_once(self, continuum_id: str) -> LiveContextCompactionArtifact | None:
        """Run one synchronous compaction attempt for the current user."""
        compaction_range = self.select_range(continuum_id)
        if compaction_range is None:
            return None

        new_history = format_messages_for_live_compaction(compaction_range.messages)
        if not new_history.strip():
            logger.debug("Live compaction selected an empty history range")
            return None

        messages = self._build_compactor_messages(compaction_range, new_history)
        response = self.llm_provider.generate_response(
            messages=messages,
            internal_llm="summary",
            allow_negative=True,
        )
        raw_brief = self.llm_provider.extract_text_content(response).strip()
        if not raw_brief:
            raise ValueError("Live context compactor returned an empty brief")
        brief_text = _wrap_brief(raw_brief, compaction_range)

        user_id = get_current_user_id()
        artifact = self.store.store_replacement(
            user_id=user_id,
            continuum_id=continuum_id,
            covered_start=compaction_range.artifact_covered_start,
            covered_end=compaction_range.covered_end,
            brief_text=brief_text,
            prior_artifact=compaction_range.prior_artifact,
        )
        logger.info(
            "Stored live context compaction artifact for continuum %s through %s (count=%d)",
            continuum_id,
            format_utc_iso(artifact.covered_end),
            artifact.compaction_count,
        )
        return artifact

    def select_range(self, continuum_id: str) -> LiveCompactionRange | None:
        """Select the next DB-backed range while preserving recent raw turns."""
        user_id = get_current_user_id()
        active_sentinel = self.continuum_repo.find_active_segment(continuum_id, user_id)
        if active_sentinel is None:
            logger.debug("No active segment found for live context compaction")
            return None

        prior_artifact = self.store.get(user_id)
        if prior_artifact is not None and prior_artifact.continuum_id != continuum_id:
            self.store.clear(user_id)
            prior_artifact = None

        new_range_start = (
            prior_artifact.covered_end
            if prior_artifact is not None
            else active_sentinel.created_at
        )
        artifact_covered_start = (
            prior_artifact.covered_start
            if prior_artifact is not None
            else new_range_start
        )

        candidates = self.continuum_repo.load_messages_for_live_compaction_range(
            continuum_id=continuum_id,
            user_id=user_id,
            covered_start=new_range_start,
            covered_end=utc_now(),
        )
        user_messages = [message for message in candidates if message.role == "user"]
        if len(user_messages) <= config.api.compaction_raw_user_turns_to_preserve:
            return None

        preserve_from = user_messages[-config.api.compaction_raw_user_turns_to_preserve].created_at
        compactable_messages = [
            message for message in candidates
            if message.created_at < preserve_from
        ]
        if not compactable_messages:
            return None

        covered_end = compactable_messages[-1].created_at
        messages = self.continuum_repo.load_messages_for_live_compaction_range(
            continuum_id=continuum_id,
            user_id=user_id,
            covered_start=new_range_start,
            covered_end=covered_end,
        )
        if not messages:
            return None

        return LiveCompactionRange(
            new_range_start=new_range_start,
            artifact_covered_start=artifact_covered_start,
            covered_end=covered_end,
            messages=messages,
            prior_artifact=prior_artifact,
        )

    def format_messages_for_api(self, continuum: Continuum) -> list[dict[str, object]]:
        """Return provider-formatted messages after live compaction filtering."""
        visible_messages = self.get_visible_messages(continuum)
        return format_messages_for_api(visible_messages)

    def get_visible_messages(self, continuum: Continuum) -> list[Message]:
        """Return request-visible messages without mutating the continuum."""
        user_id = get_current_user_id()
        try:
            artifact = self.store.get(user_id)
        except Exception:
            logger.error(
                "Live context compaction artifact unavailable; returning full continuum for this request",
                exc_info=True,
            )
            return list(continuum.messages)

        return filter_messages_for_live_context(
            continuum.messages,
            artifact,
            str(continuum.id),
        )

    def _build_compactor_messages(
        self,
        compaction_range: LiveCompactionRange,
        new_history: str,
    ) -> list[dict[str, object]]:
        covered_end = format_utc_iso(compaction_range.covered_end)
        system_prompt = self._system_template.replace("{covered_end}", covered_end)
        user_prompt = self._user_template.replace("{new_history}", new_history)

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
        ]
        if compaction_range.prior_artifact is not None:
            messages.append({
                "role": "assistant",
                "content": compaction_range.prior_artifact.brief_text,
            })
        messages.append({"role": "user", "content": user_prompt})
        return messages
