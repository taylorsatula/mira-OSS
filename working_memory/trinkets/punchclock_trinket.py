"""Punchclock trinket for working memory."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.timezone_utils import convert_from_utc, format_datetime, utc_now
from utils.user_context import get_user_timezone

from .base import EventAwareTrinket
from tools.implementations.punchclock_tool import list_punchclock_sessions

logger = logging.getLogger(__name__)


class PunchclockTrinket(EventAwareTrinket):
    """Expose active punchclock sessions inside the system prompt."""

    def _get_variable_name(self) -> str:
        return "punchclock_status"

    def generate_content(self, context: Dict[str, Any]) -> str:
        # Let infrastructure failures propagate
        sessions = list_punchclock_sessions()

        if not sessions:
            return ""  # Legitimately empty - no active sessions

        now = utc_now()

        # User timezone is optional configuration - fallback to UTC if not set
        # RuntimeError indicates user context not configured, not system failure
        try:
            user_tz = get_user_timezone()
        except RuntimeError:
            logger.debug("User timezone not configured, using UTC")
            user_tz = "UTC"

        running, paused, completed = self._partition_sessions(sessions, now)

        if not running and not paused and not completed:
            return ""

        content_lines: List[str] = ["=== PUNCHCLOCK OVERVIEW ==="]

        if running:
            content_lines.append("= RUNNING SESSIONS =")
            for item in running:
                line = self._format_running_entry(item, user_tz)
                content_lines.append(line)
                note_line = self._format_notes(item)
                if note_line:
                    content_lines.append(note_line)

        if paused:
            content_lines.append("= PAUSED SESSIONS =")
            for item in paused:
                line = self._format_paused_entry(item, user_tz)
                content_lines.append(line)
                note_line = self._format_notes(item)
                if note_line:
                    content_lines.append(note_line)

        if completed:
            content_lines.append("= RECENTLY COMPLETED =")
            for item in completed[:3]:
                line = self._format_completed_entry(item, user_tz)
                content_lines.append(line)
                note_line = self._format_notes(item)
                if note_line:
                    content_lines.append(note_line)

        return "\n".join(content_lines)

    # Formatting helpers ---------------------------------------------------------
    def _partition_sessions(
        self, sessions: List[Dict[str, Any]], now: datetime
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        running: List[Dict[str, Any]] = []
        paused: List[Dict[str, Any]] = []
        completed: List[Dict[str, Any]] = []

        for session in sessions:
            status = session.get("status")
            item = self._augment_session(session, now)
            if status == "running":
                running.append(item)
            elif status == "paused":
                paused.append(item)
            elif status == "completed":
                completed.append(item)

        completed.sort(
            key=lambda item: item.get("completed_at") or datetime.min.isoformat(),
            reverse=True,
        )

        return running, paused, completed

    def _augment_session(self, session: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        total_seconds = float(session.get("total_seconds", 0.0))
        augmented = dict(session)

        status = session.get("status")
        active_start = session.get("active_segment_start")
        if status == "running" and active_start:
            start_dt = self._safe_fromiso(active_start)
            if start_dt:
                total_seconds += max(0.0, (now - start_dt).total_seconds())

        augmented["total_seconds"] = total_seconds
        augmented["elapsed_human"] = self._format_duration(total_seconds)
        return augmented

    def _format_running_entry(self, session: Dict[str, Any], tz: str) -> str:
        label = session.get("label", "Unnamed session")
        start_text = self._format_local(session.get("first_started_at"), tz)
        elapsed = session.get("elapsed_human", "0s")
        short_id = self._short_id(session.get("id"))
        if start_text:
            return f"- [{short_id}] {label} — started {start_text} (elapsed {elapsed})"
        return f"- [{short_id}] {label} — elapsed {elapsed}"

    def _format_paused_entry(self, session: Dict[str, Any], tz: str) -> str:
        label = session.get("label", "Unnamed session")
        paused_text = self._format_local(session.get("paused_at"), tz)
        elapsed = session.get("elapsed_human", "0s")
        short_id = self._short_id(session.get("id"))
        if paused_text:
            return f"- [{short_id}] {label} — paused at {paused_text} (total {elapsed})"
        return f"- [{short_id}] {label} — paused (total {elapsed})"

    def _format_completed_entry(self, session: Dict[str, Any], tz: str) -> str:
        label = session.get("label", "Unnamed session")
        start_text = self._format_local(session.get("first_started_at"), tz)
        end_text = self._format_local(session.get("completed_at"), tz)
        elapsed = session.get("elapsed_human", "0s")
        short_id = self._short_id(session.get("id"))

        details = []
        if start_text:
            details.append(start_text)
        if end_text:
            details.append(end_text)

        if details:
            window = " → ".join(details)
            return f"- [{short_id}] {label} — {elapsed} ({window})"
        return f"- [{short_id}] {label} — {elapsed}"

    def _format_notes(self, session: Dict[str, Any]) -> Optional[str]:
        notes = session.get("notes")
        if notes:
            return f"  Notes: {notes.strip()}"
        return None

    @staticmethod
    def _short_id(value: Optional[str]) -> str:
        if not value:
            return "??????"
        return value[:6]

    @staticmethod
    def _safe_fromiso(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _format_local(self, value: Optional[str], tz: str) -> Optional[str]:
        dt = self._safe_fromiso(value)
        if not dt:
            return None
        local_dt = convert_from_utc(dt, tz)
        return format_datetime(local_dt, "date_time_short", tz_name=None, include_timezone=False)

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        seconds = max(0, int(round(total_seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        parts: List[str] = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)
