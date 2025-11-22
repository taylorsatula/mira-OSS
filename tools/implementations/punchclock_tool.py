"""Punchclock tool for tracking focused work sessions."""
import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tools.registry import registry
from tools.repo import Tool
from utils.timezone_utils import convert_to_utc, ensure_utc, parse_time_string, utc_now
from utils.user_context import get_current_user_id, get_user_timezone
from utils.userdata_manager import get_user_data_manager

logger = logging.getLogger(__name__)


class PunchclockToolConfig(BaseModel):
    """Configuration for the punchclock tool."""

    enabled: bool = Field(default=True, description="Whether the punchclock tool is enabled")


REGISTRY_NAME = "punchclock_tool"
TABLE_NAME = "punchclock_sessions"

PUNCHCLOCK_SCHEMA = """
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    notes TEXT,
    status TEXT NOT NULL,
    total_seconds REAL NOT NULL DEFAULT 0,
    active_segment_start TEXT,
    paused_at TEXT,
    first_started_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    segments TEXT NOT NULL,
    history TEXT NOT NULL
"""


registry.register(REGISTRY_NAME, PunchclockToolConfig)


def _get_manager():
    return get_user_data_manager(get_current_user_id())


def ensure_punchclock_schema():
    manager = _get_manager()
    manager.create_table(TABLE_NAME, PUNCHCLOCK_SCHEMA)
    return manager


def _encode_session_for_storage(session: Dict[str, Any], include_id: bool = True) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "label": session.get("label"),
        "notes": session.get("notes"),
        "status": session.get("status"),
        "total_seconds": float(session.get("total_seconds", 0.0)),
        "active_segment_start": session.get("active_segment_start"),
        "paused_at": session.get("paused_at"),
        "first_started_at": session.get("first_started_at"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "completed_at": session.get("completed_at"),
        "segments": json.dumps(session.get("segments", []), ensure_ascii=False),
        "history": json.dumps(session.get("history", []), ensure_ascii=False),
    }
    if include_id:
        payload["id"] = session["id"]
    return payload


def _decode_session_row(row: Dict[str, Any]) -> Dict[str, Any]:
    decoded = dict(row)
    decoded["total_seconds"] = float(row.get("total_seconds") or 0.0)

    segments_raw = row.get("segments") or "[]"
    history_raw = row.get("history") or "[]"
    try:
        decoded["segments"] = json.loads(segments_raw)
    except json.JSONDecodeError:
        decoded["segments"] = []
    try:
        decoded["history"] = json.loads(history_raw)
    except json.JSONDecodeError:
        decoded["history"] = []

    return decoded


def list_punchclock_sessions() -> List[Dict[str, Any]]:
    manager = ensure_punchclock_schema()
    rows = manager.select(TABLE_NAME)
    return [_decode_session_row(row) for row in rows]


def get_punchclock_session(session_id: str) -> Optional[Dict[str, Any]]:
    manager = ensure_punchclock_schema()
    rows = manager.select(TABLE_NAME, "id = :session_id", {"session_id": session_id})
    if not rows:
        return None
    return _decode_session_row(rows[0])


def insert_punchclock_session(session: Dict[str, Any]) -> None:
    manager = ensure_punchclock_schema()
    manager.insert(TABLE_NAME, _encode_session_for_storage(session, include_id=True))


def update_punchclock_session(session: Dict[str, Any]) -> None:
    manager = ensure_punchclock_schema()
    payload = _encode_session_for_storage(session, include_id=False)
    manager.update(TABLE_NAME, payload, "id = :session_id", {"session_id": session["id"]})


def clear_punchclock_sessions() -> None:
    manager = ensure_punchclock_schema()
    manager.delete(TABLE_NAME, "1=1")


class PunchclockTool(Tool):
    """Track punchclock-style work sessions with pause/resume support."""

    name = REGISTRY_NAME

    simple_description = "Log work time with labeled sessions. Punch in to start, pause/resume as needed, punch out when done. Supports time offsets like '-10m'. Check status to see active and completed sessions."

    anthropic_schema = {
        "name": name,
        "description": simple_description,
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "punch_in",
                            "pause",
                            "resume",
                            "punch_out",
                            "status",
                        ],
                        "description": "Punchclock operation to execute",
                    },
                    "label": {
                        "type": "string",
                        "description": "Session label for punch_in",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional notes for the session",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time override (ISO timestamp or offset like '-5m')",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Target session identifier",
                    },
                    "time": {
                        "type": "string",
                        "description": "Time override for pause, resume, or punch_out",
                    },
                    "include_completed": {
                        "type": "boolean",
                        "description": "Whether status should include completed sessions",
                    },
                    "max_completed": {
                        "type": "integer",
                        "description": "Maximum completed sessions to return (status op)",
                        "minimum": 1,
                    },
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        }

    tool_hints = (
        "Use punchclock_tool to keep lightweight time logs: punch_in with a label, pause/resume as needed,"
        " and punch_out when finished. For offsets pass '-10m', '+30m', or natural times like '2025-01-02T09:00'."
    )

    def __init__(self, working_memory: Optional["WorkingMemory"] = None):
        super().__init__()
        self._working_memory = working_memory

    # Public API -----------------------------------------------------------------
    def run(self, operation: str, **params: Any) -> Dict[str, Any]:
        """Execute punchclock operations."""

        if not operation:
            raise ValueError("operation is required")

        operation = operation.strip().lower()
        payload = self._normalize_params(params)

        if operation == "punch_in":
            result = self._handle_punch_in(payload)
            self._publish_trinket_refresh()
            return {"operation": "punch_in", **result}
        if operation == "pause":
            result = self._handle_pause(payload)
            self._publish_trinket_refresh()
            return {"operation": "pause", **result}
        if operation == "resume":
            result = self._handle_resume(payload)
            self._publish_trinket_refresh()
            return {"operation": "resume", **result}
        if operation == "punch_out":
            result = self._handle_punch_out(payload)
            self._publish_trinket_refresh()
            return {"operation": "punch_out", **result}
        if operation == "status":
            return {"operation": "status", **self._handle_status(payload)}

        raise ValueError(f"Unsupported operation '{operation}'")

    # Parameter helpers ----------------------------------------------------------
    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        if "kwargs" in params:
            raw_kwargs = params.pop("kwargs")
            if isinstance(raw_kwargs, str):
                if raw_kwargs.strip():
                    try:
                        decoded = json.loads(raw_kwargs)
                    except json.JSONDecodeError as exc:
                        raise ValueError("kwargs must be valid JSON") from exc
                else:
                    decoded = {}
            elif isinstance(raw_kwargs, dict):
                decoded = raw_kwargs
            else:
                raise ValueError("kwargs must be a JSON object or mapping")
            payload.update(decoded)

        payload.update(params)

        # Accept `time` instead of start_time for punch_in inputs
        if "time" in payload and "start_time" not in payload:
            payload["start_time"] = payload["time"]

        return payload

    # Time helpers ---------------------------------------------------------------
    def _parse_time_input(self, time_input: Optional[str]) -> datetime:
        if time_input is None or (isinstance(time_input, str) and not time_input.strip()):
            return utc_now()

        value = time_input.strip().lower()
        if value in {"now", "current", "immediately"}:
            return utc_now()

        offset_match = self._match_offset(value)
        if offset_match:
            sign, amount_str, unit = offset_match
            amount = int(amount_str)
            delta = self._offset_delta(amount, unit)
            if sign == "+":
                return utc_now() + delta
            if sign == "-" or value.endswith("ago"):
                return utc_now() - delta
            # Default to future if sign omitted but phrase says "ago"
            return utc_now() + delta

        try:
            tz_name = get_user_timezone()
        except RuntimeError:
            tz_name = "UTC"

        try:
            parsed = parse_time_string(time_input, tz_name=tz_name)
        except ValueError as exc:
            raise ValueError(
                "Could not parse time input. Use ISO strings or offsets like '-15m'."
            ) from exc

        return ensure_utc(convert_to_utc(parsed))

    _OFFSET_PATTERN = re.compile(r"^(?P<sign>[+-]?)(?P<amount>\d+)(?P<unit>[smhd])(?:\s*ago)?$")

    def _match_offset(self, value: str) -> Optional[List[str]]:
        match = self._OFFSET_PATTERN.match(value.replace(" ", ""))
        if not match:
            return None
        sign = match.group("sign") or ("-" if value.endswith("ago") else "+")
        return [sign, match.group("amount"), match.group("unit")]

    @staticmethod
    def _offset_delta(amount: int, unit: str) -> timedelta:
        mapping = {
            "s": timedelta(seconds=amount),
            "m": timedelta(minutes=amount),
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
        }
        if unit not in mapping:
            raise ValueError(f"Unsupported offset unit '{unit}'")
        return mapping[unit]

    @staticmethod
    def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        return ensure_utc(dt)

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

    # State helpers --------------------------------------------------------------
    def _get_session(self, session_id: str) -> Dict[str, Any]:
        session = get_punchclock_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        return session

    def _serialize_session(self, session: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or utc_now()

        total_seconds = float(session.get("total_seconds", 0))
        status = session.get("status", "unknown")

        if status == "running" and session.get("active_segment_start"):
            start_dt = self._iso_to_dt(session["active_segment_start"])
            if start_dt:
                total_seconds += max(0.0, (now - start_dt).total_seconds())

        serialized = {
            "id": session.get("id"),
            "label": session.get("label"),
            "notes": session.get("notes"),
            "status": status,
            "total_seconds": total_seconds,
            "elapsed": self._format_duration(total_seconds),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "first_started_at": session.get("first_started_at"),
            "completed_at": session.get("completed_at"),
        }

        if session.get("active_segment_start"):
            serialized["active_segment_start"] = session["active_segment_start"]
        if session.get("paused_at"):
            serialized["paused_at"] = session["paused_at"]
        if session.get("segments"):
            serialized["segments"] = session["segments"]

        return serialized

    # Handlers ------------------------------------------------------------------
    def _handle_punch_in(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        label = payload.get("label")
        if not label or not str(label).strip():
            raise ValueError("label is required for punch_in")

        start_dt = self._parse_time_input(payload.get("start_time"))
        created_dt = utc_now()
        notes = payload.get("notes")

        session_id = str(uuid.uuid4())
        start_iso = ensure_utc(start_dt).isoformat()
        created_iso = ensure_utc(created_dt).isoformat()

        session = {
            "id": session_id,
            "label": str(label).strip(),
            "notes": notes,
            "status": "running",
            "segments": [],
            "active_segment_start": start_iso,
            "paused_at": None,
            "total_seconds": 0.0,
            "first_started_at": start_iso,
            "created_at": created_iso,
            "updated_at": created_iso,
            "completed_at": None,
            "history": [
                {
                    "event": "punch_in",
                    "at": created_iso,
                }
            ],
        }

        insert_punchclock_session(session)

        return {"session": self._serialize_session(session)}

    def _handle_pause(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required for pause")

        session = self._get_session(session_id)

        if session.get("status") != "running":
            raise ValueError("Session must be running to pause")

        start_iso = session.get("active_segment_start")
        if not start_iso:
            raise ValueError("Session is missing active start time")

        start_dt = self._iso_to_dt(start_iso)
        pause_dt = self._parse_time_input(payload.get("time"))

        if start_dt and pause_dt < start_dt:
            raise ValueError("pause time cannot be before the active segment started")

        duration = max(0.0, (pause_dt - start_dt).total_seconds()) if start_dt else 0.0

        segment = {
            "start": start_iso,
            "end": ensure_utc(pause_dt).isoformat(),
            "duration_seconds": duration,
        }
        session.setdefault("segments", []).append(segment)
        session["total_seconds"] = float(session.get("total_seconds", 0.0) + duration)
        session["status"] = "paused"
        session["paused_at"] = ensure_utc(pause_dt).isoformat()
        session["active_segment_start"] = None
        session["updated_at"] = ensure_utc(pause_dt).isoformat()
        session.setdefault("history", []).append({"event": "pause", "at": session["paused_at"]})

        update_punchclock_session(session)
        return {"session": self._serialize_session(session)}

    def _handle_resume(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required for resume")

        session = self._get_session(session_id)

        if session.get("status") != "paused":
            raise ValueError("Session must be paused to resume")

        resume_dt = self._parse_time_input(payload.get("time"))
        paused_iso = session.get("paused_at")
        paused_dt = self._iso_to_dt(paused_iso)

        if paused_dt and resume_dt < paused_dt:
            raise ValueError("resume time cannot be before the pause time")

        resume_iso = ensure_utc(resume_dt).isoformat()
        session["status"] = "running"
        session["active_segment_start"] = resume_iso
        session["paused_at"] = None
        session["updated_at"] = resume_iso
        session.setdefault("history", []).append({"event": "resume", "at": resume_iso})

        update_punchclock_session(session)
        return {"session": self._serialize_session(session)}

    def _handle_punch_out(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required for punch_out")

        session = self._get_session(session_id)

        if session.get("status") not in {"running", "paused"}:
            raise ValueError("Session must be running or paused to punch out")

        end_dt = self._parse_time_input(payload.get("time"))

        if session.get("status") == "running":
            start_iso = session.get("active_segment_start")
            if not start_iso:
                raise ValueError("Session is missing active start time")
            start_dt = self._iso_to_dt(start_iso)
            if start_dt and end_dt < start_dt:
                raise ValueError("punch_out time cannot be before the active segment started")
            duration = max(0.0, (end_dt - start_dt).total_seconds()) if start_dt else 0.0
            session.setdefault("segments", []).append(
                {
                    "start": start_iso,
                    "end": ensure_utc(end_dt).isoformat(),
                    "duration_seconds": duration,
                }
            )
            session["total_seconds"] = float(session.get("total_seconds", 0.0) + duration)
        else:
            paused_iso = session.get("paused_at")
            paused_dt = self._iso_to_dt(paused_iso)
            if paused_dt and end_dt < paused_dt:
                raise ValueError("punch_out time cannot be before the pause time")

        end_iso = ensure_utc(end_dt).isoformat()
        session["status"] = "completed"
        session["completed_at"] = end_iso
        session["updated_at"] = end_iso
        session["active_segment_start"] = None
        session["paused_at"] = None
        session.setdefault("history", []).append({"event": "punch_out", "at": end_iso})

        update_punchclock_session(session)
        return {"session": self._serialize_session(session)}

    def _handle_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sessions = list_punchclock_sessions()
        now = utc_now()

        session_id = payload.get("session_id")
        if session_id:
            session = self._get_session(session_id)
            return {"session": self._serialize_session(session, now=now)}

        include_completed = bool(payload.get("include_completed", True))
        max_completed = payload.get("max_completed", 5)

        running: List[Dict[str, Any]] = []
        paused: List[Dict[str, Any]] = []
        completed: List[Dict[str, Any]] = []

        for session in sessions:
            serialized = self._serialize_session(session, now=now)
            status = serialized.get("status")
            if status == "running":
                running.append(serialized)
            elif status == "paused":
                paused.append(serialized)
            elif status == "completed":
                completed.append(serialized)

        completed.sort(key=lambda s: s.get("completed_at") or "", reverse=True)
        if include_completed:
            if isinstance(max_completed, int) and max_completed > 0:
                completed = completed[:max_completed]
        else:
            completed = []

        return {
            "running": running,
            "paused": paused,
            "completed": completed,
        }

    # Working memory integration -------------------------------------------------
    def _publish_trinket_refresh(self) -> None:
        if self._working_memory is None:
            return

        try:
            self._working_memory.publish_trinket_update(
                target_trinket="PunchclockTrinket",
                context={},
            )
        except Exception as exc:
            logger.debug("Failed to publish punchclock trinket update: %s", exc)
