"""
Sidebar Tool -- Agent lifecycle management for sidebar agents.

Two concerns:
1. Scratchpad: persistent working notes between agent iterations
2. Task completion: explicit signal that the agent is done

Registered but disabled (enabled=False) -- never appears in the main
conversation's tool schema. Sidebar agents access it via
tool_repo.get_tool('sidebar_tool'), which succeeds for disabled tools.
"""
import logging
from typing import Dict, Any

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry

logger = logging.getLogger(__name__)


# -------------------- CONFIGURATION --------------------

class SidebarToolConfig(BaseModel):
    """Configuration for sidebar_tool."""
    enabled: bool = Field(
        default=False,
        description="Disabled -- sidebar agents only, not main conversation",
    )


registry.register("sidebar_tool", SidebarToolConfig)


# -------------------- SCHEMAS --------------------
# Scratchpad + activity DDL live in agents.base.ensure_activity_schema()

AUDIT_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS sidebar_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    inbound_item_id TEXT NOT NULL,
    recipient TEXT NOT NULL,
    subject TEXT,
    body_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)"""


# -------------------- TOOL --------------------

class SidebarTool(Tool):
    """Agent lifecycle management: scratchpad and task completion."""

    name = "sidebar_tool"

    tool_schema = {
        "name": "sidebar_tool",
        "description": (
            "Manage your working state and signal task completion.\n\n"
            "SCRATCHPAD — persistent notes between your iterations:\n"
            "  write_note(note): Save an observation or plan.\n"
            "  read_notes(): Retrieve your prior notes.\n"
            "  clear_notes(): Remove notes for a resolved thread.\n\n"
            "COMPLETION — call when your task is done:\n"
            "  complete_task(summary, status): Write a one-line summary "
            "and set status to 'handled' or 'escalated'. If escalating, "
            "include escalation_reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "write_note",
                        "read_notes",
                        "clear_notes",
                        "complete_task",
                    ],
                },
                "note": {
                    "type": "string",
                    "description": (
                        "Note text to save. Required for write_note only."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "One-line summary of what you did. Required for "
                        "complete_task. Displayed in the activity feed."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": ["handled", "escalated"],
                    "description": (
                        "Outcome. Required for complete_task. 'handled' = "
                        "you acted successfully. 'escalated' = needs human."
                    ),
                },
                "escalation_reason": {
                    "type": "string",
                    "description": (
                        "Why this needs human attention. Required when "
                        "status is 'escalated'."
                    ),
                },
            },
            "required": ["operation"],
        },
    }

    def __init__(self):
        super().__init__()
        self._schema_ensured = False

    def _ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        from agents.base import ensure_activity_schema
        ensure_activity_schema(self.db)
        self.db.execute(AUDIT_TABLE_DDL)
        self._schema_ensured = True

    def run(self, **params) -> Dict[str, Any]:
        self._ensure_schema()
        operation = params.get("operation")

        if operation == "write_note":
            return self._write_note(params)
        elif operation == "read_notes":
            return self._read_notes(params)
        elif operation == "clear_notes":
            return self._clear_notes(params)
        elif operation == "complete_task":
            return self._complete_task(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # ------------------------------------------------------------------
    # Scratchpad operations
    # ------------------------------------------------------------------

    def _write_note(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        note = params.get("note")
        if not thread_id or not note:
            raise ValueError("write_note requires thread_id and note")

        self.db.execute(
            "INSERT INTO scratchpad (thread_id, note) "
            "VALUES (:thread_id, :note)",
            {'thread_id': thread_id, 'note': note},
        )
        return {"success": True, "message": "Note saved."}

    def _read_notes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("read_notes requires thread_id")

        rows = self.db.select(
            "scratchpad",
            where="thread_id = :thread_id",
            params={'thread_id': thread_id},
            order_by="created_at ASC",
        )
        notes = [
            {"note": r['note'], "created_at": r['created_at']}
            for r in rows
        ]
        return {"thread_id": thread_id, "notes": notes}

    def _clear_notes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("clear_notes requires thread_id")

        self.db.execute(
            "DELETE FROM scratchpad WHERE thread_id = :thread_id",
            {'thread_id': thread_id},
        )
        return {"success": True, "message": "Notes cleared."}

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def _complete_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write activity record to SQLite via UPSERT.

        The SidebarAgent base class enriches params with thread_id,
        interface_name, agent_id, and run_count from the WorkItem
        before execution.
        """
        summary = params.get("summary")
        status = params.get("status", "handled")
        if not summary:
            raise ValueError("complete_task requires summary")

        thread_id = params.get("thread_id", "unknown")
        interface_name = params.get("interface_name", "unknown")
        agent_id = params.get("agent_id", "unknown")
        escalation_reason = params.get("escalation_reason")
        run_count = params.get("run_count", 1)

        self.db.execute(
            "INSERT INTO sidebar_activity "
            "(interface_name, thread_id, agent_id, summary, status, "
            "escalation_reason, run_count, updated_at) "
            "VALUES (:interface_name, :thread_id, :agent_id, :summary, "
            ":status, :escalation_reason, :run_count, datetime('now')) "
            "ON CONFLICT(interface_name, thread_id) DO UPDATE SET "
            "agent_id = excluded.agent_id, summary = excluded.summary, "
            "status = excluded.status, escalation_reason = excluded.escalation_reason, "
            "run_count = excluded.run_count, updated_at = datetime('now')",
            {
                'interface_name': interface_name,
                'thread_id': thread_id,
                'agent_id': agent_id,
                'summary': summary,
                'status': status,
                'escalation_reason': escalation_reason,
                'run_count': run_count,
            },
        )

        return {
            "completed": True,
            "interface_name": interface_name,
            "thread_id": thread_id,
            "status": status,
            "summary": summary,
        }
