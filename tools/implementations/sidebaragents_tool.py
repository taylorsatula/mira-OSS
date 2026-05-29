"""
Sidebar Agents Tool -- Main conversation tool for managing sidebar activity.

Reads the same sidebar_activity and scratchpad SQLite tables that
sidebar_tool writes. Both use self.db (same UserDataManager per user).
"""
import html
import logging
from typing import Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from agents.base import ensure_activity_schema
from tools.repo import Tool
from tools.registry import registry

if TYPE_CHECKING:
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)


# -------------------- CONFIGURATION --------------------

class SidebarAgentsToolConfig(BaseModel):
    """Configuration for sidebaragents_tool."""
    enabled: bool = Field(
        default=True,
        description="Active in main conversation for managing sidebar activity",
    )


registry.register("sidebaragents_tool", SidebarAgentsToolConfig)


# -------------------- TOOL --------------------

class SidebarAgentsTool(Tool):
    """Review, inspect, and manage sidebar agent activity."""

    name = "sidebaragents_tool"

    tool_schema = {
        "name": "sidebaragents_tool",
        "description": (
            "Review and manage activity from sidebar agents (email, etc.).\n\n"
            "  list_activity(interface_name?): See all active items, "
            "optionally filtered by interface.\n"
            "  get_details(thread_id): Read the agent's working notes "
            "for a specific thread.\n"
            "  dismiss(thread_id): Remove an item from the activity feed.\n"
            "  resolve(thread_id): Mark a thread as resolved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "list_activity",
                        "get_details",
                        "dismiss",
                        "resolve",
                    ],
                },
                "interface_name": {
                    "type": "string",
                    "description": (
                        "Filter by interface. Optional for list_activity. "
                        "Example: 'email_watcher'."
                    ),
                },
                "thread_id": {
                    "type": "string",
                    "description": (
                        "Thread to inspect or act on. Required for "
                        "get_details, dismiss, resolve."
                    ),
                },
            },
            "required": ["operation"],
        },
    }

    def __init__(self, working_memory: 'WorkingMemory'):
        super().__init__()
        self._schema_ensured = False
        self.event_bus = working_memory.event_bus if working_memory else None

    def _ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        ensure_activity_schema(self.db)
        self._schema_ensured = True

    def run(self, **params) -> Dict[str, Any]:
        self._ensure_schema()
        operation = params.get("operation")

        if operation == "list_activity":
            return self._list_activity(params)
        elif operation == "get_details":
            return self._get_details(params)
        elif operation == "dismiss":
            return self._dismiss(params)
        elif operation == "resolve":
            return self._resolve(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _list_activity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        interface_name = params.get("interface_name")

        if interface_name:
            rows = self.db.select(
                "sidebar_activity",
                where="interface_name = :iface AND status != 'dismissed'",
                params={'iface': interface_name},
                order_by="updated_at DESC",
            )
        else:
            rows = self.db.select(
                "sidebar_activity",
                where="status != 'dismissed'",
                order_by="updated_at DESC",
            )

        items = [
            {
                "thread_id": r['thread_id'],
                "interface_name": r['interface_name'],
                "agent_id": r['agent_id'],
                "summary": r['summary'],
                "status": r['status'],
                "escalation_reason": r.get('escalation_reason'),
                "run_count": r.get('run_count', 1),
                "updated_at": r['updated_at'],
            }
            for r in rows
        ]
        return {"items": items, "count": len(items)}

    def _get_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("get_details requires thread_id")

        # Activity record
        activity_rows = self.db.select(
            "sidebar_activity",
            where="thread_id = :tid",
            params={'tid': thread_id},
        )
        activity = activity_rows[0] if activity_rows else None

        # Scratchpad content originates from untrusted input; escape before return
        note_rows = self.db.select(
            "scratchpad",
            where="thread_id = :tid",
            params={'tid': thread_id},
            order_by="created_at ASC",
        )
        notes = [
            {
                "note": html.escape(r['note']),
                "created_at": r['created_at'],
            }
            for r in note_rows
        ]

        result: Dict[str, Any] = {
            "thread_id": thread_id,
            "notes": notes,
        }
        if activity:
            result["activity"] = {
                "summary": activity['summary'],
                "status": activity['status'],
                "escalation_reason": activity.get('escalation_reason'),
                "updated_at": activity['updated_at'],
            }

        return result

    def _dismiss(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("dismiss requires thread_id")

        self.db.execute(
            "UPDATE sidebar_activity SET status = 'dismissed', "
            "updated_at = datetime('now') WHERE thread_id = :tid",
            {'tid': thread_id},
        )

        self._refresh_trinket()
        return {"success": True, "thread_id": thread_id, "action": "dismissed"}

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = params.get("thread_id")
        if not thread_id:
            raise ValueError("resolve requires thread_id")

        self.db.execute(
            "UPDATE sidebar_activity SET status = 'resolved', "
            "updated_at = datetime('now') WHERE thread_id = :tid",
            {'tid': thread_id},
        )

        self._refresh_trinket()
        return {"success": True, "thread_id": thread_id, "action": "resolved"}

    def _refresh_trinket(self) -> None:
        if self.event_bus is None:
            return
        from agents.base import _publish_trinket_refresh
        _publish_trinket_refresh(self.event_bus)
