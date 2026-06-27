"""
WhileTheCatsAway Tool — Dispatch background curiosity-driven research.

Mira calls this when she wants to learn more about a topic outside the
active conversation. Fire-and-forget: the agent runs in batch mode
(cheap, slow) and stores findings as LT_Memory entries that surface
naturally in future conversations.
"""
import logging
import threading
import uuid
from typing import Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.user_context import get_current_user_id, get_current_user

if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory


# -------------------- CONFIGURATION --------------------

class WhileTheCatsAwayToolConfig(BaseModel):
    """Configuration for whilethecatsaway_tool."""
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )


registry.register("whilethecatsaway_tool", WhileTheCatsAwayToolConfig)


# -------------------- TOOL --------------------

class WhileTheCatsAwayTool(Tool):
    """Dispatches a background agent for curiosity-driven research."""

    name = "whilethecatsaway_tool"

    simple_description = """
    Dispatch a background agent to explore a topic you're curious about.
    Runs asynchronously in batch mode — no rush, no cost pressure.
    Findings are stored as long-term memories that surface naturally later.
    Use when you want to learn more about something but the user isn't
    waiting for the answer right now.
    """

    tool_schema = {
        "name": "whilethecatsaway_tool",
        "description": (
            "Dispatch a background agent to explore a topic at leisure. "
            "Runs in batch mode (cheap, slow) and stores findings as "
            "long-term memories. Use when you're curious about something "
            "and want to learn more outside the active conversation. "
            "Fire-and-forget — no results returned to the current conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "What you want to learn about. Be specific enough to "
                        "guide research but open enough for exploration. "
                        "Example: 'the history and current state of "
                        "livestock guardian dog breeds in the American West'"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Why you're curious about this. What conversation or "
                        "thought sparked the interest? The agent uses this to "
                        "judge what angles are worth pursuing. "
                        "Example: 'Taylor mentioned their neighbor uses Great "
                        "Pyrenees for predator deterrence and I realized I "
                        "know very little about working LGD breeds beyond "
                        "the basics.'"
                    ),
                },
            },
            "required": ["topic"],
        },
    }

    def __init__(
        self,
        tool_repo: 'ToolRepository',
        working_memory: 'WorkingMemory',
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_repo = tool_repo
        self.working_memory = working_memory

        self.event_bus = None
        if self.working_memory:
            self.event_bus = self.working_memory.event_bus

    def run(self, **kwargs) -> Dict[str, Any]:
        """Dispatch a whilethecatsaway agent."""
        topic = kwargs.get('topic')
        context = kwargs.get('context', '')

        if not topic:
            raise ValueError("topic is required")

        return self._dispatch(topic, context)

    def _dispatch(self, topic: str, context: str) -> Dict[str, Any]:
        """Launch a background whilethecatsaway agent."""
        task_id = str(uuid.uuid4())

        user_context = get_current_user()
        user_id = get_current_user_id()
        continuum_id = user_context.get('continuum_id', user_id)

        from contextvars import copy_context
        ctx = copy_context()

        thread = threading.Thread(
            target=ctx.run,
            args=(self._run_agent, task_id, continuum_id, topic, context),
            name=f"whilethecatsaway-{task_id[:8]}",
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "task_id": task_id,
            "message": (
                f"Background research dispatched for: '{topic}'. "
                "The agent will explore at its own pace and store "
                "findings as memories."
            ),
        }

    def _run_agent(
        self, task_id: str, continuum_id: str, topic: str, context: str,
    ) -> None:
        """Background thread entry point — runs WhileTheCatsAwayAgent."""
        try:
            from agents.implementations.whilethecatsaway_agent import (
                WhileTheCatsAwayAgent,
            )
            from agents.sidebar import WorkItem

            work_item = WorkItem(
                item_id=task_id,
                interface_name="whilethecatsaway",
                context={"topic": topic, "context": context},
            )

            agent = WhileTheCatsAwayAgent(tool_repo=self.tool_repo)
            agent.run(work_item, self.event_bus)

        except Exception as e:
            self.logger.error(
                f"WhileTheCatsAway agent thread crashed for task "
                f"{task_id[:8]}: {e}",
                exc_info=True,
            )
