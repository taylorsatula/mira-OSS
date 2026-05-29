"""
Forage Tool — Speculative background context gathering.

Dispatches a background Agent to forage across conversations, memories,
and web. The primary LLM provides a query and conversational context;
the agent searches autonomously and publishes a written briefing to
the ForageTrinket when complete.

Fire-and-forget: returns instantly, results appear in the context window
when the agent finishes.
"""
import logging
import threading
import uuid
from typing import Dict, Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.user_context import get_current_user_id, get_current_user

if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory


# -------------------- CONFIGURATION --------------------

class ForageToolConfig(BaseModel):
    """Configuration for forage_tool."""
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled",
    )
    max_iterations: int = Field(
        default=12,
        description="Maximum agent loop iterations before forcing a summary",
    )
    search_timeout_seconds: int = Field(
        default=120,
        description="Maximum wall-clock seconds before the agent reports timeout",
    )


registry.register("forage_tool", ForageToolConfig)


# -------------------- TOOL --------------------

class ForageTool(Tool):
    """Dispatches a background forage agent for speculative context gathering."""

    name = "forage_tool"

    simple_description = """
    Dispatches a background Agent to forage across conversations, memories, and web.
    Searches in parallel for context relevant to the current topic.
    Returns instantly — findings surface in your context window when ready.
    Forage early: an empty search is cheap, but context you never looked for can't help you.
    """

    tool_schema = {
        "name": "forage_tool",
        "description": (
            "Dispatch a background Agent to forage for context across conversations, "
            "memories, and web. Returns instantly — results appear in your context window "
            "when ready. Use dismiss_task_id to remove a stale or unhelpful result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for. Be specific — this drives the agent's "
                        "search strategy. Example: 'agility training centers near "
                        "Portland with good reviews for herding breeds'"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Conversational context motivating the search. What are you "
                        "and the user discussing, and why would this information be "
                        "useful? The agent uses this to judge relevance. "
                        "Example (paired with query above): 'User and I are discussing "
                        "English Shepherd training. Their facility in Hillsboro closed "
                        "last month and they need a new one for their 2-year-old dog "
                        "Birch who has completed basic obedience but not started agility.'"
                    ),
                },
                "refine_task_id": {
                    "type": "string",
                    "description": (
                        "Task ID of a previous forage result to refine. Must be paired "
                        "with query and context. The previous result is auto-dismissed "
                        "and passed to the new Agent as starting context. Query becomes "
                        "the refinement instruction. Example: original forage for "
                        "'agility training centers near Portland' returned generic "
                        "results — set query to 'ones that specialize in large breeds "
                        "over 80 pounds' and provide this task ID."
                    ),
                },
                "dismiss_task_id": {
                    "type": "string",
                    "description": (
                        "Task ID of a forage result to dismiss from the context window. "
                        "Mutually exclusive with query/context — providing this ignores "
                        "both query and context."
                    ),
                },
            },
            "required": [],
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

        from config.config_manager import config
        self.config = config.forage_tool

        self.event_bus = None
        if self.working_memory:
            self.event_bus = self.working_memory.event_bus

    def run(self, **kwargs) -> Dict[str, Any]:
        """Dispatch a forage agent, refine a previous result, or dismiss."""
        dismiss_task_id = kwargs.get('dismiss_task_id')
        if dismiss_task_id:
            return self._dismiss(dismiss_task_id)

        query = kwargs.get('query')
        context = kwargs.get('context', '')
        refine_task_id = kwargs.get('refine_task_id')

        if not query:
            raise ValueError("query is required when not dismissing")

        return self._dispatch(query, context, refine_task_id)

    def _dispatch(self, query: str, context: str,
                  refine_task_id: Optional[str] = None) -> Dict[str, Any]:
        """Launch a background forage agent, optionally refining a previous result."""
        task_id = str(uuid.uuid4())

        user_context = get_current_user()
        user_id = get_current_user_id()
        continuum_id = user_context.get('continuum_id', user_id)

        # If refining, extract previous result and auto-dismiss it
        previous_result = None
        if refine_task_id and self.working_memory:
            trinket = self.working_memory.get_trinket('ForageTrinket')
            if trinket and refine_task_id in trinket.active_results:
                previous_data = trinket.active_results[refine_task_id].get('data', {})
                previous_result = previous_data.get('result', '')
            self._publish_event(continuum_id, refine_task_id, 'dismissed', {})

        # Resolve trace directory while user context is available (main thread)
        trace_dir = str(self.user_data_path)

        # Publish pending state to trinket immediately
        self._publish_event(continuum_id, task_id, 'pending', {'query': query})

        # Copy ambient context for the background thread
        from contextvars import copy_context
        ctx = copy_context()

        thread = threading.Thread(
            target=ctx.run,
            args=(self._run_agent, task_id, continuum_id, query, context,
                  previous_result, trace_dir),
            name=f"forage-{task_id[:8]}",
        )
        thread.daemon = True
        thread.start()

        return {
            "success": True,
            "task_id": task_id,
            "message": f"Forage agent dispatched for: '{query}'. Results will appear in your context window when ready.",
        }

    def _run_agent(self, task_id: str, continuum_id: str, query: str,
                   context: str, previous_result: Optional[str] = None,
                   trace_dir: Optional[str] = None) -> None:
        """Background thread entry point — runs ForageAgent."""
        try:
            from agents.implementations.forage_agent import ForageAgent
            from agents.sidebar import WorkItem

            work_item = WorkItem(
                item_id=task_id,
                interface_name="forage",
                context={
                    "query": query,
                    "context": context,
                    "previous_result": previous_result,
                },
            )

            agent = ForageAgent(tool_repo=self.tool_repo)
            agent.run(work_item, self.event_bus)

        except Exception as e:
            self.logger.error(
                f"Forage agent thread crashed for task {task_id[:8]}: {e}",
                exc_info=True,
            )
            self._publish_event(continuum_id, task_id, 'failed', {
                'error': str(e),
                'error_type': 'ThreadCrash',
            })

    def _dismiss(self, task_id: str) -> Dict[str, Any]:
        """Remove a forage result from the context window."""
        user_context = get_current_user()
        user_id = get_current_user_id()
        continuum_id = user_context.get('continuum_id', user_id)

        self._publish_event(continuum_id, task_id, 'dismissed', {})

        return {
            "success": True,
            "task_id": task_id,
            "message": f"Forage result {task_id[:8]} dismissed.",
        }

    def _publish_event(self, continuum_id: str, task_id: str,
                       status: str, data: Dict[str, Any]) -> None:
        """Publish a status update to the ForageTrinket."""
        if self.event_bus is None:
            return
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket='ForageTrinket',
            context={'task_id': task_id, 'status': status, **data},
        ))
