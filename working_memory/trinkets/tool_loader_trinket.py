"""
Tool loader state management trinket.

Tracks loaded tools, maintains available tool hints in working memory,
and handles automatic cleanup of idle tools via TurnCompletedEvent.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


@dataclass
class LoadedToolInfo:
    """Information about a loaded tool."""
    loaded_turn: int
    last_used_turn: int
    description: str
    is_fallback: bool = False  # True if loaded via fallback mode


class ToolLoaderTrinket(EventAwareTrinket):
    """
    Manages dynamic tool loading state for the system.

    This trinket:
    1. Maintains list of available tools with descriptions (always visible)
    2. Tracks currently loaded tools and their usage
    3. Handles auto-cleanup of idle tools via TurnCompletedEvent
    4. Publishes tool availability info to working memory
    """

    def __init__(self, event_bus, working_memory, tool_repo=None):
        """
        Initialize the tool loader trinket.

        Args:
            event_bus: CNS event bus for subscribing to events
            working_memory: Working memory instance for registration
            tool_repo: Optional tool repository for cleanup operations
        """
        super().__init__(event_bus, working_memory)

        # State tracking
        self.available_tools: Dict[str, str] = {}  # tool_name -> description
        self.loaded_tools: Dict[str, LoadedToolInfo] = {}  # tool_name -> info
        self.essential_tools: List[str] = []
        self.current_turn: int = 0
        self.tool_repo = tool_repo

        # Configuration
        from config.config_manager import config
        self.idle_threshold = config.tools.invokeother_tool.idle_threshold

        # Subscribe to turn completed events for cleanup
        self.event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)
        logger.info("ToolLoaderTrinket initialized and subscribed to TurnCompletedEvent")

    def _get_variable_name(self) -> str:
        """Tool loader publishes to 'tool_hints'."""
        return "tool_hints"

    def handle_update_request(self, event) -> None:
        """
        Handle update requests from InvokeOtherTool or other sources.

        Processes actions: initialize, tool_loaded, tool_unloaded,
        tool_used, fallback_mode
        """
        # Process specific tool loader actions
        context = event.context
        action = context.get('action')

        if action == 'initialize':
            self._handle_initialize(context)
        elif action == 'tool_loaded':
            self._handle_tool_loaded(context)
        elif action == 'tool_unloaded':
            self._handle_tool_unloaded(context)
        elif action == 'tool_used':
            self._handle_tool_used(context)
        elif action == 'fallback_mode':
            self._handle_fallback_mode(context)

        # Call parent to handle standard trinket update flow
        # This will call generate_content and publish the result
        super().handle_update_request(event)

    def _handle_initialize(self, context: Dict[str, Any]) -> None:
        """Initialize with available tools and essential tools list."""
        self.available_tools = context.get('available_tools', {})
        self.essential_tools = context.get('essential_tools', [])
        logger.info(f"Initialized with {len(self.available_tools)} available tools")

    def _handle_tool_loaded(self, context: Dict[str, Any]) -> None:
        """Track a newly loaded tool."""
        tool_name = context.get('tool_name')
        if not tool_name:
            return

        # Move from available to loaded
        description = self.available_tools.pop(tool_name, "")
        self.loaded_tools[tool_name] = LoadedToolInfo(
            loaded_turn=self.current_turn,
            last_used_turn=self.current_turn,
            description=description,
            is_fallback=False
        )
        logger.debug(f"Tool {tool_name} loaded at turn {self.current_turn}")

    def _handle_tool_unloaded(self, context: Dict[str, Any]) -> None:
        """Move tool back to available."""
        tool_name = context.get('tool_name')
        if not tool_name or tool_name not in self.loaded_tools:
            return

        # Move from loaded back to available
        tool_info = self.loaded_tools.pop(tool_name)
        if tool_info.description and tool_name not in self.essential_tools:
            self.available_tools[tool_name] = tool_info.description
        logger.debug(f"Tool {tool_name} unloaded")

    def _handle_tool_used(self, context: Dict[str, Any]) -> None:
        """Update last used turn for a tool."""
        tool_name = context.get('tool_name')
        if tool_name in self.loaded_tools:
            self.loaded_tools[tool_name].last_used_turn = self.current_turn
            logger.debug(f"Tool {tool_name} used at turn {self.current_turn}")

    def _handle_fallback_mode(self, context: Dict[str, Any]) -> None:
        """Mark all loaded tools as fallback mode for aggressive cleanup."""
        loaded_tools = context.get('loaded_tools', [])
        for tool_name in loaded_tools:
            if tool_name in self.available_tools:
                description = self.available_tools.pop(tool_name, "")
                self.loaded_tools[tool_name] = LoadedToolInfo(
                    loaded_turn=self.current_turn,
                    last_used_turn=self.current_turn,
                    description=description,
                    is_fallback=True  # Mark for cleanup after 1 turn
                )
        logger.info(f"Fallback mode: loaded {len(loaded_tools)} tools")

    def _handle_turn_completed(self, event) -> None:
        """
        Handle turn completion - check for idle tools to clean up.

        Event handler continues processing even if individual tool cleanup fails,
        but distinguishes infrastructure failures from logic errors.
        """
        try:
            # Update current turn from event (calculated from message count, survives restarts)
            self.current_turn = event.turn_number

            # Find tools to clean up
            tools_to_cleanup = []
            for tool_name, tool_info in self.loaded_tools.items():
                # Skip essential tools
                if tool_name in self.essential_tools:
                    continue

                # Fallback tools cleaned up after 1 turn
                if tool_info.is_fallback:
                    tools_to_cleanup.append(tool_name)
                    continue

                # Regular tools cleaned up after idle threshold
                idle_turns = self.current_turn - tool_info.last_used_turn
                if idle_turns > self.idle_threshold:
                    tools_to_cleanup.append(tool_name)
                    logger.debug(f"Tool {tool_name} idle for {idle_turns} turns")

            # Cleanup idle tools
            if tools_to_cleanup:
                logger.info(f"Auto-cleaning {len(tools_to_cleanup)} idle tools at turn {self.current_turn}")

                for tool_name in tools_to_cleanup:
                    try:
                        # Disable in repository if available
                        if self.tool_repo:
                            if self.tool_repo.is_tool_enabled(tool_name):
                                self.tool_repo.disable_tool(tool_name)

                        # Move back to available
                        tool_info = self.loaded_tools.pop(tool_name, None)
                        if tool_info and tool_info.description:
                            self.available_tools[tool_name] = tool_info.description

                    except Exception as e:
                        # Event handler continues - categorize error for observability
                        error_type = type(e).__name__
                        if 'Database' in error_type or 'Valkey' in error_type or 'Connection' in error_type:
                            logger.error(
                                f"Infrastructure failure cleaning up tool {tool_name}: {e}",
                                exc_info=True,
                                extra={'error_category': 'infrastructure'}
                            )
                        else:
                            logger.error(
                                f"Error cleaning up tool {tool_name}: {e}",
                                exc_info=True,
                                extra={'error_category': 'logic'}
                            )

                # Trigger an update to refresh content after cleanup
                self.working_memory.publish_trinket_update(
                    target_trinket="ToolLoaderTrinket",
                    context={"action": "cleanup_completed"}
                )

        except Exception as e:
            # Outer handler catches catastrophic failures in event processing
            logger.error(f"Critical error in turn completion handler: {e}", exc_info=True)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate tool hints content for working memory.

        Returns formatted list of available and loaded tools.
        """
        parts = []

        # Available tools section
        if self.available_tools:
            parts.append("=== AVAILABLE TOOLS ===")
            parts.append("Use invokeother_tool to load these when needed:")
            parts.append("")
            for tool_name, description in sorted(self.available_tools.items()):
                # Format multi-line descriptions properly
                lines = description.split('\n')
                parts.append(f"- {tool_name}: {lines[0]}")
                for line in lines[1:]:
                    if line.strip():
                        parts.append(f"  {line}")

        # Currently loaded tools section
        if self.loaded_tools:
            if parts:  # Add spacing if we had available tools
                parts.append("")
            parts.append("=== CURRENTLY LOADED TOOLS ===")
            parts.append("")

            for tool_name, tool_info in sorted(self.loaded_tools.items()):
                idle_turns = self.current_turn - tool_info.last_used_turn
                status = []

                if tool_info.is_fallback:
                    status.append("fallback mode - will unload next turn")
                elif idle_turns > 0:
                    status.append(f"idle {idle_turns} turn{'s' if idle_turns != 1 else ''}")

                status_str = f" ({', '.join(status)})" if status else ""
                parts.append(f"- {tool_name}{status_str}")

        return "\n".join(parts) if parts else ""