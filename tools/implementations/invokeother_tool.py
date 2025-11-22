"""
Dynamic tool loading meta-tool for MIRA.

This tool allows the LLM to dynamically load and unload tools on demand,
reducing context window usage by 80-90% compared to loading all tools upfront.
Tool hints are always visible via working memory, enabling intelligent loading decisions.
"""

# Standard library imports
import json
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from tools.repo import Tool
from tools.registry import registry

# Type checking imports
if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory


# -------------------- CONFIGURATION --------------------

class InvokeOtherToolConfig(BaseModel):
    """
    Configuration for the invokeother_tool.

    Defines the parameters that control the dynamic tool loader's behavior.
    """
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled by default"
    )
    idle_threshold: int = Field(
        default=5,
        description="Number of turns before an unused tool is automatically unloaded"
    )

# Register with registry
registry.register("invokeother_tool", InvokeOtherToolConfig)


# -------------------- MAIN TOOL CLASS --------------------

class InvokeOtherTool(Tool):
    """
    Meta-tool for dynamic tool loading and management.

    This tool provides on-demand loading of other tools to optimize context usage.
    Tool hints are maintained in working memory via ToolLoaderTrinket, allowing
    the LLM to see all available tools without loading their full definitions.
    """

    name = "invokeother_tool"

    simple_description = """
    Dynamically load and unload tools to manage context efficiently. Check working memory for available tools."""

    anthropic_schema = {
        "name": "invokeother_tool",
        "description": "Dynamically load and unload tools to manage context efficiently. Check working memory for available tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["load", "unload", "fallback"],
                    "description": "Operation mode: load tools, unload tools, or emergency fallback"
                },
                "query": {
                    "type": "string",
                    "description": "Tool name(s) to load/unload, comma-separated. Ignored for fallback mode."
                }
            },
            "required": ["mode"],
            "additionalProperties": False
        }
    }

    def __init__(self, tool_repo: 'ToolRepository', working_memory: 'WorkingMemory'):
        """
        Initialize the dynamic tool loader.

        Args:
            tool_repo: Repository for enabling/disabling tools
            working_memory: Working memory for publishing tool state updates
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_repo = tool_repo
        self.working_memory = working_memory

        # Get configuration
        from config.config_manager import config
        self.config = config.tools.invokeother_tool
        self.essential_tools = config.tools.essential_tools

        # Initialize tool hints in working memory on first load
        self._initialize_tool_hints()

    def _initialize_tool_hints(self) -> None:
        """Send initial tool hints to ToolLoaderTrinket via working memory."""
        try:
            # Collect all available tools and their descriptions
            available_tools = {}
            all_tools = self.tool_repo.list_all_tools()

            for tool_name in all_tools:
                # Skip essential tools and self
                if tool_name in self.essential_tools or tool_name == self.name:
                    continue

                # Skip tools disabled in config
                from config import config
                tool_config = getattr(config, tool_name, None)
                if tool_config:
                    is_enabled = getattr(tool_config, 'enabled', True)
                    if not is_enabled:
                        self.logger.debug(f"Skipping disabled tool {tool_name} in hints")
                        continue

                try:
                    # Get tool instance to access simple_description
                    tool = self.tool_repo.get_tool(tool_name)
                    if hasattr(tool, 'simple_description'):
                        # Use full simple_description
                        desc = tool.simple_description.strip()
                        available_tools[tool_name] = desc
                except Exception as e:
                    self.logger.warning(f"Could not get description for {tool_name}: {e}")

            # Send to ToolLoaderTrinket
            self.working_memory.publish_trinket_update(
                target_trinket="ToolLoaderTrinket",
                context={
                    "action": "initialize",
                    "available_tools": available_tools,
                    "essential_tools": self.essential_tools
                }
            )

            self.logger.info(f"Initialized ToolLoaderTrinket with {len(available_tools)} tool hints")

        except Exception as e:
            self.logger.error(f"Error initializing tool hints: {e}")

    def run(self, mode: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute tool loading operations.

        Args:
            mode: Operation mode (load, unload, fallback)
            query: Tool names for load/unload operations

        Returns:
            Dict containing operation results

        Raises:
            ValueError: If mode is invalid or operation fails
        """
        try:
            if mode == "load":
                return self._load_tools(query or "")
            elif mode == "unload":
                return self._unload_tools(query or "")
            elif mode == "fallback":
                return self._fallback_mode()
            else:
                self.logger.error(f"Invalid mode: {mode}")
                raise ValueError(f"Invalid mode: {mode}. Valid modes are: load, unload, fallback")

        except Exception as e:
            self.logger.error(f"Error in invokeother_tool: {e}")
            raise

    def _load_tools(self, query: str) -> Dict[str, Any]:
        """
        Load specified tools into context.

        Args:
            query: Comma-separated tool names

        Returns:
            Success status and loaded tools
        """
        if not query.strip():
            return {
                "success": False,
                "message": "No tools specified to load. Provide comma-separated tool names."
            }

        # Parse tool names
        requested_tools = [t.strip() for t in query.split(',') if t.strip()]
        loaded = []
        errors = []

        for tool_name in requested_tools:
            try:
                # Check if tool exists
                if tool_name not in self.tool_repo.list_all_tools():
                    errors.append(f"{tool_name} not found")
                    continue

                # Check if tool is enabled in config
                from config import config
                tool_config = getattr(config, tool_name, None)
                if tool_config:
                    is_enabled = getattr(tool_config, 'enabled', True)
                    if not is_enabled:
                        errors.append(f"{tool_name} is disabled in config (enabled=false)")
                        self.logger.warning(f"Attempted to load disabled tool: {tool_name}")
                        continue

                # Check if already enabled
                if self.tool_repo.is_tool_enabled(tool_name):
                    loaded.append(tool_name)
                    self.logger.debug(f"{tool_name} already enabled")
                    continue

                # Enable the tool
                self.tool_repo.enable_tool(tool_name)
                loaded.append(tool_name)

                # Notify ToolLoaderTrinket
                self.working_memory.publish_trinket_update(
                    target_trinket="ToolLoaderTrinket",
                    context={
                        "action": "tool_loaded",
                        "tool_name": tool_name
                    }
                )

                self.logger.info(f"Loaded tool: {tool_name}")

            except Exception as e:
                self.logger.error(f"Error loading {tool_name}: {e}")
                errors.append(f"{tool_name}: {str(e)}")

        # Build response
        if loaded and not errors:
            return {
                "success": True,
                "loaded": loaded,
                "message": f"Successfully loaded: {', '.join(loaded)}"
            }
        elif loaded and errors:
            return {
                "success": True,
                "loaded": loaded,
                "errors": errors,
                "message": f"Loaded {len(loaded)} tools with {len(errors)} errors"
            }
        else:
            return {
                "success": False,
                "errors": errors,
                "message": "Failed to load any tools"
            }

    def _unload_tools(self, query: str) -> Dict[str, Any]:
        """
        Unload specified tools from context.

        Args:
            query: Comma-separated tool names

        Returns:
            Success status and unloaded tools
        """
        if not query.strip():
            return {
                "success": False,
                "message": "No tools specified to unload. Provide comma-separated tool names."
            }

        # Parse tool names
        requested_tools = [t.strip() for t in query.split(',') if t.strip()]
        unloaded = []
        errors = []

        for tool_name in requested_tools:
            try:
                # Don't allow unloading essential tools
                if tool_name in self.essential_tools:
                    errors.append(f"{tool_name} is essential and cannot be unloaded")
                    continue

                # Check if enabled
                if not self.tool_repo.is_tool_enabled(tool_name):
                    self.logger.debug(f"{tool_name} already disabled")
                    continue

                # Disable the tool
                self.tool_repo.disable_tool(tool_name)
                unloaded.append(tool_name)

                # Notify ToolLoaderTrinket
                self.working_memory.publish_trinket_update(
                    target_trinket="ToolLoaderTrinket",
                    context={
                        "action": "tool_unloaded",
                        "tool_name": tool_name
                    }
                )

                self.logger.info(f"Unloaded tool: {tool_name}")

            except Exception as e:
                self.logger.error(f"Error unloading {tool_name}: {e}")
                errors.append(f"{tool_name}: {str(e)}")

        # Build response
        if unloaded and not errors:
            return {
                "success": True,
                "unloaded": unloaded,
                "message": f"Successfully unloaded: {', '.join(unloaded)}"
            }
        elif unloaded and errors:
            return {
                "success": True,
                "unloaded": unloaded,
                "errors": errors,
                "message": f"Unloaded {len(unloaded)} tools with {len(errors)} errors"
            }
        else:
            return {
                "success": False,
                "errors": errors,
                "message": "Failed to unload any tools"
            }

    def _fallback_mode(self) -> Dict[str, Any]:
        """
        Emergency mode: Load all available tools for one turn.

        Returns:
            Success status with all loaded tools
        """
        try:
            # Get all available tools
            all_tools = self.tool_repo.list_all_tools()
            non_essential = [t for t in all_tools
                           if t not in self.essential_tools and t != self.name]

            loaded = []
            for tool_name in non_essential:
                try:
                    if not self.tool_repo.is_tool_enabled(tool_name):
                        self.tool_repo.enable_tool(tool_name)
                        loaded.append(tool_name)
                except Exception as e:
                    self.logger.warning(f"Could not load {tool_name} in fallback: {e}")

            # Notify ToolLoaderTrinket about fallback mode
            self.working_memory.publish_trinket_update(
                target_trinket="ToolLoaderTrinket",
                context={
                    "action": "fallback_mode",
                    "loaded_tools": loaded
                }
            )

            total_loaded = len([t for t in all_tools if self.tool_repo.is_tool_enabled(t)])

            return {
                "success": True,
                "loaded": loaded,
                "total_active": total_loaded,
                "message": f"Fallback mode: All {total_loaded} tools loaded for this turn only"
            }

        except Exception as e:
            self.logger.error(f"Error in fallback mode: {e}")
            return {
                "success": False,
                "message": f"Fallback mode failed: {str(e)}"
            }