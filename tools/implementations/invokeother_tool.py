"""
Dynamic tool loading meta-tool for MIRA.

Allows the LLM to load tools on demand via constrained enum parameters.
Two loading modes: ephemeral (current turn only) and pinned (rest of session).
The tool catalog is encoded directly in the tool_schema for prompt cache
stability — no dynamic working memory content needed.
"""

import logging
from typing import Dict, Any, List, TYPE_CHECKING

from tools.repo import Tool

if TYPE_CHECKING:
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class InvokeOtherTool(Tool):
    """
    Meta-tool for on-demand tool loading with dual lifetime modes.

    The tool catalog (available tools + descriptions) is baked into the
    tool_schema at construction time, giving the LLM constrained
    enum choices and stable prompt caching.

    Two parameters serve as operations:
    - load: ephemeral tools, auto-unloaded after the current turn
    - load_for_rest_of_session: pinned tools, persist until session ends
    """

    name = "invokeother_tool"

    simple_description = """
    Dynamically load tools on demand. Available tools listed in schema."""

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__()
        self.tool_repo = tool_repo

        from tools.repo import ESSENTIAL_TOOLS
        self.essential_tools = ESSENTIAL_TOOLS

        # Build schema with tool catalog baked into description + enum
        catalog = self._build_catalog()
        self.tool_schema = self._build_schema(catalog)

    def _build_catalog(self) -> Dict[str, str]:
        """Build tool name -> description catalog for non-essential, non-gated tools."""
        from config import config

        catalog = {}
        all_tools = self.tool_repo.list_all_tools()

        for tool_name in all_tools:
            if tool_name in self.essential_tools or tool_name == self.name:
                continue
            if tool_name in self.tool_repo.gated_tools:
                continue

            # Skip config-disabled tools
            tool_config = getattr(config, tool_name, None)
            if tool_config and not getattr(tool_config, 'enabled', True):
                continue

            try:
                tool = self.tool_repo.get_tool(tool_name)
                if hasattr(tool, 'simple_description'):
                    catalog[tool_name] = tool.simple_description.strip()
            except Exception as e:
                logger.warning(f"Could not get description for {tool_name}: {e}")

        return catalog

    def _build_schema(self, catalog: Dict[str, str]) -> Dict[str, Any]:
        """Build tool_schema with dual load parameters and tool enum."""
        tool_names = sorted(catalog.keys())

        # Build description with inline catalog
        desc_parts = ["Load additional tools on demand.\n\nAvailable tools:"]
        for name in tool_names:
            summary = catalog[name].split('\n')[0]
            desc_parts.append(f"- {name}: {summary}")

        enum_schema = {"type": "string", "enum": tool_names} if tool_names else {"type": "string"}

        return {
            "name": "invokeother_tool",
            "description": "\n".join(desc_parts),
            "input_schema": {
                "type": "object",
                "properties": {
                    "load": {
                        "type": "array",
                        "items": enum_schema,
                        "description": "Tool names to make available for this turn only. Automatically removed from the tool list after the current response completes"
                    },
                    "load_for_rest_of_session": {
                        "type": "array",
                        "items": enum_schema,
                        "description": "Tool names to make available for the remainder of this conversation session. Use when the same tool is needed across multiple turns to avoid repeated loading"
                    }
                },
                "additionalProperties": False
            }
        }

    def run(self, load: List[str] = None, load_for_rest_of_session: List[str] = None) -> Dict[str, Any]:
        """
        Load tools with specified lifetime.

        Args:
            load: Tool names to load for this turn only (ephemeral).
            load_for_rest_of_session: Tool names to pin for the entire session.

        Returns:
            Dict containing operation results.
        """
        ephemeral = load or []
        pinned = load_for_rest_of_session or []

        if not ephemeral and not pinned:
            return {
                "success": False,
                "message": "No tools specified. Use 'load' for this-turn-only or 'load_for_rest_of_session' for persistent loading."
            }

        loaded = []
        errors = []

        # Process ephemeral tools
        for tool_name in ephemeral:
            result = self._enable_tool(tool_name)
            if result is None:
                loaded.append(tool_name)
            else:
                errors.append(result)

        # Process pinned tools — enable + add to pinned set
        for tool_name in pinned:
            result = self._enable_tool(tool_name)
            if result is None:
                self.tool_repo._pinned_tools.add(tool_name)
                loaded.append(tool_name)
                logger.info(f"Pinned tool for session: {tool_name}")
            else:
                errors.append(result)

        if loaded and not errors:
            return {"success": True, "loaded": loaded, "message": f"Loaded: {', '.join(loaded)}"}
        elif loaded:
            return {"success": True, "loaded": loaded, "errors": errors,
                    "message": f"Loaded {len(loaded)} tools with {len(errors)} errors"}
        else:
            return {"success": False, "errors": errors, "message": "Failed to load any tools"}

    def _enable_tool(self, tool_name: str) -> str | None:
        """
        Enable a single tool. Returns None on success, error string on failure.
        """
        try:
            if tool_name not in self.tool_repo.list_all_tools():
                return f"{tool_name} not found"

            # Check config-disabled
            from config import config
            tool_config = getattr(config, tool_name, None)
            if tool_config and not getattr(tool_config, 'enabled', True):
                logger.warning(f"Attempted to load disabled tool: {tool_name}")
                return f"{tool_name} is disabled in config"

            # Check gated tool availability
            if tool_name in self.tool_repo.gated_tools:
                try:
                    tool = self.tool_repo.get_tool(tool_name)
                    if not (hasattr(tool, 'is_available') and tool.is_available()):
                        return f"{tool_name} is not currently available (gated)"
                except Exception as e:
                    logger.warning(f"Gated tool availability check failed for {tool_name}: {e}")
                    return f"{tool_name}: availability check failed"

            # Already enabled — idempotent success
            if self.tool_repo.is_tool_enabled(tool_name):
                return None

            self.tool_repo.enable_tool(tool_name)
            logger.info(f"Loaded tool: {tool_name}")
            return None

        except Exception as e:
            logger.error(f"Error loading {tool_name}: {e}")
            return f"{tool_name}: {str(e)}"
