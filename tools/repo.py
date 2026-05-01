"""
Tool repository for the botwithmemory system.

This module provides the base Tool class and ToolRepository for managing, 
discovering, and using tools within the continuum system.
"""
import inspect
import importlib
import json
import logging
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Type, Union, get_args, get_origin
from pathlib import Path

from pydantic import BaseModel, create_model
from utils.user_context import get_current_user_id
from utils.userdata_manager import get_user_data_manager

from tools.registry import registry


def get_config():
    from config import config
    return config


ESSENTIAL_TOOLS = [
    "web_tool", "invokeother_tool", "continuum_tool", "reminder_tool",
    "memory_tool", "domaindoc_tool", "forage_tool", "sidebaragents_tool",
    "email_tool"
]

# Anthropic beta feature constants
CODE_EXECUTION_BETA_FLAG = "code-execution-2025-08-25"
FILES_API_BETA_FLAG = "files-api-2025-04-14"

# Code execution tool definition
CODE_EXECUTION = {"type": "code_execution_20250825", "name": "code_execution"}

# Combined beta flags for Anthropic API calls
ANTHROPIC_BETA_FLAGS = [CODE_EXECUTION_BETA_FLAG, FILES_API_BETA_FLAG]


class Tool(ABC):
    """
    Base class for all tools in the botwithmemory system.

    This class defines the standard interface and behavior that all tools
    should implement. It includes metadata, parameter handling, and execution logic.

    Class Attributes:
        name (str): The unique name of the tool.
        description (str): A human-readable description of the tool's purpose.
        usage_examples (List[Dict]): Example usage of the tool.
        parallel_safe (bool): Whether this tool can safely be called in parallel with
            other tools. Set to False for tools that mutate shared state where ordering
            matters (e.g., create-then-edit operations). Default is True.
            For per-operation control, override is_call_parallel_safe() instead.
    """

    name = "base_tool"
    description = "Base class for all tools"
    usage_examples: List[Dict[str, Any]] = []
    parallel_safe: bool = True

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        """Determine if a specific tool call can run in parallel.

        Override this in tools with mixed parallel/sequential operations.
        The execution engine calls this per tool call to decide scheduling.

        Args:
            tool_input: The input dict for this tool call (contains 'operation', etc.)
        """
        return cls.parallel_safe

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool configuration and return any discovered data.

        Override this in tools that need custom validation (e.g., connection tests,
        auto-discovery of settings). Called by /actions/tools/{tool}/validate endpoint.

        Args:
            config: The configuration dict to validate

        Returns:
            Dict with discovered data (e.g., {"folders": [...], "discovered_folders": {...}})
            Return empty dict if no special validation needed.

        Raises:
            ValueError: If validation fails
        """
        return {}

    def __init__(self):
        self.logger = logging.getLogger(f"tools.{self.name}")
        self._db = None
        
        if self.name not in registry._registry:
            self.logger.debug(f"Auto-registering default config for tool: {self.name}")
            
            class_name = f"{self.name.capitalize()}Config"
            if self.name.endswith('_tool'):
                # Generate CamelCase class name: reminder_tool -> ReminderToolConfig
                parts = self.name.split('_')
                class_name = ''.join(part.capitalize() for part in parts[:-1]) + 'ToolConfig'
            
            default_config = create_model(
                class_name,
                __base__=BaseModel,
                enabled=(bool, True),
                __doc__=f"Default configuration for {self.name}"
            )

            registry.register(self.__class__.name, default_config)
    
    @property
    def user_id(self) -> str:
        return get_current_user_id()
    
    @property
    def user_data_path(self) -> Path:
        from utils.userdata_manager import get_user_data_manager
        user_data = get_user_data_manager(self.user_id)
        return user_data.get_tool_data_dir(self.name)
    
    @property
    def db(self):
        current_user_id = self.user_id
        if not self._db or self._db.user_id != current_user_id:
            self._db = get_user_data_manager(current_user_id)
        return self._db
    
    # User-aware file operations - tools can use these without knowing about user scoping
    
    def make_dir(self, path: str) -> Path:
        full_path = self.user_data_path / path
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path
    
    def get_file_path(self, filename: str) -> Path:
        return self.user_data_path / filename
    
    def open_file(self, filename: str, mode: str = 'r'):
        return open(self.get_file_path(filename), mode)
    
    def file_exists(self, filename: str) -> bool:
        return self.get_file_path(filename).exists()
    
    @abstractmethod
    def run(self, **params) -> Dict[str, Any]:
        """
        Execute the tool with the provided parameters.
        
        Args:
            **params: Keyword arguments containing the tool's parameters.
            
        Returns:
            A dictionary containing the tool's response.
        """
        raise NotImplementedError("Tool subclasses must implement the run method")
    
    def get_metadata(self) -> Dict[str, Any]:
        # Extract parameter metadata from run method signature
        sig = inspect.signature(self.run)
        parameters = {}
        required_parameters = []
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
                
            param_info = {
                "type": "any",
                "description": f"Parameter: {name}"
            }
            
            if param.default is inspect.Parameter.empty:
                required_parameters.append(name)
            
            if param.annotation is not inspect.Parameter.empty:
                param_info["type"] = str(param.annotation).replace("<class '", "").replace("'>", "")
            
            parameters[name] = param_info
        
        # Parse docstring for parameter descriptions using state machine
        if self.run.__doc__:
            doc_content = inspect.getdoc(self.run)
            if doc_content is not None:
                doc_lines = doc_content.split('\n')
            
            # State machine to parse Google-style docstring Args section
            param_section = False
            current_param = None
            
            for line in doc_lines:
                line = line.strip()
                
                if line.lower().startswith('args:'):
                    param_section = True
                    continue
                
                if param_section and (not line or line.lower().startswith(('returns:', 'raises:'))):
                    param_section = False
                    current_param = None
                    continue
                
                if param_section:
                    import re
                    param_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:(.*)$', line)
                    
                    if param_match:
                        current_param = param_match.group(1).strip()
                        description = param_match.group(2).strip()
                        
                        if current_param in parameters:
                            parameters[current_param]["description"] = description
                    
                    elif current_param and current_param in parameters:
                        parameters[current_param]["description"] += " " + line
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "required_parameters": required_parameters,
            "usage_examples": self.usage_examples
        }
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_formatted_description(self) -> str:
        metadata = self.get_metadata()
        
        result = f"{metadata['name']}: {metadata['description']}\n"
        
        if metadata['parameters']:
            result += "Parameters:\n"
            for param_name, param_spec in metadata['parameters'].items():
                required = " (required)" if param_name in metadata['required_parameters'] else ""
                param_desc = param_spec.get("description", "No description")
                result += f"  - {param_name}{required}: {param_desc}\n"
        
        if metadata['usage_examples']:
            result += "\nExample usage:\n"
            for example in metadata['usage_examples']:
                result += f"  Input: {json.dumps(example.get('input', {}))}\n"
                result += f"  Output: {json.dumps(example.get('output', {}))}\n"
        
        return result


class ToolRepository:
    """
    Repository for managing and accessing tools.

    This class is responsible for registering, discovering, and resolving
    dependencies between tools.

    Attributes:
        tool_classes (Dict[str, Type[Tool]]): Dictionary mapping tool names to tool classes.
        enabled_tools (Set[str]): Set of names of currently enabled tools.
        gated_tools (Set[str]): Set of gated tools that self-determine availability via is_available().
        working_memory (Optional[WorkingMemory]): WorkingMemory instance for tool DI.
    """

    def __init__(self, working_memory=None):
        self.logger = logging.getLogger("tool_repository")
        self.tool_classes: Dict[str, Type[Tool]] = {}  # Store tool classes for lazy instantiation
        self.enabled_tools: Set[str] = set()
        self.gated_tools: Set[str] = set()  # Tools that self-determine availability
        self.working_memory = working_memory
        self._pinned_tools: Set[str] = set()  # Tools pinned for rest of session via load_for_rest_of_session
    
    def register_tool_class(self, tool_class: Type[Tool], tool_name: str) -> None:
        """Register a tool class for lazy instantiation."""
        if tool_name in self.tool_classes:
            self.logger.error(f"Tool registration failed: Tool with name '{tool_name}' is already registered")
            raise ValueError(f"Tool with name '{tool_name}' is already registered")
            
        self.tool_classes[tool_name] = tool_class
        self.logger.info(f"Registered tool class: {tool_name}")

    def register_gated_tool(self, tool_name: str) -> None:
        """
        Register a tool as gated - it self-determines availability via is_available().

        Gated tools automatically appear/disappear from the tool list based on their
        internal state (e.g., manifest file, user preferences). Unlike enabled_tools,
        gated tools don't require explicit enable/disable calls.

        Args:
            tool_name: Name of the tool to register as gated

        Raises:
            KeyError: If the tool class hasn't been registered yet
        """
        if tool_name not in self.tool_classes:
            raise KeyError(f"Tool '{tool_name}' must be registered before marking as gated")
        self.gated_tools.add(tool_name)
        self.logger.info(f"Registered gated tool: {tool_name}")

    def enable_tool(self, name: str) -> None:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot enable tool '{name}': Tool not found")
            raise KeyError(f"Cannot enable tool '{name}': Tool not found")

        # Gated tools use is_available(), not enable_tool()
        if name in self.gated_tools:
            raise ValueError(f"Cannot enable gated tool '{name}' - availability controlled by is_available()")

        # Auto-enable dependencies recursively
        dependencies = self.resolve_dependencies(name)
        for dep_name in dependencies:
            if dep_name not in self.enabled_tools:
                self.enable_tool(dep_name)
        
        self.enabled_tools.add(name)
        self.logger.info(f"Enabled tool: {name}")
    
    def disable_tool(self, name: str) -> None:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot disable tool '{name}': Tool not found")
            raise KeyError(f"Cannot disable tool '{name}': Tool not found")
        
        if name in self.enabled_tools:
            self.enabled_tools.remove(name)
            self.logger.info(f"Disabled tool: {name}")
        else:
            self.logger.debug(f"Tool '{name}' was already disabled")

    def cleanup_ephemeral_tools(self, essential_tools: Set[str]) -> None:
        """Disable all non-essential, non-pinned tools. Called on TurnCompletedEvent."""
        ephemeral = self.enabled_tools - essential_tools - self._pinned_tools
        if not ephemeral:
            return

        self.logger.info(f"Cleaning up {len(ephemeral)} ephemeral tools")
        self.enabled_tools -= ephemeral

    def get_tool(self, name: str) -> Tool:
        """Get tool instance, creating it lazily with current user context."""
        if name not in self.tool_classes:
            self.logger.error(f"Tool not found: {name}")
            raise KeyError(f"Tool not found: {name}")
            
        # Create new instance with current user context - no caching to prevent user data leakage
        try:
            tool_class = self.tool_classes[name]
            
            # Dependency injection: check constructor signature for known types
            dependencies = {}
            sig = inspect.signature(tool_class.__init__)
            
            for param_name, param in sig.parameters.items():
                if param_name != 'self' and param.default is inspect.Parameter.empty:
                    param_type = param.annotation

                    # Resolve forward references and Optional[...] annotations
                    annotation_name = None
                    if isinstance(param_type, str):
                        annotation_name = param_type
                    else:
                        annotation_name = getattr(param_type, '__name__', None)

                        if annotation_name is None:
                            origin = get_origin(param_type)
                            if origin is Union:
                                args = [arg for arg in get_args(param_type) if arg is not type(None)]
                                if args:
                                    candidate = args[0]
                                    if isinstance(candidate, str):
                                        annotation_name = candidate
                                    else:
                                        annotation_name = getattr(candidate, '__name__', None)

                    # Inject known dependency types
                    if annotation_name in ('LLMBridge', 'LLMProvider'):
                        from clients.llm_provider import LLMProvider
                        dependencies[param_name] = LLMProvider()
                    elif annotation_name == 'ToolRepository':
                        dependencies[param_name] = self
                    elif annotation_name == 'WorkingMemory':
                        if self.working_memory is not None:
                            dependencies[param_name] = self.working_memory
                        else:
                            self.logger.debug(
                                "Tool %s requested WorkingMemory dependency but repository has none",
                                name
                            )
            
            tool_instance = tool_class(**dependencies)
            self.logger.debug(f"Instantiated tool: {name}")
            return tool_instance
            
        except Exception as e:
            self.logger.error(f"Error instantiating tool {name}: {e}")
            raise
    
    def invoke_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot invoke tool '{name}': Tool not found")
            raise KeyError(f"Cannot invoke tool '{name}': Tool not found")

        # Check if tool is invocable: either explicitly enabled OR a gated tool that's available
        if name not in self.enabled_tools:
            if name in self.gated_tools:
                # Gated tool - check is_available() at invocation time
                tool = self.get_tool(name)
                if not (hasattr(tool, 'is_available') and tool.is_available()):
                    self.logger.error(f"Cannot invoke gated tool '{name}': Tool is not available")
                    raise RuntimeError(f"Cannot invoke gated tool '{name}': Tool is not available")
                # Gated tool is available - allow invocation to proceed
            else:
                # Auto-enable the tool on first invocation
                self.logger.info(f"Auto-enabling tool '{name}' on first invocation")
                self.enable_tool(name)

        if isinstance(params, str):
            try:
                decoded = json.loads(params)
                if isinstance(decoded, dict):
                    params = decoded
                else:
                    params = {"value": decoded}
            except json.JSONDecodeError:
                params = {"query": params}
        elif not isinstance(params, dict):
            self.logger.error(
                "Invalid parameter type for tool '%s': expected mapping, received %s",
                name,
                type(params).__name__
            )
            raise TypeError(f"Parameters for tool '{name}' must be a mapping or JSON string")

        tool = self.get_tool(name)  # This creates a fresh instance with current user context
        self.logger.debug(f"Invoking tool: {name} with params: {params}")

        # TODO: Add type coercion layer here based on tool's anthropic_schema.
        # Currently, tools receive params as-is from JSON parsing, which means numeric
        # values may arrive as strings (e.g., "10" instead of 10). Each tool handles
        # this individually with int()/float() casts (see email_tool, kasa_tool,
        # weather_tool, continuum_tool._coerce_to_int). A unified solution would:
        # 1. Read the tool's input_schema from anthropic_schema
        # 2. Coerce each param to its declared type (integer, number, boolean, array)
        # 3. Handle LLM quirks like [10] instead of 10 (single-element list unwrapping)
        # This would eliminate ad-hoc casting in every tool and centralize error handling.

        try:
            result = tool.run(**params)
            return result
        except TypeError as e:
            self.logger.error(f"Invalid parameters for tool '{name}': {str(e)}")
            raise TypeError(f"Invalid parameters for tool '{name}': {str(e)}")
    
    def list_all_tools(self) -> List[str]:
        return list(self.tool_classes.keys())
    
    def get_enabled_tools(self) -> List[str]:
        return list(self.enabled_tools)
    
    def is_tool_enabled(self, name: str) -> bool:
        return name in self.enabled_tools
    
    def get_tool_metadata(self, name: str) -> Dict[str, Any]:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot get metadata for tool '{name}': Tool not found")
            raise KeyError(f"Cannot get metadata for tool '{name}': Tool not found")
            
        # Create temporary instance to get metadata
        tool = self.get_tool(name)
        return tool.get_metadata()
    
    def get_tool_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the Anthropic schema definition for a specific tool.

        Args:
            name: The name of the tool

        Returns:
            The tool's Anthropic schema if available, None otherwise
        """
        if name not in self.tool_classes:
            self.logger.warning(f"Tool '{name}' not found in repository")
            return None

        tool = self.get_tool(name)
        if hasattr(tool, 'anthropic_schema'):
            return tool.anthropic_schema
        else:
            self.logger.warning(f"Tool '{name}' does not have an anthropic_schema attribute")
            return None

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas for LLM context - only enabled and available tools.

        invokeother_tool pattern: Essential tools are always enabled. Other tools
        loaded on-demand via invokeother_tool. Ephemeral tools cleaned up on turn end;
        pinned tools persist for the session.
        """
        definitions = []

        # Standard enabled tools (explicit enable/disable)
        for name in self.enabled_tools:
            tool = self.get_tool(name)
            if hasattr(tool, 'anthropic_schema'):
                definitions.append(tool.anthropic_schema)
            else:
                self.logger.warning(f"Tool '{name}' does not have an anthropic_schema attribute")

        # Gated tools - check is_available() at runtime
        for name in self.gated_tools:
            try:
                tool = self.get_tool(name)
                if hasattr(tool, 'is_available') and tool.is_available():
                    if hasattr(tool, 'anthropic_schema'):
                        definitions.append(tool.anthropic_schema)
                        self.logger.debug(f"Gated tool '{name}' is available")
                    else:
                        self.logger.warning(f"Gated tool '{name}' has no anthropic_schema")
            except Exception as e:
                # Gated tool check failures should not break the tool list
                self.logger.warning(f"Error checking gated tool '{name}': {e}")

        # Add code_execution tool - always available since we send the beta flag
        # This is a server-side tool that runs in Anthropic's sandbox
        definitions.insert(0, CODE_EXECUTION.copy())

        return definitions

    def resolve_dependencies(self, tool_name: str) -> List[str]:
        from utils.user_context import has_user_context

        # Dependency resolution requires tool instantiation, which requires user context.
        # During startup (no user context), return empty list - dependencies will be
        # validated when tools are first enabled in user sessions. This defers validation
        # but is acceptable since dependency failures will surface before user requests.
        if not has_user_context():
            return []
            
        visited = set()
        result = []
        
        def dfs(name):
            # Detect cycles - if we're revisiting a node, we have a circular dependency
            if name in visited:
                raise ValueError(f"Circular dependency detected involving tool '{name}'")
                
            visited.add(name)
            
            if name not in self.tool_classes:
                raise KeyError(f"Dependency '{name}' not found")
                
            tool = self.get_tool(name)
            dependencies = tool.get_dependencies()
            
            # Recursively resolve dependencies depth-first
            for dep_name in dependencies:
                if dep_name not in visited:
                    dfs(dep_name)
                    result.append(dep_name)
        
        dfs(tool_name)
        return result
    
    def discover_tools(self, package_path: str = "tools.implementations") -> None:
        self.logger.info(f"Discovering tools in package: {package_path}")

        package = importlib.import_module(package_path)

        # Iterate through all modules in the package
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            module_name = module_info.name.split('.')[-1]

            # Skip private modules and avoid processing this file
            if module_name.startswith('_') or module_name == 'repo':
                continue

            self._process_module(module_info.name)
            
    def _process_module(self, module_path: str) -> None:
        self.logger.debug(f"Importing module: {module_path}")
        module = importlib.import_module(module_path)

        # Scan module for concrete Tool subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Filter for valid Tool subclasses defined in this module
            if (inspect.isclass(attr) and
                issubclass(attr, Tool) and
                attr is not Tool and
                attr.__module__ == module.__name__ and
                not getattr(attr, '_is_abstract_base_class', False)):

                self.logger.debug(f"Found Tool subclass: {attr_name}")

                if not hasattr(attr, 'name') or not attr.name:
                    self.logger.warning(f"Skipping Tool class without name: {attr_name}")
                    continue

                # Register tool class for lazy instantiation
                # Actual dependency injection happens in get_tool() when instantiating
                self.register_tool_class(attr, attr.name)
    
    def enable_tools_from_config(self) -> None:
        """Enable essential tools at startup. Other tools loaded on-demand via invokeother_tool."""
        config = get_config()

        self.logger.info(f"Enabling essential tools at startup: {ESSENTIAL_TOOLS}")

        for name in ESSENTIAL_TOOLS:
            # Check if tool is actually enabled in its config
            tool_config = getattr(config, name, None)
            if tool_config is None:
                self.logger.warning(f"No config found for essential tool {name}, enabling anyway")
                self.enable_tool(name)
                continue

            is_enabled = getattr(tool_config, 'enabled', True)

            if is_enabled:
                self.enable_tool(name)
            else:
                self.logger.warning(
                    f"Essential tool {name} is disabled in config (enabled=false). "
                    f"This may break core functionality. Skipping."
                )
    
    def enable_all_tools(self) -> None:
        self.logger.info("Enabling all registered tools")
        
        for name in self.tool_classes:
            try:
                if name not in self.enabled_tools:
                    self.enable_tool(name)
            except Exception as e:
                self.logger.error(f"Error enabling tool {name}: {e}")
    
