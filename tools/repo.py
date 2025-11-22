"""
Tool repository for the botwithmemory system.

This module provides the base Tool class and ToolRepository for managing, 
discovering, and using tools within the continuum system.
"""
import inspect
import importlib
import json
import logging
import os
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Type, Callable, Union, get_args, get_origin
from pathlib import Path

from pydantic import BaseModel, create_model
from utils.user_context import get_current_user_id
from utils.userdata_manager import get_user_data_manager

from tools.registry import registry


def get_config():
    from config import config
    return config


class Tool(ABC):
    """
    Base class for all tools in the botwithmemory system.
    
    This class defines the standard interface and behavior that all tools
    should implement. It includes metadata, parameter handling, and execution logic.
    
    Class Attributes:
        name (str): The unique name of the tool.
        description (str): A human-readable description of the tool's purpose.
        usage_examples (List[Dict]): Example usage of the tool.
    """
    
    name = "base_tool"
    description = "Base class for all tools"
    usage_examples: List[Dict[str, Any]] = []

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
    dependencies between tools. Tool hints are published to working memory
    via events when tools are enabled/disabled.

    Attributes:
        tool_classes (Dict[str, Type[Tool]]): Dictionary mapping tool names to tool classes.
        enabled_tools (Set[str]): Set of names of currently enabled tools.
        working_memory (Optional[WorkingMemory]): WorkingMemory instance for publishing tool updates.
    """

    def __init__(self, working_memory=None):
        self.logger = logging.getLogger("tool_repository")
        self.tool_classes: Dict[str, Type[Tool]] = {}  # Store tool classes for lazy instantiation
        self.enabled_tools: Set[str] = set()
        self.working_memory = working_memory
        config = get_config()
        self.tool_list_path: str = os.path.join(config.paths.data_dir, "tools", "tool_list.json")
    
    def register_tool_class(self, tool_class: Type[Tool], tool_name: str) -> None:
        """Register a tool class for lazy instantiation."""
        if tool_name in self.tool_classes:
            self.logger.error(f"Tool registration failed: Tool with name '{tool_name}' is already registered")
            raise ValueError(f"Tool with name '{tool_name}' is already registered")
            
        self.tool_classes[tool_name] = tool_class
        self.logger.info(f"Registered tool class: {tool_name}")
        
        # Ensure tool has dedicated directory for persistent data storage
        config = get_config()
        tool_data_dir = os.path.join(config.paths.data_dir, "tools", tool_name)
        try:
            os.makedirs(tool_data_dir, exist_ok=True)
            self.logger.debug(f"Created or verified tool data directory: {tool_data_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to create tool data directory for {tool_name}: {e}")
        
        self._update_tool_list_file()
        self._update_tool_guidance()
    
    def enable_tool(self, name: str) -> None:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot enable tool '{name}': Tool not found")
            raise KeyError(f"Cannot enable tool '{name}': Tool not found")
        
        # Auto-enable dependencies recursively
        dependencies = self.resolve_dependencies(name)
        for dep_name in dependencies:
            if dep_name not in self.enabled_tools:
                self.enable_tool(dep_name)
        
        self.enabled_tools.add(name)
        self.logger.info(f"Enabled tool: {name}")
        self._update_tool_guidance()
    
    def disable_tool(self, name: str) -> None:
        if name not in self.tool_classes:
            self.logger.error(f"Cannot disable tool '{name}': Tool not found")
            raise KeyError(f"Cannot disable tool '{name}': Tool not found")
        
        if name in self.enabled_tools:
            self.enabled_tools.remove(name)
            self.logger.info(f"Disabled tool: {name}")
            self._update_tool_guidance()
        else:
            self.logger.debug(f"Tool '{name}' was already disabled")
    
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

        if name not in self.enabled_tools:
            self.logger.error(f"Cannot invoke tool '{name}': Tool is not enabled")
            raise RuntimeError(f"Cannot invoke tool '{name}': Tool is not enabled")

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

        try:
            result = tool.run(**params)

            # Notify ToolLoaderTrinket about tool usage for idle tracking
            if self.working_memory and name != "invokeother_tool":
                self.working_memory.publish_trinket_update(
                    target_trinket="ToolLoaderTrinket",
                    context={
                        "action": "tool_used",
                        "tool_name": name
                    }
                )

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
        definitions = []

        for name in self.enabled_tools:
            tool = self.get_tool(name)
            if hasattr(tool, 'anthropic_schema'):
                definitions.append(tool.anthropic_schema)
            else:
                self.logger.warning(f"Tool '{name}' does not have an anthropic_schema attribute")

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
        """Enable only essential tools at startup. Other tools loaded on-demand via invokeother_tool."""
        config = get_config()
        essential_tools = config.tools.essential_tools

        self.logger.info(f"Enabling essential tools at startup: {essential_tools}")

        for name in essential_tools:
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
    
    def update_working_memory(self) -> None:
        self._update_tool_guidance()

    def _update_tool_guidance(self) -> None:
        if not self.working_memory:
            return

        enabled_tools = self.get_enabled_tools()
        
        # Collect hints from enabled tools
        tool_hints = {}
        for tool_name in enabled_tools:
            try:
                tool_instance = self.get_tool(tool_name)
                if tool_instance:
                    # Get tool hints if available (gracefully handle missing attribute)
                    hint = getattr(tool_instance, 'tool_hints', None)
                    if hint:
                        tool_hints[tool_name] = hint
            except Exception as e:
                self.logger.warning(f"Could not get hints for tool {tool_name}: {e}")
        
        # Send hints to ToolGuidanceTrinket
        context = {'tool_hints': tool_hints}
        
        self.working_memory.publish_trinket_update(
            target_trinket="ToolGuidanceTrinket",
            context=context
        )

        self.logger.debug(f"Sent {len(tool_hints)} tool hints to ToolGuidanceTrinket")

    def _update_tool_list_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.tool_list_path), exist_ok=True)
            
            # Build comprehensive tool registry for external inspection/debugging
            tool_list = []
            
            for name, tool_class in self.tool_classes.items():
                try:
                    # Create basic metadata from class info without instantiation if no user context
                    from utils.user_context import has_user_context
                    
                    if has_user_context():
                        # If user context exists, get full metadata from instance
                        tool = self.get_tool(name)
                        metadata = tool.get_metadata()
                        dependencies = tool.get_dependencies()
                    else:
                        # Fallback: create basic metadata from class without user context
                        metadata = {
                            "name": name,
                            "description": getattr(tool_class, 'description', 'No description available'),
                            "parameters": {},
                            "required_parameters": []
                        }
                        dependencies = []
                        
                    tool_list.append({
                        "name": name,
                        "description": metadata["description"],
                        "parameters": metadata["parameters"],
                        "required_parameters": metadata["required_parameters"],
                        "dependencies": dependencies
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting metadata for tool {name}: {e}")
                    # Add basic entry so tool isn't completely missing from list
                    tool_list.append({
                        "name": name,
                        "description": getattr(tool_class, 'description', 'Error loading tool'),
                        "parameters": {},
                        "required_parameters": [],
                        "dependencies": []
                    })
            
            with open(self.tool_list_path, 'w') as f:
                json.dump(tool_list, indent=2, sort_keys=True, default=str, fp=f)
                
            self.logger.debug(f"Updated tool list file: {self.tool_list_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating tool list file: {e}")
