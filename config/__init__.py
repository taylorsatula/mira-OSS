"""
Centralized configuration management package.

This package provides a single configuration interface that loads settings
from various sources, validates them, and makes them available throughout
the application.

Usage:
    from config import config
    
    # Access using attribute notation
    max_tokens = config.api.max_tokens

    # Or using get() method with dot notation
    max_tokens = config.get("api.max_tokens")
    
    # For required values (raises exception if missing)
    api_key = config.require("api.key")
    
    # For tool configurations
    enabled = config.web_tool.enabled
"""

# First, initialize the registry (which has no dependencies)
from tools.registry import registry

# Then, import the configuration system
from config.config_manager import AppConfig, config

# Export the public interface
__all__ = ["config", "AppConfig", "registry"]
