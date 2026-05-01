"""
Stream event types for LLM provider streaming.

Provides a clean, type-safe event hierarchy for streaming responses
through the LLM pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic.types


class GenerationCancelled(Exception):
    """Raised when the user cancels generation midstream."""
    pass


@dataclass
class StreamEvent:
    """Base event for all streaming events."""
    type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TextEvent(StreamEvent):
    """Text content chunk from LLM."""
    content: str
    type: str = field(default="text", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ThinkingEvent(StreamEvent):
    """Thinking content chunk from LLM with extended thinking enabled."""
    content: str
    type: str = field(default="thinking", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolDetectedEvent(StreamEvent):
    """Tool detected in LLM response."""
    tool_name: str
    tool_id: str
    type: str = field(default="tool_detected", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolExecutingEvent(StreamEvent):
    """Tool execution started."""
    tool_name: str
    tool_id: str
    arguments: dict[str, object]
    type: str = field(default="tool_executing", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolCompletedEvent(StreamEvent):
    """Tool execution completed successfully."""
    tool_name: str
    tool_id: str
    result: str | list[dict[str, object]]
    type: str = field(default="tool_completed", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ToolErrorEvent(StreamEvent):
    """Tool execution failed."""
    tool_name: str
    tool_id: str
    error: str
    type: str = field(default="tool_error", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class CompleteEvent(StreamEvent):
    """Stream completed with final response."""
    response: anthropic.types.Message
    type: str = field(default="complete", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ErrorEvent(StreamEvent):
    """Stream error occurred."""
    error: str
    type: str = field(default="error", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class CircuitBreakerEvent(StreamEvent):
    """Circuit breaker triggered during tool execution."""
    reason: str
    type: str = field(default="circuit_breaker", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class RetryEvent(StreamEvent):
    """Retry attempt for malformed tool calls."""
    attempt: int
    max_attempts: int
    reason: str
    type: str = field(default="retry", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class FileArtifactEvent(StreamEvent):
    """File artifact produced by code execution."""
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    type: str = field(default="file_artifact", init=False)
    timestamp: float = field(default_factory=time.time, init=False)


@dataclass
class ProviderSwitchEvent(StreamEvent):
    """Generic provider stalled, switching to backup."""
    original_endpoint: str
    backup_model: str
    reason: str
    type: str = field(default="provider_switch", init=False)
    timestamp: float = field(default_factory=time.time, init=False)
