"""Provider-neutral LLM request and result types."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, cast, get_args

DialectName = Literal["anthropic", "openai", "openrouter", "groq"]
NativeField = Literal["effort", "budget"]
CacheTTL = Literal["5m", "1h"]
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "pause_turn",
    "refusal",
    "error",
]
STOP_REASONS = set(get_args(StopReason))
DIALECT_NAMES = set(get_args(DialectName))
NATIVE_FIELDS = set(get_args(NativeField))
EFFORT_LEVEL_ORDER = cast(tuple[EffortLevel, ...], get_args(EffortLevel))
EFFORT_LEVELS = set(EFFORT_LEVEL_ORDER)


def coerce_stop_reason(value: object) -> StopReason:
    """Validate provider stop reason before it crosses the neutral boundary."""
    if isinstance(value, str) and value in STOP_REASONS:
        return cast(StopReason, value)
    raise ValueError(f"Unknown stop_reason: {value}")


def coerce_dialect_name(value: object) -> DialectName:
    """Validate configured dialect names at the typed boundary."""
    if isinstance(value, str) and value in DIALECT_NAMES:
        return cast(DialectName, value)
    raise ValueError(f"Unknown LLM dialect_name: {value}")


def coerce_effort(value: object) -> EffortLevel:
    """Validate configured reasoning effort before dialect serialization."""
    if isinstance(value, str) and value in EFFORT_LEVELS:
        return cast(EffortLevel, value)
    raise ValueError(f"Unknown LLM effort: {value}")


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _freeze_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"{field_name} must be a mapping")
    if not isinstance(value, MappingABC):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


def _copy_mapping_sequence(value: Sequence[Mapping[str, Any]], field_name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of mappings")
    copied: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, MappingABC):
            raise ValueError(f"{field_name}[{index}] must be a mapping")
        copied.append(dict(item))
    return tuple(copied)


@dataclass(frozen=True)
class ThinkingConfig:
    """Caller's thinking/effort intent for one LLM request.

    Permissive surface: both effort and budget_tokens may be set simultaneously.
    Each dialect translates the combination to its native shape, clamping or
    discarding fields as appropriate, and logs a TranslationNote when information
    is lost. Callers do not need to know which knob their selected provider
    supports.
    """

    effort: EffortLevel | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.effort is not None:
            object.__setattr__(self, "effort", coerce_effort(self.effort))
        if self.budget_tokens is not None and (
            type(self.budget_tokens) is not int or self.budget_tokens <= 0
        ):
            raise ValueError("budget_tokens must be a positive integer")

    @property
    def active(self) -> bool:
        return self.effort is not None or self.budget_tokens is not None


@dataclass(frozen=True)
class Usage:
    """Normalized token usage for a provider response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Usage.{field_name} must be a non-negative integer")

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            raise TypeError("Usage can only be added to Usage")
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


@dataclass(frozen=True)
class RequestMetadata:
    """Provider-neutral request provenance used by lifecycle policy."""

    endpoint_url: str | None = None
    internal_llm_name: str | None = None
    conversation_llm_name: str | None = None


@dataclass(frozen=True)
class ProviderMetadata:
    """Provider-neutral response provenance used by billing and diagnostics."""

    dialect_name: str | None = None
    model: str | None = None
    endpoint_url: str | None = None
    internal_llm_name: str | None = None
    conversation_llm_name: str | None = None

    def merged(self, override: "ProviderMetadata") -> "ProviderMetadata":
        return ProviderMetadata(
            dialect_name=(
                override.dialect_name
                if override.dialect_name is not None
                else self.dialect_name
            ),
            model=override.model if override.model is not None else self.model,
            endpoint_url=(
                override.endpoint_url
                if override.endpoint_url is not None
                else self.endpoint_url
            ),
            internal_llm_name=(
                override.internal_llm_name
                if override.internal_llm_name is not None
                else self.internal_llm_name
            ),
            conversation_llm_name=(
                override.conversation_llm_name
                if override.conversation_llm_name is not None
                else self.conversation_llm_name
            ),
        )


@dataclass(frozen=True)
class ToolDefinition:
    """MIRA-owned tool schema contract."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    cache: CacheTTL | None = None  # Dialect translates to provider cache_control

    def __post_init__(self) -> None:
        name = _require_non_empty_str(self.name, "ToolDefinition.name")
        description = _require_non_empty_str(
            self.description,
            f"ToolDefinition '{name}' description",
        )
        input_schema = _freeze_mapping(self.input_schema, f"ToolDefinition '{name}' input_schema")
        if input_schema:
            schema_type = input_schema.get("type")
            if schema_type != "object":
                raise ValueError(f"ToolDefinition '{name}' input_schema.type must be 'object'")
            properties = input_schema.get("properties", {})
            if not isinstance(properties, MappingABC):
                raise ValueError(f"ToolDefinition '{name}' input_schema.properties must be a mapping")
            required = input_schema.get("required", [])
            if not isinstance(required, SequenceABC) or isinstance(required, (str, bytes, bytearray)):
                raise ValueError(f"ToolDefinition '{name}' input_schema.required must be a list of strings")
            invalid_required = [field for field in required if not isinstance(field, str) or not field]
            if invalid_required:
                raise ValueError(f"ToolDefinition '{name}' input_schema.required contains invalid field names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(
            self,
            "provider_options",
            _freeze_mapping(self.provider_options, f"ToolDefinition '{name}' provider_options"),
        )

    # Keys recognized by ToolDefinition; everything else is an error.
    _KNOWN_KEYS = frozenset({"name", "description", "input_schema", "cache", "provider_options"})

    @classmethod
    def from_mapping(cls, schema: Mapping[str, Any]) -> "ToolDefinition":
        if not isinstance(schema, MappingABC):
            raise ValueError("ToolDefinition schema must be a mapping")

        unknown_keys = set(schema.keys()) - cls._KNOWN_KEYS
        if unknown_keys:
            err_msg = (
                f"ToolDefinition has unrecognized key(s): {sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(cls._KNOWN_KEYS)}."
            )
            if sorted(unknown_keys)[0] in {"descripton", "descripion", "desciption"}:
                err_msg += f" Did you mean 'description' (not '{sorted(unknown_keys)[0]}'?)"
            raise ValueError(err_msg)

        name = _require_non_empty_str(schema.get("name"), "ToolDefinition.name")
        if "description" not in schema:
            raise ValueError(f"ToolDefinition '{name}' requires description")
        description = _require_non_empty_str(
            schema.get("description"),
            f"ToolDefinition '{name}' description",
        )
        input_schema = schema["input_schema"] if "input_schema" in schema else {}
        cache = schema.get("cache")
        provider_options = schema.get("provider_options", {})
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            provider_options=provider_options,
            cache=cache,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            **dict(self.provider_options),
        }
        if self.cache is not None:
            result["cache"] = self.cache
        return result


@dataclass(frozen=True)
class ToolCall:
    """Provider-normalized local tool call."""

    id: str
    tool_name: str
    input: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty_str(self.id, "ToolCall.id"))
        object.__setattr__(
            self,
            "tool_name",
            _require_non_empty_str(self.tool_name, "ToolCall.tool_name"),
        )
        object.__setattr__(self, "input", _freeze_mapping(self.input, "ToolCall.input"))

    def to_message_block(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "id": self.id,
            "name": self.tool_name,
            "input": dict(self.input),
        }


@dataclass(frozen=True)
class ToolResult:
    """Result returned by local tool execution."""

    tool_call_id: str
    content: str | list[dict[str, Any]]
    is_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_call_id",
            _require_non_empty_str(self.tool_call_id, "ToolResult.tool_call_id"),
        )
        if not isinstance(self.content, str) and not isinstance(self.content, list):
            raise ValueError("ToolResult.content must be a string or list of content blocks")
        if type(self.is_error) is not bool:
            raise ValueError("ToolResult.is_error must be a bool")


@dataclass(frozen=True)
class ReasoningEntry:
    """A single thinking block from the API response."""

    text: str
    signature: str | None = None  # Anthropic-specific; None for providers that don't sign

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("ReasoningEntry.text must be a string")
        if self.signature is not None:
            object.__setattr__(
                self,
                "signature",
                _require_non_empty_str(self.signature, "ReasoningEntry.signature"),
            )


@dataclass(frozen=True)
class ReasoningArtifact:
    """Internal reasoning artifact for display and provider round-trip."""

    entries: tuple[ReasoningEntry, ...] = ()
    redacted_data: tuple[str, ...] = ()
    provider_details: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "redacted_data", tuple(self.redacted_data))
        if self.provider_details is None:
            provider_details: tuple[dict[str, Any], ...] = ()
        else:
            if not isinstance(self.provider_details, SequenceABC) or isinstance(self.provider_details, (str, bytes, bytearray)):
                raise ValueError("provider_details must be a sequence of mappings")
            details: list[dict[str, Any]] = []
            for index, item in enumerate(self.provider_details):
                if not isinstance(item, MappingABC):
                    raise ValueError(f"provider_details[{index}] must be a mapping")
                details.append(dict(item))
            provider_details = tuple(details)
        object.__setattr__(self, "provider_details", provider_details)

    @property
    def text(self) -> str:
        """Combined reasoning text for backward-compatible consumers."""
        return "\n\n".join(e.text for e in self.entries)

    def to_content_blocks(self) -> list[dict[str, Any]]:
        """Produce ReasoningBlocks for content."""
        return [{"type": "reasoning", "text": e.text} for e in self.entries if e.text]

    def to_signatures(self) -> list[dict[str, Any]]:
        """Produce ordered signature/redacted blocks for metadata.
        Only meaningful to dialects that need round-trip verification."""
        blocks: list[dict[str, Any]] = []
        for entry in self.entries:
            if entry.signature is not None:
                blocks.append({"type": "thinking", "signature": entry.signature})
        blocks.extend({"type": "redacted_thinking", "data": d} for d in self.redacted_data)
        return blocks


@dataclass(frozen=True)
class Request:
    """Provider-neutral live LLM request."""

    messages: Sequence[Mapping[str, Any]]
    model: str
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float | None = None
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    container_id: str | None = None
    metadata: RequestMetadata = field(default_factory=RequestMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _copy_mapping_sequence(self.messages, "Request.messages"))
        object.__setattr__(self, "model", _require_non_empty_str(self.model, "Request.model"))
        if self.system is not None:
            if isinstance(self.system, str):
                object.__setattr__(self, "system", _require_non_empty_str(self.system, "Request.system"))
            elif isinstance(self.system, list):
                if not self.system:
                    raise ValueError("Request.system content block list must not be empty")
                object.__setattr__(self, "system", list(_copy_mapping_sequence(self.system, "Request.system")))
            else:
                raise ValueError("Request.system must be a string, list of content blocks, or None")
        tools = tuple(self.tools)
        for index, tool in enumerate(tools):
            if not isinstance(tool, ToolDefinition):
                raise ValueError(f"Request.tools[{index}] must be a ToolDefinition")
        object.__setattr__(self, "tools", tools)
        # type() not isinstance — bool subclasses int
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("Request.max_tokens must be a positive integer")
        object.__setattr__(self, "max_tokens", self.max_tokens)
        if self.temperature is not None and (
            isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
        ):
            raise ValueError("Request.temperature must be numeric when provided")
        if not isinstance(self.thinking, ThinkingConfig):
            raise ValueError("Request.thinking must be a ThinkingConfig")
        if self.container_id is not None:
            object.__setattr__(
                self,
                "container_id",
                _require_non_empty_str(self.container_id, "Request.container_id"),
            )
        if not isinstance(self.metadata, RequestMetadata):
            raise ValueError("Request.metadata must be RequestMetadata")

    def with_messages(self, messages: Sequence[Mapping[str, Any]]) -> "Request":
        return Request(
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            thinking=self.thinking,
            container_id=self.container_id,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class Result:
    """Flat semantic LLM result. No provider content-block ordering lives here."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: ReasoningArtifact | None = None
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    provider_metadata: ProviderMetadata = field(default_factory=ProviderMetadata)
    container_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("Result.text must be a string")
        object.__setattr__(self, "stop_reason", coerce_stop_reason(self.stop_reason))
        tool_calls = tuple(self.tool_calls)
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, ToolCall):
                raise ValueError(f"Result.tool_calls[{index}] must be a ToolCall")
        object.__setattr__(self, "tool_calls", tool_calls)
        if self.reasoning is not None and not isinstance(self.reasoning, ReasoningArtifact):
            raise ValueError("Result.reasoning must be a ReasoningArtifact")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise ValueError("Result.usage must be Usage")
        if not isinstance(self.provider_metadata, ProviderMetadata):
            raise ValueError("Result.provider_metadata must be ProviderMetadata")
        if self.container_id is not None:
            object.__setattr__(
                self,
                "container_id",
                _require_non_empty_str(self.container_id, "Result.container_id"),
            )

    def with_usage(self, usage: Usage | None) -> "Result":
        return Result(
            text=self.text,
            tool_calls=self.tool_calls,
            reasoning=self.reasoning,
            usage=usage,
            stop_reason=self.stop_reason,
            provider_metadata=self.provider_metadata,
            container_id=self.container_id,
        )

    def with_provider_metadata(self, metadata: ProviderMetadata) -> "Result":
        return Result(
            text=self.text,
            tool_calls=self.tool_calls,
            reasoning=self.reasoning,
            usage=self.usage,
            stop_reason=self.stop_reason,
            provider_metadata=self.provider_metadata.merged(metadata),
            container_id=self.container_id,
        )
