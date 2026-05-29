"""Base dialect contract for provider-specific LLM transports.

A Dialect translates a provider-neutral Request into the wire format of one
provider's HTTP endpoint, executes the request, and normalizes the response
back into a Result. The class attribute native_thinking_fields declares which
thinking knobs the provider natively accepts; the dialect's _serialize_thinking
hook (or its Anthropic-specific equivalent) is responsible for translating
non-native fields and logging a TranslationNote when information is lost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from clients.llm.events import StreamEvent
from clients.llm.capabilities import Capabilities
from clients.llm.thinking import TranslationNote
from clients.llm.types import NativeField, Request, Result, Usage

if TYPE_CHECKING:  # avoid runtime import cycle
    from clients.llm.artifacts import FileArtifactSink
    from clients.llm.resolver import ModelSelection

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Base class for normalized provider transport errors."""

    def __init__(self, endpoint: str, mode: str, message: str):
        self.endpoint = endpoint
        self.mode = mode
        self.provider_message = message
        super().__init__(f"Provider {endpoint} failed during {mode}: {message}")


class ProviderAuthError(ProviderError):
    """Authentication or authorization failure from provider."""


class ProviderProtocolError(ProviderError):
    """Provider returned a malformed, incomplete, or semantically invalid response."""


class ProviderRetryableError(ProviderError):
    """Transient provider failure eligible for fallback or retry."""

    def __init__(self, endpoint: str, status_code: int | None, mode: str, message: str):
        self.status_code = status_code
        prefix = f"HTTP {status_code}: " if status_code is not None else ""
        super().__init__(endpoint, mode, f"{prefix}{message}")


class ProviderContextOverflowError(ProviderError):
    """Provider reported that the request exceeds context limits."""

    def __init__(
        self,
        endpoint: str,
        mode: str,
        message: str,
        estimated_tokens: int | None = None,
        context_window: int | None = None,
    ):
        self.estimated_tokens = estimated_tokens
        self.context_window = context_window
        super().__init__(endpoint, mode, message)


class ProviderStallError(ProviderRetryableError):
    """Provider accepted the request but produced no output."""

    def __init__(self, endpoint: str, timeout_seconds: int, mode: str):
        self.timeout_seconds = timeout_seconds
        self.status_code = None
        ProviderError.__init__(
            self,
            endpoint,
            mode,
            f"stalled for {timeout_seconds}s without producing output",
        )


class Dialect(ABC):
    """Provider-specific transport and normalization contract.

    Subclasses declare:
      - dialect_name: must be a non-base member of DialectName (validated by
        the discovery routine in clients.llm.dialect_registry).
      - capabilities: which provider features (batch, files, container reuse,
        server code execution) the dialect supports.
      - native_thinking_fields: which thinking knobs ("effort", "budget") the
        provider accepts directly. The dialect's translation logic uses native
        fields verbatim and translates non-native ones, logging a
        TranslationNote when information is lost.
      - _accepted_round_trip_fields: which provider-specific metadata fields
        (e.g., OpenRouter's reasoning_details, Anthropic's thinking_signatures)
        this dialect copies onto the wire. Anything not in the union of
        _STANDARD_MESSAGE_FIELDS and _accepted_round_trip_fields is stripped by
        sanitize_outbound_message before conversion code sees it.

    Subclasses implement:
      - from_selection: construct an instance from a resolved ModelSelection,
        api_key, timeout, and artifact_sink. The orchestrator never
        knows which kwargs each dialect needs.
      - complete / stream: the actual provider interaction.
    """

    # Keys that are part of the standard neutral message format. These pass
    # through sanitize_outbound_message unconditionally. Only grows when the
    # core message model gains a new universal field.
    _STANDARD_MESSAGE_FIELDS: frozenset[str] = frozenset({
        "role", "content", "tool_calls", "tool_call_id", "name", "is_error",
    })

    # Provider-specific metadata fields this dialect accepts on the wire.
    # Each concrete dialect declares only the fields its provider understands;
    # everything else is stripped by sanitize_outbound_message.
    _accepted_round_trip_fields: frozenset[str] = frozenset()

    dialect_name: str = "base"
    capabilities: Capabilities = Capabilities()
    # Thinking knobs this dialect natively accepts. Empty tuple means the
    # dialect does not expose any thinking knob (model runs in default mode
    # regardless of caller's ThinkingConfig).
    native_thinking_fields: tuple[NativeField, ...] = ()
    # Marker for the registry: subclasses that are genuine transports override
    # to True so the registry can skip base/mixin classes without string matching.
    is_abstract: bool = True

    @classmethod
    @abstractmethod
    def from_selection(
        cls,
        selection: "ModelSelection",
        *,
        api_key: str | None,
        timeout: int,
        artifact_sink: "FileArtifactSink | None",
    ) -> "Dialect":
        """Construct a dialect instance from a resolved ModelSelection.

        Implementations source the credential from the api_key argument or
        from selection.api_key_name via Vault. Failure modes (missing
        endpoint, missing key) raise loudly here so the orchestrator never
        sees a partially-constructed dialect.
        """

    @abstractmethod
    def complete(self, request: Request) -> Result:
        """Execute a non-streaming provider request."""

    @abstractmethod
    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """Execute a streaming provider request and yield normalized events."""

    def current_partial_usage(self) -> Usage | None:
        """Best-effort snapshot of accumulated provider usage during a live stream.

        Dialects whose underlying SDK exposes mid-stream usage override this;
        the default returns None. Callers must treat None as "usage unavailable"
        and skip billing rather than substitute fabricated numbers.
        """
        return None

    def sanitize_outbound_message(self, message: dict) -> dict:
        """Strip provider-specific metadata this dialect doesn't accept.

        Returns a copy of *message* with only keys in the union of
        _STANDARD_MESSAGE_FIELDS and _accepted_round_trip_fields retained.
        Called in each dialect's message-assembly method before conversion
        to wire format, so conversion code never sees foreign metadata.
        """
        allowed = self._STANDARD_MESSAGE_FIELDS | self._accepted_round_trip_fields
        return {k: v for k, v in message.items() if k in allowed}

    def _log_translation(self, note: TranslationNote) -> None:
        """Emit a translation note at WARNING level.

        Dialects call this for any thinking translation that drops or
        reinterprets caller intent — e.g., a budget_tokens hint mapped to
        effort, or an effort value clamped to a model's documented maximum.
        """
        logger.warning(
            "Dialect '%s' translated %s: requested=%r applied=%r (%s)",
            self.dialect_name, note.field, note.requested, note.applied, note.reason,
        )
