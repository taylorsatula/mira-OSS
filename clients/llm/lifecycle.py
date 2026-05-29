"""Shared live request lifecycle for LLM calls."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from config import config
from clients.llm.events import (
    CompleteEvent,
    GenerationCancelled,
    ProviderSwitchEvent,
    StreamEvent,
    TextEvent,
)
from clients.llm.accounting import UsageAccountingPolicy, UsageAccountingService
from clients.llm.dialects.base import (
    Dialect,
    ProviderProtocolError,
    ProviderRetryableError,
    ProviderStallError,
)
from clients.llm.dialects.openai_chat_base import ToolNotLoadedError
from clients.llm.types import ProviderMetadata, Request, Result, ToolCall, Usage
from utils.user_context import check_cancelled, get_cancel_event

logger = logging.getLogger(__name__)


class LLMLifecycle:
    """One provider request lifecycle for completion, streaming, fallback, and billing."""

    def __init__(
        self,
        *,
        response_timeout_seconds: int | None = None,
        accounting_policy: UsageAccountingPolicy | None = None,
        accounting_service: UsageAccountingService | None = None,
    ) -> None:
        self.response_timeout_seconds = (
            response_timeout_seconds
            if response_timeout_seconds is not None
            else config.api.provider_response_timeout
        )
        self.accounting_policy = accounting_policy
        self.accounting_service = accounting_service or UsageAccountingService()

    def complete(
        self,
        request: Request,
        dialect: Dialect,
        *,
        fallback_factory: Callable[[], tuple[Dialect, Request]] | None = None,
    ) -> Result:
        final: Result | None = None
        for event in self._run(
            request,
            dialect,
            stream=False,
            fallback_factory=fallback_factory,
        ):
            if isinstance(event, CompleteEvent):
                final = event.response
        if final is None:
            raise ProviderProtocolError(
                self._endpoint(dialect),
                "non-streaming",
                "Lifecycle ended without completion event",
            )
        return final

    def stream(
        self,
        request: Request,
        dialect: Dialect,
        *,
        fallback_factory: Callable[[], tuple[Dialect, Request]] | None = None,
    ) -> Iterator[StreamEvent]:
        yield from self._run(
            request,
            dialect,
            stream=True,
            fallback_factory=fallback_factory,
        )

    def _run(
        self,
        request: Request,
        dialect: Dialect,
        *,
        stream: bool,
        fallback_factory: Callable[[], tuple[Dialect, Request]] | None,
    ) -> Iterator[StreamEvent]:
        active_dialect = dialect
        active_request = request
        used_fallback = False
        streamed_output_chars = 0

        while True:
            try:
                check_cancelled()
                result: Result | None = None
                events = (
                    active_dialect.stream(active_request)
                    if stream
                    else self._complete_as_events(active_dialect, active_request)
                )
                event_iterator = iter(events)
                while True:
                    try:
                        event = self._run_with_response_timeout(
                            lambda: next(event_iterator),
                            endpoint=self._endpoint(active_dialect),
                            mode="streaming" if stream else "non-streaming",
                        )
                    except StopIteration:
                        break

                    cancel_evt = get_cancel_event()
                    if cancel_evt is not None and cancel_evt.is_set():
                        self._bill_partial_on_cancel(
                            active_dialect,
                            active_request,
                            streamed_output_chars,
                        )
                        raise GenerationCancelled()

                    if isinstance(event, CompleteEvent):
                        result = event.response
                    else:
                        if isinstance(event, TextEvent):
                            streamed_output_chars += len(event.content)
                        yield event

                if result is None:
                    raise ProviderProtocolError(
                        self._endpoint(active_dialect),
                        "streaming" if stream else "non-streaming",
                        "Provider ended without a complete response",
                    )

                final_result = self._with_transport_metadata(
                    result,
                    active_dialect,
                    active_request,
                )
                if self.accounting_policy is not None:
                    self.accounting_service.finalize(final_result, self.accounting_policy)
                yield CompleteEvent(response=final_result)
                return

            except ToolNotLoadedError as error:
                # Provider attempted to call a tool that was not in the request.
                # Recover by synthesizing an invokeother_tool call so the orchestrator's
                # tool loop loads the tool and re-invokes the provider with it available.
                logger.info(
                    "Provider %s called unloaded tool '%s'; recovering via invokeother_tool",
                    self._endpoint(active_dialect),
                    error.tool_name,
                )
                synthetic = Result(
                    text="",
                    tool_calls=(
                        ToolCall(
                            id=f"toolu_{uuid.uuid4().hex[:24]}",
                            tool_name="invokeother_tool",
                            input={"mode": "load", "query": error.tool_name},
                        ),
                    ),
                    reasoning=None,
                    usage=None,
                    stop_reason="tool_use",
                    provider_metadata=ProviderMetadata(
                        dialect_name=active_dialect.dialect_name,
                        model=active_request.model,
                        endpoint_url=active_request.metadata.endpoint_url,
                        internal_llm_name=active_request.metadata.internal_llm_name,
                        conversation_llm_name=active_request.metadata.conversation_llm_name,
                    ),
                )
                yield CompleteEvent(response=synthetic)
                return

            except (ProviderRetryableError, ProviderStallError) as error:
                if used_fallback:
                    raise
                if fallback_factory is None:
                    raise
                fallback_dialect, fallback_request = fallback_factory()
                used_fallback = True
                yield ProviderSwitchEvent(
                    original_endpoint=error.endpoint,
                    backup_model=fallback_request.model,
                    reason=str(error),
                )
                active_dialect = fallback_dialect
                active_request = fallback_request

    def _complete_as_events(self, dialect: Dialect, request: Request) -> Iterator[CompleteEvent]:
        yield CompleteEvent(response=dialect.complete(request))

    def _run_with_response_timeout(
        self,
        operation: Callable[[], Any],
        *,
        endpoint: str,
        mode: str,
    ) -> Any:
        result: list[Any] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(operation())
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=self.response_timeout_seconds)
        if worker.is_alive():
            raise ProviderStallError(endpoint, self.response_timeout_seconds, mode)
        if errors:
            raise errors[0]
        return result[0]

    def _endpoint(self, dialect: Dialect) -> str:
        return getattr(dialect, "endpoint_url", dialect.dialect_name)

    def _bill_partial_on_cancel(
        self,
        dialect: Dialect,
        request: Request,
        streamed_output_chars: int,
    ) -> None:
        """Best-effort partial billing when the user cancels mid-stream."""
        if self.accounting_policy is None:
            return
        partial = dialect.current_partial_usage()
        if partial is None:
            return
        estimated_output = max(streamed_output_chars // 4, partial.output_tokens, 1)
        usage = Usage(
            input_tokens=partial.input_tokens,
            output_tokens=estimated_output,
            cache_creation_input_tokens=partial.cache_creation_input_tokens,
            cache_read_input_tokens=partial.cache_read_input_tokens,
        )
        partial_result = Result(
            text="",
            tool_calls=(),
            reasoning=None,
            usage=usage,
            stop_reason="end_turn",
            provider_metadata=ProviderMetadata(
                dialect_name=dialect.dialect_name,
                model=request.model,
                endpoint_url=request.metadata.endpoint_url,
                internal_llm_name=request.metadata.internal_llm_name,
                conversation_llm_name=request.metadata.conversation_llm_name,
            ),
        )
        try:
            self.accounting_service.finalize(partial_result, self.accounting_policy)
        except Exception as error:
            logger.warning(
                "Cancel-time billing failed (non-fatal): %s",
                error,
                exc_info=True,
            )

    def _with_transport_metadata(
        self,
        result: Result,
        dialect: Dialect,
        request: Request,
    ) -> Result:
        endpoint_url = request.metadata.endpoint_url
        if endpoint_url is None and hasattr(dialect, "endpoint_url"):
            endpoint_url = getattr(dialect, "endpoint_url")
        return result.with_provider_metadata(
            ProviderMetadata(
                dialect_name=dialect.dialect_name,
                model=request.model,
                endpoint_url=endpoint_url,
                internal_llm_name=request.metadata.internal_llm_name,
                conversation_llm_name=request.metadata.conversation_llm_name,
            )
        )
