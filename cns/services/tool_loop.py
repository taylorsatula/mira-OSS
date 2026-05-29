"""Local tool execution support for orchestrator-owned LLM tool loops."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
from collections.abc import Generator, Iterator
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from clients.llm.events import StreamEvent, ToolCompletedEvent, ToolErrorEvent, ToolExecutingEvent
from clients.llm.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Record of a local tool execution used for loop detection."""

    tool_name: str
    result_hash: str | None
    error: Exception | None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Result of invoking one local tool call."""

    tool_call: ToolCall
    result_content: str | list[dict[str, Any]]
    raw_result: Any
    hash_material: Any
    error: Exception | None


@dataclass
class CircuitBreaker:
    """Stops local tool chains on repeated failures or identical loops."""

    tool_results: list[ToolExecution] = field(default_factory=list)

    def record_execution(self, tool_name: str, result: Any, error: Exception | None = None) -> None:
        serialized_result = json.dumps(result, sort_keys=True, default=str) if error is None else ""
        self.tool_results.append(
            ToolExecution(
                tool_name=tool_name,
                result_hash=None if error else hashlib.sha256(serialized_result.encode()).hexdigest(),
                error=error,
            )
        )

    def should_continue(self) -> tuple[bool, str]:
        if not self.tool_results:
            return True, "First tool"
        last = self.tool_results[-1]
        if last.error is not None:
            prior_errors = sum(
                1
                for execution in self.tool_results[:-1]
                if execution.tool_name == last.tool_name and execution.error is not None
            )
            if prior_errors > 0:
                return False, f"Tool '{last.tool_name}' failed after correction attempt: {last.error}"
        if len(self.tool_results) >= 2:
            current = self.tool_results[-1]
            previous = self.tool_results[-2]
            if (
                current.tool_name == previous.tool_name
                and current.result_hash == previous.result_hash
                and current.result_hash is not None
            ):
                return False, "Repeated identical results"
        return True, "Continue"


class ToolLoopExecutor:
    """Executes local tool calls for the orchestrator's model-turn loop."""

    def __init__(self, tool_repo: Any) -> None:
        self.tool_repo = tool_repo

    def execute_tools(
        self,
        tool_calls: list[ToolCall],
        breaker: CircuitBreaker,
    ) -> Generator[StreamEvent, None, tuple[ToolResult, ...]]:
        sequential = []
        parallel = []
        for tool_call in tool_calls:
            tool_class = self.tool_repo.tool_classes.get(tool_call.tool_name)
            if tool_class and not tool_class.is_call_parallel_safe(dict(tool_call.input)):
                sequential.append(tool_call)
            else:
                parallel.append(tool_call)

        for tool_call in tool_calls:
            yield ToolExecutingEvent(
                tool_name=tool_call.tool_name,
                tool_id=tool_call.id,
                arguments=dict(tool_call.input),
            )

        results: list[ToolResult] = []
        for tool_call in sequential:
            execution = self._execute_tool(tool_call)
            self._emit_tool_result(execution, breaker, results)
            yield from self._events_for_tool_execution(execution)

        if parallel:
            context = copy_context()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(context.copy().run, self._execute_tool, tool_call): tool_call
                    for tool_call in parallel
                }
                for future in concurrent.futures.as_completed(futures):
                    execution = future.result()
                    self._emit_tool_result(execution, breaker, results)
                    yield from self._events_for_tool_execution(execution)

        return tuple(results)

    def _execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        try:
            raw_result = self.tool_repo.invoke_tool(tool_call.tool_name, dict(tool_call.input))
            if isinstance(raw_result, list):
                # Tools returning content blocks directly (e.g. imagegen_tool)
                result_content = raw_result
            elif isinstance(raw_result, dict):
                result_content = json.dumps(raw_result)
            else:
                result_content = str(raw_result)
            return ToolExecutionResult(
                tool_call,
                result_content,
                raw_result,
                self._tool_result_hash_material(raw_result, result_content),
                None,
            )
        except Exception as error:
            logger.error("Tool execution failed for %s: %s", tool_call.tool_name, error, exc_info=True)
            result_content = f"Error: {error}{self._schema_hint(tool_call.tool_name, error)}"
            return ToolExecutionResult(tool_call, result_content, None, None, error)

    def _tool_result_hash_material(
        self,
        raw_result: Any,
        result_content: str | list[dict[str, Any]],
    ) -> Any:
        return raw_result

    def _schema_hint(self, tool_name: str, error: Exception) -> str:
        error_text = str(error).lower()
        is_parameter_error = isinstance(error, ValueError) or any(
            keyword in error_text
            for keyword in ("unknown operation", "invalid", "required", "missing", "parameter")
        )
        if not is_parameter_error:
            return ""
        try:
            definition = self.tool_repo.get_tool_definition(tool_name)
            properties = dict(definition.input_schema.get("properties", {}))
        except (AttributeError, KeyError):
            return ""
        return f"\n\nCORRECT PARAMETERS:\n{json.dumps(properties, indent=2)}"

    def _emit_tool_result(
        self,
        execution: ToolExecutionResult,
        breaker: CircuitBreaker,
        results: list[ToolResult],
    ) -> None:
        breaker.record_execution(
            execution.tool_call.tool_name,
            execution.hash_material,
            execution.error,
        )
        results.append(ToolResult(
            tool_call_id=execution.tool_call.id,
            content=execution.result_content,
            is_error=execution.error is not None,
        ))

    def _events_for_tool_execution(
        self,
        execution: ToolExecutionResult,
    ) -> Iterator[StreamEvent]:
        if execution.error is not None:
            yield ToolErrorEvent(
                tool_name=execution.tool_call.tool_name,
                tool_id=execution.tool_call.id,
                error=str(execution.error),
                result=execution.result_content,
            )
            return
        yield ToolCompletedEvent(
            tool_name=execution.tool_call.tool_name,
            tool_id=execution.tool_call.id,
            result=execution.result_content,
        )
