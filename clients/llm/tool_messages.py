"""Provider-neutral local tool continuation message assembly."""

from __future__ import annotations

from typing import Any

from clients.llm.types import Result, ToolResult


def assistant_message_from_result(
    result: Result,
    *,
    include_tool_call_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build the assistant message that records model-requested local tool calls.

    When include_tool_call_ids is provided, only tool calls whose IDs appear in
    the set are included. This prevents orphaned tool_call/tool-result pairs
    when some tool calls (e.g. server-side code_execution) don't have results.
    """
    tool_calls = (
        tuple(tc for tc in result.tool_calls if tc.id in include_tool_call_ids)
        if include_tool_call_ids is not None
        else result.tool_calls
    )

    content: list[dict[str, Any]] = []
    if result.reasoning:
        content.extend(result.reasoning.to_content_blocks())
    if result.text:
        content.append({"type": "text", "text": result.text})
    content.extend(tool_call.to_message_block() for tool_call in tool_calls)

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if result.reasoning:
        signatures = result.reasoning.to_signatures()
        if signatures:
            message["thinking_signatures"] = signatures
        if result.reasoning.provider_details:
            message["reasoning_details"] = list(result.reasoning.provider_details)
    return message


def tool_result_messages(tool_results: tuple[ToolResult, ...]) -> list[dict[str, Any]]:
    """Build individual role="tool" messages for completed local tool results.

    Each tool result becomes its own role="tool" message rather than
    being packed into a single user-role message.
    """
    return [
        {
            "role": "tool",
            "tool_call_id": tr.tool_call_id,
            "content": tr.content,
            **({"is_error": True} if tr.is_error else {}),
        }
        for tr in tool_results
    ]


def append_tool_result_messages(
    messages: list[dict[str, Any]],
    result: Result,
    tool_results: tuple[ToolResult, ...],
) -> list[dict[str, Any]]:
    """Append a provider-neutral local tool use/result pair to message history.

    Only includes tool calls that have matching results. Tool calls without
    results (e.g. server-side code_execution) are stripped from the assistant
    message to prevent provider 400 errors for orphaned tool_call/tool pairs.
    """
    result_ids = {tr.tool_call_id for tr in tool_results}
    assistant_msg = assistant_message_from_result(result, include_tool_call_ids=result_ids)

    return [
        *messages,
        assistant_msg,
        *tool_result_messages(tool_results),
    ]
