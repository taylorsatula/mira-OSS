"""
Provider-neutral message formatting for CNS.

This module converts immutable Message objects into the neutral LLM request
shape. Continuum owns the message cache; this module owns serialization.
"""
from __future__ import annotations

from collections.abc import Sequence

from .message import Message


def format_messages_for_api(messages: Sequence[Message]) -> list[dict[str, object]]:
    """Format messages for the provider-neutral LLM API."""
    from cns.services.segment_helpers import format_segment_for_display, format_precis_for_display
    from utils.timezone_utils import convert_from_utc
    from utils.user_context import get_user_preferences

    user_tz = get_user_preferences().timezone
    formatted_messages: list[dict[str, object]] = []

    for message in messages:
        # --- Tool messages: emit directly with first-class fields ---
        if message.role == "tool":
            msg_dict: dict[str, object] = {"role": "tool", "content": message.content}
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            if message.is_error:
                msg_dict["is_error"] = message.is_error
            formatted_messages.append(msg_dict)
            continue

        content = message.content

        # Collapsed segments -> synthetic tool_call/tool pair.
        # Tool result framing gives summaries higher attention weight than
        # plain assistant messages; the model treats them as retrieved data.
        if (
            message.metadata.get("is_segment_boundary")
            and message.metadata.get("status") == "collapsed"
        ):
            display_mode = message.metadata.get("display_mode", "extended")
            if display_mode == "precis":
                summary_content = format_precis_for_display(message)
            elif display_mode == "extended":
                summary_content = format_segment_for_display(message)
            else:
                raise ValueError(
                    f"Unknown display_mode '{display_mode}' on segment "
                    f"{message.metadata.get('segment_id')}"
                )
            call_id = f"seg_{message.metadata['segment_id'][:22]}"

            formatted_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "id": call_id,
                    "name": "continuum_tool",
                    "input": {
                        "operation": "search",
                        "query": message.metadata["display_title"],
                    },
                }],
            })
            formatted_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": summary_content,
            })
            continue

        # Inject ephemeral timestamps for user/assistant messages (not persisted).
        # Skip timestamp injection for tool-call assistant messages: they are
        # structural, not conversational.
        if (
            message.role in ("user", "assistant")
            and not message.metadata.get("is_segment_boundary")
            and not message.metadata.get("system_notification")
            and not message.metadata.get("has_tool_calls")
        ):
            local_dt = convert_from_utc(message.created_at, user_tz)
            timestamp = local_dt.strftime("%-I:%M%p").lower()
            if isinstance(content, str):
                content = f"[{timestamp}] {content}"
            elif isinstance(content, list):
                content = [block.copy() for block in content]
                for block in content:
                    if block.get("type") == "text":
                        block["text"] = f"[{timestamp}] {block['text']}"
                        break

        # Normalize string content to block list for assistant messages.
        if message.role == "assistant" and isinstance(content, str):
            content = [{"type": "text", "text": content}]

        msg_dict = {"role": message.role, "content": content}
        if message.metadata.get("has_tool_calls"):
            msg_dict["has_tool_calls"] = True
        if message.metadata.get("thinking_signatures"):
            msg_dict["thinking_signatures"] = message.metadata["thinking_signatures"]
        if message.metadata.get("reasoning_details"):
            msg_dict["reasoning_details"] = message.metadata["reasoning_details"]
        formatted_messages.append(msg_dict)

    return formatted_messages
