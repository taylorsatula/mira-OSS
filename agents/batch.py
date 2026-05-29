"""Batch transport for sidebar agents.

Submits a single LLM call to the Anthropic Batch API, polls for the
result, and returns the same neutral Result type as the sync path.
"""
import logging
import threading
import time
from typing import Any
from uuid import uuid4

import anthropic
from clients.llm.dialects.anthropic import anthropic_thinking_params, normalize_anthropic_message
from clients.llm.types import Result, ThinkingConfig, coerce_effort

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 30

_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        from clients.vault_client import get_api_key
        from utils.logging_config import instrument_anthropic_client

        _client = anthropic.Anthropic(
            api_key=get_api_key("anthropic_batch_key"),
            timeout=120.0,
        )
        instrument_anthropic_client(_client)
        logger.info("Batch transport client initialized")
        return _client


def batch_generate_response(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    system_prompt: str,
    llm_cfg: Any,
    timeout_seconds: int,
) -> Result:
    """Submit a single LLM call as a batch request. Blocks until result.

    Args:
        messages: Anthropic-format message list (accumulated by the agent loop).
        tool_schemas: Tool schemas from tool_repo.
        system_prompt: Agent system prompt.
        llm_cfg: Resolved InternalLLMConfig (from get_internal_llm()).
        timeout_seconds: Max seconds to wait for batch completion.

    Returns:
        Result — identical type to LLMProvider.generate_response().

    Raises:
        TimeoutError: Batch did not complete within timeout_seconds.
        RuntimeError: Batch request errored or expired on Anthropic's side.
    """
    client = _get_client()

    params: dict[str, Any] = {
        "model": llm_cfg.model,
        "max_tokens": llm_cfg.max_tokens,
        "system": [
            {"type": "text", "text": system_prompt}
        ],
        "messages": messages,
    }
    if llm_cfg.effort:
        thinking_params, _ = anthropic_thinking_params(
            model=llm_cfg.model,
            thinking=ThinkingConfig(effort=coerce_effort(llm_cfg.effort)),
        )
        params.update(thinking_params)

    if tool_schemas:
        tools = []
        for schema in tool_schemas:
            if hasattr(schema, "to_dict"):
                tool_dict = schema.to_dict()
            else:
                tool_dict = dict(schema)
            # Translate cache hint to Anthropic cache_control
            if "cache" in tool_dict:
                tool_dict["cache_control"] = {"type": "ephemeral", "ttl": tool_dict.pop("cache")}
            tools.append(tool_dict)
        params["tools"] = tools

    custom_id = uuid4().hex
    batch = client.beta.messages.batches.create(
        requests=[{"custom_id": custom_id, "params": params}]
    )
    logger.info(f"Submitted batch {batch.id} for {llm_cfg.model}")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)

        batch_info = client.beta.messages.batches.retrieve(batch.id)
        if batch_info.processing_status == "ended":
            return _extract_result(client, batch.id, custom_id)

    raise TimeoutError(
        f"Batch {batch.id} did not complete within {timeout_seconds}s"
    )


def _extract_result(
    client: anthropic.Anthropic,
    batch_id: str,
    custom_id: str,
) -> Result:
    """Stream batch results and return the Message for our custom_id."""
    for result in client.beta.messages.batches.results(batch_id):
        if result.custom_id != custom_id:
            continue

        if result.result.type == "succeeded":
            return normalize_anthropic_message(result.result.message)

        if result.result.type == "errored":
            error = result.result.error
            raise RuntimeError(
                f"Batch request errored: {error.type} — {error.message}"
            )

        if result.result.type == "expired":
            raise RuntimeError("Batch request expired")

    raise RuntimeError(f"No result found for custom_id {custom_id} in batch {batch_id}")
