"""
Valkey-based message cache for continuum messages.

Provides distributed caching with event-driven invalidation via segment timeout.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from cns.core.message import Message, MessageMetadata
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)

CACHE_PATCH_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class ToolResultCachePatch:
    """Conditional replacement for one tool result in the hot cache."""

    message_id: UUID
    expected_content: str
    compacted_content: str


class ValkeyMessageCache:
    """
    Manages continuum message cache in Valkey.

    Cache invalidation is event-driven (triggered by segment timeout),
    not TTL-based. Cache miss indicates new session requiring boundary marker.
    """

    def __init__(self):
        """
        Initialize Valkey continuum cache.

        Cache invalidation is event-driven via segment timeout, not TTL-based.
        """
        from clients.valkey_client import get_valkey_client
        self.valkey = get_valkey_client()

        self.key_prefix = "continuum"

        logger.info("ValkeyMessageCache initialized (event-driven invalidation)")
    
    def _get_key(self, user_id: str) -> str:
        """Generate cache key for user continuum messages."""
        return f"{self.key_prefix}:{user_id}:messages"

    def _serialize_messages(self, messages: list[Message]) -> str:
        """
        Serialize messages to JSON for storage.

        Args:
            messages: List of Message objects

        Returns:
            JSON string representation
        """
        return json.dumps([msg.to_dict() for msg in messages])

    def _deserialize_messages(self, data: str) -> list[Message]:
        """
        Deserialize JSON data back to Message objects.

        Args:
            data: JSON string from Valkey

        Returns:
            List of Message objects
        """
        messages = []
        serialized = json.loads(data)

        for msg_dict in serialized:
            messages.append(Message.from_dict(msg_dict))

        return messages
    
    def get_continuum(self) -> list[Message] | None:
        """
        Get continuum messages from Valkey cache.

        Cache miss indicates a new session (invalidated by segment timeout).

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            List of messages if cached, None if not found in cache

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        key = self._get_key(user_id)
        data = self.valkey.get(key)

        if data:
            logger.debug(f"Found cached continuum for user {user_id}")
            return self._deserialize_messages(data)
        else:
            logger.debug(f"No cached continuum found for user {user_id}")
            return None

    def set_continuum(self, messages: list[Message]) -> None:
        """
        Store continuum messages in Valkey.

        Cache remains until explicitly invalidated by segment timeout handler.

        Args:
            messages: List of messages to cache

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        key = self._get_key(user_id)
        data = self._serialize_messages(messages)

        # Set without expiration - invalidation is event-driven
        self.valkey.set(key, data)

        logger.debug(f"Cached continuum for user {user_id}")

    def apply_tool_result_patches(
        self,
        patches: list[ToolResultCachePatch],
    ) -> int:
        """Atomically compact still-current tool messages by message ID."""
        if not patches:
            return 0

        user_id = get_current_user_id()
        key = self._get_key(user_id)
        patches_by_id = {patch.message_id: patch for patch in patches}

        for _attempt in range(CACHE_PATCH_MAX_ATTEMPTS):
            serialized = self.valkey.get(key)
            if serialized is None:
                logger.debug(
                    "Continuum cache missing while compacting tool results for user %s",
                    user_id,
                )
                return 0

            messages = self._deserialize_messages(serialized)
            updated_messages: list[Message] = []
            applied_count = 0

            for message in messages:
                patch = patches_by_id.get(message.id)
                if (
                    patch is None
                    or message.role != "tool"
                    or not isinstance(message.content, str)
                    or message.content != patch.expected_content
                    or message.metadata.get("tool_result_compacted")
                ):
                    updated_messages.append(message)
                    continue

                metadata = cast(MessageMetadata, dict(message.metadata))
                metadata["tool_result_compacted"] = True
                metadata["tool_result_original_chars"] = len(patch.expected_content)
                updated_messages.append(Message(
                    id=message.id,
                    content=patch.compacted_content,
                    role=message.role,
                    created_at=message.created_at,
                    metadata=metadata,
                    tool_call_id=message.tool_call_id,
                    is_error=message.is_error,
                ))
                applied_count += 1

            if applied_count == 0:
                return 0

            replacement = self._serialize_messages(updated_messages)
            if self.valkey.compare_and_set(key, serialized, replacement):
                logger.debug(
                    "Compacted %d cached tool result(s) for user %s",
                    applied_count,
                    user_id,
                )
                return applied_count

        raise RuntimeError(
            f"Continuum cache changed during {CACHE_PATCH_MAX_ATTEMPTS} "
            f"tool-result compaction attempts for user {user_id}"
        )

    def invalidate_continuum(self) -> bool:
        """
        Invalidate continuum cache entry.

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            True if cache entry was invalidated, False if entry didn't exist

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        messages_key = self._get_key(user_id)

        messages_result = self.valkey.delete(messages_key)

        if messages_result:
            logger.debug(f"Invalidated cached continuum for user {user_id}")

        return bool(messages_result)
