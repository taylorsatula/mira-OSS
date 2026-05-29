"""
Valkey-based message cache for continuum messages.

Provides distributed caching with event-driven invalidation via segment timeout.
"""
from __future__ import annotations

import json
import logging

from cns.core.message import Message
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


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
