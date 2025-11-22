"""
Valkey-based message cache for continuum messages.

Provides distributed caching with event-driven invalidation via segment timeout.
"""
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from cns.core.message import Message
from clients.valkey_client import ValkeyClient
from config import config
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class ValkeyMessageCache:
    """
    Manages continuum message cache in Valkey.

    Cache invalidation is event-driven (triggered by segment timeout),
    not TTL-based. Cache miss indicates new session requiring boundary marker.
    """

    def __init__(self, valkey_client: Optional[ValkeyClient] = None):
        """
        Initialize Valkey continuum cache.

        Cache invalidation is event-driven via segment timeout, not TTL-based.

        Args:
            valkey_client: Valkey client instance (creates one if not provided)
        """
        # Get or create Valkey client
        if valkey_client:
            self.valkey = valkey_client
        else:
            from clients.valkey_client import get_valkey_client
            self.valkey = get_valkey_client()

        self.key_prefix = "continuum"

        logger.info("ValkeyMessageCache initialized (event-driven invalidation)")
    
    def _get_key(self, user_id: str) -> str:
        """Generate cache key for user continuum messages."""
        return f"{self.key_prefix}:{user_id}:messages"

    def _get_thinking_budget_key(self, user_id: str) -> str:
        """Generate cache key for thinking budget preference."""
        return f"{self.key_prefix}:{user_id}:thinking_budget"
    
    def _serialize_messages(self, messages: List[Message]) -> str:
        """
        Serialize messages to JSON for storage.
        
        Args:
            messages: List of Message objects
            
        Returns:
            JSON string representation
        """
        serialized = []
        for msg in messages:
            msg_dict = {
                'id': str(msg.id),
                'content': msg.content,
                'role': msg.role,
                'created_at': msg.created_at.isoformat() if msg.created_at else None,
                'metadata': msg.metadata
            }
            serialized.append(msg_dict)
        
        return json.dumps(serialized)
    
    def _deserialize_messages(self, data: str) -> List[Message]:
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
            # Parse created_at if present
            created_at = None
            if msg_dict.get('created_at'):
                created_at = datetime.fromisoformat(msg_dict['created_at'])
            
            message = Message(
                id=msg_dict['id'],
                content=msg_dict['content'],
                role=msg_dict['role'],
                created_at=created_at,
                metadata=msg_dict.get('metadata', {})
            )
            messages.append(message)
        
        return messages
    
    def get_continuum(self) -> Optional[List[Message]]:
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

    def set_continuum(self, messages: List[Message]) -> None:
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

    def get_thinking_budget(self) -> Optional[int]:
        """
        Get thinking budget preference from Valkey cache.

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            Thinking budget value if cached, None if not found in cache

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        key = self._get_thinking_budget_key(user_id)
        data = self.valkey.get(key)

        if data:
            try:
                return int(data)
            except (ValueError, TypeError):
                logger.warning(f"Invalid thinking budget value in cache: {data}")
                return None

        return None

    def set_thinking_budget(self, budget: Optional[int]) -> None:
        """
        Store thinking budget preference in Valkey.

        Cache remains until explicitly invalidated by segment timeout handler.

        Args:
            budget: Thinking budget value (None, 0, or positive int)

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        key = self._get_thinking_budget_key(user_id)

        if budget is None:
            # Delete key if budget is None (use system default)
            self.valkey.delete(key)
            logger.debug(f"Cleared thinking budget for user {user_id}")
        else:
            # Set without expiration - invalidation is event-driven
            self.valkey.set(key, str(budget))
            logger.debug(f"Set thinking budget to {budget} for user {user_id}")

    def invalidate_continuum(self) -> bool:
        """
        Invalidate continuum cache entry and thinking budget preference.

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            True if cache entry was invalidated, False if entry didn't exist

        Raises:
            ValkeyError: If Valkey infrastructure is unavailable
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        messages_key = self._get_key(user_id)
        thinking_budget_key = self._get_thinking_budget_key(user_id)

        # Delete both keys
        messages_result = self.valkey.delete(messages_key)
        self.valkey.delete(thinking_budget_key)

        if messages_result:
            logger.debug(f"Invalidated cached continuum and thinking budget for user {user_id}")

        return bool(messages_result)
