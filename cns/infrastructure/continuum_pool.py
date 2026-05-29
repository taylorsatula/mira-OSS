"""
Continuum pool using Valkey for distributed caching.

Provides session detection and automatic expiration for continuums,
replacing the in-memory LRU pool with Valkey-based caching.
"""
from __future__ import annotations

import logging
import threading

from cns.core.continuum import Continuum
from cns.core.message import Message
from cns.infrastructure.continuum_repository import ContinuumRepository
from cns.infrastructure.valkey_message_cache import ValkeyMessageCache
from cns.core.segment_cache_loader import SegmentCacheLoader
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern for continuum operations.
    
    Accumulates changes during a continuum turn and commits them
    atomically to both database and cache.
    """
    
    def __init__(self, continuum: Continuum, pool: 'ContinuumPool'):
        """
        Initialize unit of work.

        Args:
            continuum: Continuum being modified
            pool: Parent continuum pool for persistence operations
        """
        self.continuum = continuum
        self.pool = pool
        self.pending_messages: list[Message] = []
        self.metadata_updated = False
        
    def add_messages(self, *messages: Message) -> None:
        """
        Queue messages for persistence.

        Enforces a per-message character limit as a safety net against
        oversized content bricking the conversation. Tool results have
        a tighter, format-aware limit upstream in the orchestrator.

        Args:
            *messages: One or more Message objects to persist
        """
        # Hard cap on any single message content before DB persistence.
        # Larger than tool result truncation (100k) since assistant messages
        # can legitimately be long.
        limit = 150_000

        for msg in messages:
            content = msg.content
            content_len = len(content) if isinstance(content, str) else len(str(content))
            if content_len > limit:
                truncated = (content if isinstance(content, str) else str(content))[:limit]
                truncated += f"\n\n[Message truncated: {content_len:,} chars exceeded {limit:,} char limit]"
                msg = Message(
                    id=msg.id,
                    content=truncated,
                    role=msg.role,
                    created_at=msg.created_at,
                    metadata=msg.metadata,
                    tool_call_id=msg.tool_call_id,
                    is_error=msg.is_error
                )
                logger.warning(
                    "Truncated oversized %s message at persistence: %d -> %d chars",
                    msg.role, content_len, len(truncated)
                )
            self.pending_messages.append(msg)
        
    def mark_metadata_updated(self) -> None:
        """Mark that continuum metadata needs to be updated."""
        self.metadata_updated = True
        
    def commit(self) -> None:
        """
        Persist all accumulated changes atomically.

        Saves messages to database, updates cache, and persists metadata changes.
        Segment creation now happens automatically in repository.save_message().
        """
        if self.pending_messages:
            # Batch save to database
            self.pool.repository.save_messages_batch(
                self.pending_messages,
                self.continuum.id,
                self.continuum.user_id
            )

            # Update Valkey cache once with current continuum state
            self.pool.valkey_cache.set_continuum(self.continuum.messages)

            logger.debug(f"Committed {len(self.pending_messages)} messages for continuum {self.continuum.id}")

        # Update metadata if needed
        if self.metadata_updated:
            self.pool.repository.update_continuum_metadata(self.continuum)
            logger.debug(f"Updated metadata for continuum {self.continuum.id}")

    def _get_real_messages(self) -> list[Message]:
        """
        Get conversation messages, excluding summaries and boundaries.

        Returns:
            List of actual conversation messages (user/assistant exchanges)
        """
        return [
            msg for msg in self.continuum.messages
            if not msg.metadata.get('system_notification')
            and not msg.metadata.get('is_segment_boundary')
        ]


class ContinuumPool:
    """
    Continuum pool backed by Valkey with TTL-based session management.
    
    Uses Valkey for distributed caching with automatic expiration,
    enabling clear session boundary detection when continuums expire.
    """
    
    def __init__(self, repository: ContinuumRepository,
                 session_loader: SegmentCacheLoader):
        """
        Initialize pool with repository and session loader.

        Args:
            repository: Repository for continuum persistence
            session_loader: Session cache loader for new sessions
        """
        self.repository = repository
        self.session_loader = session_loader
        self.valkey_cache = ValkeyMessageCache()
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        
    def get_or_create(self) -> Continuum:
        """
        Get continuum from Valkey cache or create new one.

        Checks Valkey first - if not found, it's a new session.
        Uses ambient user context from set_current_user_id().

        Returns:
            Continuum instance with appropriate cache
        """
        user_id = get_current_user_id()

        with self._lock:
            # Check Valkey cache first
            cached_messages = self.valkey_cache.get_continuum()

            # Get continuum structure from DB (must exist from signup)
            continuum = self.repository.get_continuum(user_id)
            if not continuum:
                raise RuntimeError(f"Continuum not found for user {user_id}. Continuum should be created during signup.")

            # No callback needed - using Unit of Work pattern

            if cached_messages is None:
                # NEW SESSION - continuum expired from Valkey
                logger.info(f"New session detected for user {user_id} - loading with session boundary")

                # Load session context (segment summaries + boundary)
                messages = self.session_loader.load_session_cache(
                    str(continuum.id), user_id
                )
                continuum.apply_cache(messages)

                # Cache in Valkey for future requests
                if messages:
                    self.valkey_cache.set_continuum(messages)

            else:
                # CONTINUING SESSION - cache hit
                logger.debug(f"Continuing session for user {user_id}")

                # Apply cached messages to continuum
                continuum.apply_cache(cached_messages)

            return continuum
    
    def begin_work(self, continuum: Continuum) -> UnitOfWork:
        """
        Begin a unit of work for continuum operations.

        Args:
            continuum: Continuum to track changes for

        Returns:
            UnitOfWork instance for accumulating and committing changes
        """
        return UnitOfWork(continuum, self)

    def invalidate(self) -> None:
        """
        Remove continuum from Valkey cache.

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            RuntimeError: If no user context is set
        """
        user_id = get_current_user_id()
        if self.valkey_cache.invalidate_continuum():
            logger.debug(f"Invalidated cached continuum for user {user_id}")
        else:
            logger.debug(f"No cached continuum to invalidate for user {user_id}")
    
    def update_cache(self, user_id: str, messages: list[Message]) -> None:
        """
        Update continuum cache in Valkey.

        Called when messages are added or modified.

        Args:
            user_id: User identifier
            messages: Updated message list
        """
        self.valkey_cache.set_continuum(messages)
        logger.debug(f"Updated continuum cache for user {user_id}")



# Global continuum pool instance
_continuum_pool: ContinuumPool | None = None


def initialize_continuum_pool(repository: ContinuumRepository,
                                session_loader: SegmentCacheLoader) -> ContinuumPool:
    """
    Initialize the global continuum pool with required dependencies.

    Must be called during application startup.

    Args:
        repository: Continuum repository
        session_loader: Session cache loader for new sessions

    Returns:
        Initialized ContinuumPool instance
    """
    global _continuum_pool
    _continuum_pool = ContinuumPool(repository, session_loader)
    logger.info("Continuum pool initialized with session cache loader")
    return _continuum_pool


def get_continuum_pool() -> ContinuumPool:
    """
    Get the global continuum pool instance.

    Raises:
        RuntimeError: If pool has not been initialized
    """
    global _continuum_pool
    if _continuum_pool is None:
        raise RuntimeError(
            "Continuum pool not initialized. Call initialize_continuum_pool() "
            "during application startup."
        )
    return _continuum_pool
