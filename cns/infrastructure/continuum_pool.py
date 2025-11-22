"""
Continuum pool using Valkey for distributed caching.

Provides session detection and automatic expiration for continuums,
replacing the in-memory LRU pool with Valkey-based caching.
"""
import logging
from typing import Dict, Optional, List, Any
from collections import OrderedDict
import threading

from cns.core.continuum import Continuum
from cns.core.message import Message
from cns.infrastructure.continuum_repository import ContinuumRepository
from cns.infrastructure.valkey_message_cache import ValkeyMessageCache
from cns.core.segment_cache_loader import SegmentCacheLoader
from config import config
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
        self.pending_messages = []
        self.metadata_updated = False
        
    def add_messages(self, *messages: Message) -> None:
        """
        Queue messages for persistence.
        
        Args:
            *messages: One or more Message objects to persist
        """
        self.pending_messages.extend(messages)
        
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

    def _get_real_messages(self) -> List[Message]:
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

            # Load thinking budget preference from Valkey (works for both new and continuing sessions)
            thinking_budget = self.valkey_cache.get_thinking_budget()
            if thinking_budget is not None:
                continuum._thinking_budget_preference = thinking_budget
                logger.debug(f"Loaded thinking budget {thinking_budget} for continuum {continuum.id}")

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
    
    def get_by_id(self, continuum_id: str, user_id: str) -> Optional[Continuum]:
        """
        Get continuum by ID, checking Valkey cache.
        
        Args:
            continuum_id: Continuum identifier
            user_id: User identifier for access verification
            
        Returns:
            Continuum instance or None if not found
        """
        with self._lock:
            # Load continuum from repository
            continuum = self.repository.get_by_id(continuum_id, user_id)

            if not continuum:
                return None

            # No callback needed - using Unit of Work pattern

            # Check Valkey for cached messages
            cached_messages = self.valkey_cache.get_continuum()

            if cached_messages:
                # Apply cached messages from cache
                continuum.apply_cache(cached_messages)
                logger.debug(f"Found cached messages for continuum {continuum_id}")
            else:
                # New session or cache expired
                logger.debug(f"No cached messages for continuum {continuum_id}")
                # Messages already loaded by repository

            return continuum
    
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
    
    def update_cache(self, user_id: str, messages: List[Message]) -> None:
        """
        Update continuum cache in Valkey.

        Called when messages are added or modified.

        Args:
            user_id: User identifier
            messages: Updated message list
        """
        self.valkey_cache.set_continuum(messages)
        logger.debug(f"Updated continuum cache for user {user_id}")

    def get_thinking_budget_preference(self) -> Optional[int]:
        """
        Get thinking budget preference from Valkey cache.

        Requires: Active user context (set via set_current_user_id during authentication)

        Returns:
            Thinking budget value if set, None for system default

        Raises:
            RuntimeError: If no user context is set
        """
        return self.valkey_cache.get_thinking_budget()

    def set_thinking_budget_preference(self, budget: Optional[int]) -> None:
        """
        Set thinking budget preference in Valkey cache.

        Persists until segment timeout invalidates the session.

        Args:
            budget: Thinking budget value (None, 0, or positive int)

        Requires: Active user context (set via set_current_user_id during authentication)

        Raises:
            RuntimeError: If no user context is set
        """
        self.valkey_cache.set_thinking_budget(budget)
        logger.info(f"Set thinking budget preference to {budget}")

    def get_session_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get session information for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with session info (cached: bool)
        """
        cached_messages = self.valkey_cache.get_continuum()
        return {
            'cached': cached_messages is not None
        }


# Global continuum pool instance
_continuum_pool: Optional[ContinuumPool] = None


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